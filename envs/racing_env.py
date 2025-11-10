# envs/racing_env.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .grid_track import GridTrack, TILE_MURO
from .dynamics import DinamicaCoche
from .sensors import patch_egocentrico
from .rewards import Recompensa
from .renderer import Renderer

class RacingEnv(gym.Env):
    """Entorno de carrera para DQN (coche siempre horizontal mirando al ESTE)."""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, ruta_csv: str, patch_h: int = 11, patch_w: int = 11,
                 render_mode: str | None = None, renderer_ppu: int = 36, render_fps: int = 60):
        super().__init__()
        self.track = GridTrack.from_csv(ruta_csv)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.render_mode = render_mode

        # Dimensiones del coche (en unidades de grid)
        self.CAR_LARGO_X = 4.0
        self.CAR_ALTO_Y = 2.0

        # Estado continuo
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0

        # Referencia de centro vertical (aprox. mitad de la pista)
        self.y_centro = (self.track.alto - 1) / 2.0

        # Dinámica y recompensa
        self.dyn = DinamicaCoche(v_max=2.0, aceleracion=0.2, frenado=0.3)
        self.rew = Recompensa(k_progreso=1.0, k_tiempo=0.01, k_desvio=0.02, r_choque=5.0, r_meta=20.0)

        # Observación H×W×C con C=8 (incluye S y M)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.patch_h, self.patch_w, 8), dtype=np.float32
        )

        # Acción: (steer × throttle) = 3×3 = 9
        self.action_space = spaces.Discrete(9)

        # Renderizador
        self.render_fps = int(render_fps)
        self.renderer = Renderer(pix_por_unidad=renderer_ppu) if render_mode == "human" else None

        self._x_prev = 0.0  # para recompensa de progreso

    def _accion_a_tuplas(self, a: int) -> tuple[int, int]:
        steer_idx = a % 3       # 0=izq, 1=recto, 2=der
        throttle_idx = a // 3   # 0=frenar, 1=neutro, 2=acelerar
        return (steer_idx, throttle_idx)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Spawn derivado de 'S': trasera en borde izq de S y centro entre ambas S
        self.x, self.y = self.track.spawn_desde_salida(self.CAR_LARGO_X, self.CAR_ALTO_Y)
        self.v = 0.0
        self._x_prev = self.x
        obs = patch_egocentrico(self.track, self.x, self.y, self.patch_w, self.patch_h)
        return obs, {}

    def step(self, action: int):
        steer_idx, throttle_idx = self._accion_a_tuplas(action)

        # Tile bajo el centro (para dinámica)
        tile_y = int(np.clip(np.floor(self.y), 0, self.track.alto - 1))
        tile_x = int(np.clip(np.floor(self.x), 0, self.track.ancho - 1))
        tile_bajo = self.track.tile_en(tile_y, tile_x)

        # Actualizamos dinámica (mirando al ESTE)
        x_new, y_new, v_new = self.dyn.actualizar(self.x, self.y, self.v,
                                                  (steer_idx, throttle_idx), tile_bajo)

        # Clampear Y dentro del grid
        y_new = float(np.clip(y_new, 0.0 + self.CAR_ALTO_Y / 2.0, self.track.alto - self.CAR_ALTO_Y / 2.0))

        # Rect del coche
        x_min = x_new - self.CAR_LARGO_X / 2.0
        x_max = x_new + self.CAR_LARGO_X / 2.0
        y_min = y_new - self.CAR_ALTO_Y / 2.0
        y_max = y_new + self.CAR_ALTO_Y / 2.0

        # Choque únicamente con MURO
        choco = self.track.rect_toca_muro(x_min, y_min, x_max, y_max)
        # Llegó a META si toca casillas M
        llego_meta = self.track.rect_toca_meta(x_min, y_min, x_max, y_max)

        # Recompensa
        offset_centro = y_new - self.y_centro
        r = self.rew.paso(self._x_prev, x_new, offset_centro, choco, llego_meta)
        self._x_prev = x_new

        # Aplicar transición
        self.x, self.y, self.v = x_new, y_new, v_new

        # Observación
        obs = patch_egocentrico(self.track, self.x, self.y, self.patch_w, self.patch_h)

        terminated = bool(choco or llego_meta)
        truncated = False
        info = {"velocidad": self.v, "meta": llego_meta, "choque": choco}

        if self.render_mode == "human" and self.renderer is not None:
            self.render()

        return obs, r, terminated, truncated, info

    # Visual helpers
    def set_visual_speed_scale(self, escala: float):
        self.dyn.escala_tiempo = float(max(0.07, escala))

    def set_render_fps(self, fps: int):
        self.render_fps = int(max(1, fps))

    def render(self):
        if self.renderer is None:
            return
        self.renderer.draw(self.track.grid, self.x, self.y,
                           self.CAR_LARGO_X, self.CAR_ALTO_Y, fps=self.render_fps)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
