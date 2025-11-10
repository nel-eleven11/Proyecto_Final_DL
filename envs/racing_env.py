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
    """Entorno de carrera con orientación fija al ESTE (+X) y progreso normalizado hacia meta."""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, ruta_csv: str, patch_h: int = 13, patch_w: int = 13,
                 render_mode: str | None = None, renderer_ppu: int = 36, render_fps: int = 60):
        super().__init__()
        self.track = GridTrack.from_csv(ruta_csv)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.render_mode = render_mode

        # Dimensiones físicas del coche (en unidades de grid)
        self.CAR_LARGO_X = 4.0
        self.CAR_ALTO_Y = 2.0

        # Estado continuo
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0

        # Dinámica y recompensa
        self.dyn = DinamicaCoche(v_max=2.0, aceleracion=0.2, frenado=0.3)
        self.rew = Recompensa(k_progreso=1.0, k_tiempo=0.01, r_choque=5.0, r_meta=20.0)

        # Observación H×W×C (C=8 con S y M). El patch sigue “mirando” al Este.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.patch_h, self.patch_w, 8), dtype=np.float32
        )
        # Acción: (steer × throttle) = 3×3 = 9
        self.action_space = spaces.Discrete(9)

        # Render
        self.render_fps = int(render_fps)
        self.renderer = Renderer(pix_por_unidad=renderer_ppu) if render_mode == "human" else None

        # Meta
        self._centros_meta = self.track.centros_meta()
        self._dist_init = 1.0
        self._dist_prev = 1.0

    def _accion_a_tuplas(self, a: int) -> tuple[int, int]:
        steer_idx = a % 3       # 0=izq, 1=recto, 2=der
        throttle_idx = a // 3   # 0=frenar, 1=neutro, 2=acelerar
        return (steer_idx, throttle_idx)

    def _dist_a_meta(self, x: float, y: float) -> float:
        if not self._centros_meta:
            return 0.0
        dmin = 1e9
        for (xc, yc) in self._centros_meta:
            d = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
            if d < dmin:
                dmin = d
        return float(dmin)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Coloca el coche en la salida (mirando al Este)
        self.x, self.y = self.track.spawn_desde_salida(self.CAR_LARGO_X, self.CAR_ALTO_Y)
        self.v = 0.0

        # Progreso normalizado
        self._dist_init = self._dist_a_meta(self.x, self.y)
        self._dist_prev = self._dist_init
        self.rew.set_dist_inicial(self._dist_init)

        # Observación inicial (egocéntrica con “heading” fijo al Este => dir=0)
        obs = patch_egocentrico(self.track, self.x, self.y, dir_card=0,
                                ancho=self.patch_w, alto=self.patch_h, back_margin=3)
        return obs, {}

    def step(self, action: int):
        steer_idx, throttle_idx = self._accion_a_tuplas(action)

        # Tile bajo el centro (para dinámica)
        tile_y = int(np.clip(np.floor(self.y), 0, self.track.alto - 1))
        tile_x = int(np.clip(np.floor(self.x), 0, self.track.ancho - 1))
        tile_bajo = self.track.tile_en(tile_y, tile_x)

        # Dinámica (orientación fija al Este)
        x_new, y_new, v_new = self.dyn.actualizar(
            self.x, self.y, self.v, steer_idx, throttle_idx, tile_bajo
        )

        # Limitar a bordes verticales del grid
        y_new = float(np.clip(y_new, 0.0 + self.CAR_ALTO_Y / 2.0,
                              self.track.alto - self.CAR_ALTO_Y / 2.0))
        # Y también en X (por seguridad)
        x_new = float(np.clip(x_new, 0.0 + self.CAR_LARGO_X / 2.0,
                              self.track.ancho - self.CAR_LARGO_X / 2.0))

        # AABB del coche (fijo; no depende de orientación)
        x_min = x_new - self.CAR_LARGO_X / 2.0
        x_max = x_new + self.CAR_LARGO_X / 2.0
        y_min = y_new - self.CAR_ALTO_Y / 2.0
        y_max = y_new + self.CAR_ALTO_Y / 2.0

        # Eventos
        choco = self.track.rect_toca_muro(x_min, y_min, x_max, y_max)
        llego_meta = self.track.rect_toca_meta(x_min, y_min, x_max, y_max)

        # Recompensa por progreso hacia meta (normalizado)
        dist_act = self._dist_a_meta(x_new, y_new)
        r = self.rew.paso(self._dist_prev, dist_act, choco, llego_meta)
        self._dist_prev = dist_act

        # Aplicar transición
        self.x, self.y, self.v = x_new, y_new, v_new

        # Observación egocéntrica con “heading” fijo al Este (dir=0)
        obs = patch_egocentrico(self.track, self.x, self.y, dir_card=0,
                                ancho=self.patch_w, alto=self.patch_h, back_margin=3)

        terminated = bool(choco or llego_meta)
        truncated = False
        info = {"velocidad": self.v, "meta": llego_meta, "choque": choco}

        if self.render_mode == "human" and self.renderer is not None:
            self.render()
        return obs, r, terminated, truncated, info

    def set_visual_speed_scale(self, escala: float):
        """Ralentiza/acelera la animación (no afecta aprendizaje)."""
        self.dyn.escala_tiempo = float(max(0.05, escala))

    def set_render_fps(self, fps: int):
        self.render_fps = int(max(1, fps))

    def render(self):
        if self.renderer is None:
            return
        # Pasamos dir=0 (Este) para dibujar el coche
        self.renderer.draw(self.track.grid, self.x, self.y,
                           self.CAR_LARGO_X, self.CAR_ALTO_Y, dir_card=0, fps=self.render_fps)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
