# envs/dynamics.py
from __future__ import annotations
from .grid_track import TILE_ACEITE, TILE_TERRACERIA, TILE_BOOST

# Dir cardenal: 0=E, 1=S, 2=O, 3=N
def _avance_por_dir(dir_card: int, v: float) -> tuple[float, float]:
    if dir_card == 0:   # Este
        return v, 0.0
    if dir_card == 1:   # Sur
        return 0.0, v
    if dir_card == 2:   # Oeste
        return -v, 0.0
    # Norte
    return 0.0, -v

class DinamicaCoche:
    """Actualiza el estado (x, y, v) avanzando en la dirección cardenal actual."""
    def __init__(self, v_max: float = 2.0, aceleracion: float = 0.1, frenado: float = 0.3):
        self.v_max = float(v_max)
        self.acel = float(aceleracion)
        self.freno = float(frenado)
        self.boost_contador = 0
        self.escala_tiempo = 1.0

    def aplicar_superficie(self, tile_bajo_centro: int, v: float) -> float:
        if tile_bajo_centro == TILE_ACEITE:
            v *= 0.90
        elif tile_bajo_centro == TILE_TERRACERIA:
            v *= 0.95
        return v

    def actualizar(self, x: float, y: float, v: float,
                   throttle_idx: int, tile_bajo_centro: int, dir_card: int) -> tuple[float, float, float]:
        """Devuelve (x_nuevo, y_nuevo, v_nueva), avanzando en 'dir_card'."""
        # Velocidad
        if throttle_idx == 2:       # acelerar
            v = min(self.v_max, v + self.acel * self.escala_tiempo)
        elif throttle_idx == 0:     # frenar
            v = max(0.0, v - self.freno * self.escala_tiempo)

        if self.boost_contador > 0:
            v = min(self.v_max, v * (1.05 ** self.escala_tiempo))
            self.boost_contador -= 1

        v = self.aplicar_superficie(tile_bajo_centro, v)
        if tile_bajo_centro == TILE_BOOST:
            self.boost_contador = max(self.boost_contador, 10)

        # Avance según heading
        dx, dy = _avance_por_dir(dir_card, v * self.escala_tiempo)
        return x + dx, y + dy, v
