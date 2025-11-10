# envs/dynamics.py
from __future__ import annotations
from .grid_track import TILE_ACEITE, TILE_TERRACERIA, TILE_BOOST

class DinamicaCoche:
    """Actualiza el estado del coche (x, y, v) con orientación fija al ESTE (+X).

    - Las acciones discretas controlan aceleración/frenado y desplazamiento lateral.
    - La velocidad es en unidades de grid por paso.
    - Avance siempre hacia +X (no hay marcha atrás).
    """
    def __init__(self, v_max: float = 2.0, aceleracion: float = 0.1, frenado: float = 0.3):
        self.v_max = float(v_max)
        self.acel = float(aceleracion)
        self.freno = float(frenado)
        self.boost_contador = 0
        self.escala_tiempo = 1.0  # para ralentizar/accelerar SOLO la visualización

    def aplicar_superficie(self, tile_bajo_centro: int, v: float) -> float:
        """Aplica efecto de superficie sobre la velocidad escalar (solo multiplicativo)."""
        if tile_bajo_centro == TILE_ACEITE:
            v *= 0.90
        elif tile_bajo_centro == TILE_TERRACERIA:
            v *= 0.95
        # BOOST se maneja con contador (estado)
        return v

    def actualizar(
        self,
        x: float,
        y: float,
        v: float,
        steer_idx: int,
        throttle_idx: int,
        tile_bajo_centro: int
    ) -> tuple[float, float, float]:
        """Devuelve (x_nuevo, y_nuevo, v_nueva).

        - steer_idx: 0=izq, 1=recto, 2=der  -> desplaza y en {-1,0,+1}
        - throttle_idx: 0=frenar, 1=neutro, 2=acelerar
        - Avance SIEMPRE en +X con la velocidad actual.
        """
        # Actualiza velocidad según throttle
        if throttle_idx == 2:       # acelerar
            v = min(self.v_max, v + self.acel * self.escala_tiempo)
        elif throttle_idx == 0:     # frenar
            v = max(0.0, v - self.freno * self.escala_tiempo)
        # neutro => sin cambio

        # Boost temporal (si activo)
        if self.boost_contador > 0:
            v = min(self.v_max, v * (1.05 ** self.escala_tiempo))
            self.boost_contador -= 1

        # Efecto de superficie
        v = self.aplicar_superficie(tile_bajo_centro, v)

        # Si estamos sobre BOOST, activamos 10 pasos de boost
        if tile_bajo_centro == TILE_BOOST:
            self.boost_contador = max(self.boost_contador, 10)

        # Avance longitudinal fijo al Este
        x_nuevo = x + v * self.escala_tiempo

        # Desplazamiento lateral discreto
        if steer_idx == 0:
            y_nuevo = y - 1.0
        elif steer_idx == 2:
            y_nuevo = y + 1.0
        else:
            y_nuevo = y

        return x_nuevo, y_nuevo, v
