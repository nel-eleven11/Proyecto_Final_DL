# envs/dynamics.py
from __future__ import annotations
import numpy as np
from .grid_track import TILE_ACEITE, TILE_TERRACERIA, TILE_BOOST, TILE_PAVIMENTO

class DinamicaCoche:
    """Actualiza el estado del coche (x, y, v) en el grid.

    Supuestos:
    - El coche siempre apunta hacia +X (derecha). Sin rotación para simplificar DQN.
    - Las acciones discretas controlan aceleración y desplazamiento lateral.
    - La velocidad es en unidades de grid por paso.
    """
    def __init__(self, v_max: float = 2.0, aceleracion: float = 0.1, frenado: float = 0.3):
        self.v_max = float(v_max)
        self.acel = float(aceleracion)
        self.freno = float(frenado)
        self.boost_contador = 0
        self.escala_tiempo = 1.0

    def aplicar_superficie(self, tile_bajo_centro: int, v: float) -> float:
        """Aplica efecto de superficie sobre la velocidad escalar (solo multiplicativo).
        - Aceite: -10%
        - Terracería: -5%
        - Boost: +5% temporal (se maneja con un contador aparte)
        """
        if tile_bajo_centro == TILE_ACEITE:
            v *= 0.90
        elif tile_bajo_centro == TILE_TERRACERIA:
            v *= 0.95
        # Boost se maneja como estado: si sobre BOOST, encendemos el contador
        return v

    def actualizar(self, x: float, y: float, v: float, accion: tuple[int, int], tile_bajo_centro: int) -> tuple[float, float, float]:
        """Devuelve (x_nuevo, y_nuevo, v_nueva).

        accion = (steering_idx, throttle_idx) con:
        - steering_idx: 0=izq, 1=recto, 2=der
        - throttle_idx: 0=frenar, 1=neutro, 2=acelerar
        """
        steer, th = accion

        # Actualiza velocidad según throttle
        if th == 2:       # acelerar
            v = min(self.v_max, v + self.acel * self.escala_tiempo)
        elif th == 0:     # frenar
            v = max(0.0, v - self.freno * self.escala_tiempo)
        # th == 1: neutro (sin cambio)

        # Boost temporal (si activo)
        if self.boost_contador > 0:
            v = min(self.v_max, v * (1.05 ** self.escala_tiempo)) 
            self.boost_contador -= 1

        # Efecto instantáneo de superficie (aceite/terracería)
        v = self.aplicar_superficie(tile_bajo_centro, v)

        # Si estamos sobre BOOST, activamos 10 pasos de boost
        if tile_bajo_centro == TILE_BOOST:
            self.boost_contador = max(self.boost_contador, 10)

        # Avance longitudinal en X
        x_nuevo = x + v * self.escala_tiempo  # Δt=1 por paso

        # Desplazamiento lateral discreto
        if steer == 0:
            y_nuevo = y - 1.0
        elif steer == 2:
            y_nuevo = y + 1.0
        else:
            y_nuevo = y

        return x_nuevo, y_nuevo, v