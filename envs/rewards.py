# envs/rewards.py
from __future__ import annotations
import numpy as np

class Recompensa:
    """Funciones de shaping para el entorno de carrera."""
    def __init__(self, k_progreso: float = 1.0, k_tiempo: float = 0.01, k_desvio: float = 0.02, r_choque: float = 5.0, r_meta: float = 20.0):
        self.k_progreso = k_progreso
        self.k_tiempo = k_tiempo
        self.k_desvio = k_desvio
        self.r_choque = r_choque
        self.r_meta = r_meta

    def paso(self, x_prev: float, x_act: float, offset_centro: float, choco: bool, llego_meta: bool) -> float:
        r = 0.0
        # Progreso hacia la derecha
        r += self.k_progreso * max(0.0, (x_act - x_prev))
        # Penalización por tiempo (cada paso cuesta)
        r -= self.k_tiempo
        # Penalización por desviarse del centro del carril (offset absoluto)
        r -= self.k_desvio * abs(offset_centro)
        # Choque contra muro
        if choco:
            r -= self.r_choque
        # Bonus por cruzar meta
        if llego_meta:
            r += self.r_meta
        return r