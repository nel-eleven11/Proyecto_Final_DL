# envs/rewards.py
from __future__ import annotations
import numpy as np

class Recompensa:
    """Shaping de recompensas:
    - Progreso normalizado hacia meta (Δdist / dist_inicial)
    - Penalización por tiempo
    - Castigo por choque
    - Bonus por meta
    """
    def __init__(self, k_progreso: float = 1.0, k_tiempo: float = 0.01,
                 r_choque: float = 5.0, r_meta: float = 20.0):
        self.k_progreso = k_progreso
        self.k_tiempo = k_tiempo
        self.r_choque = r_choque
        self.r_meta = r_meta
        self.dist_inicial = 1.0  # se setea en reset

    def set_dist_inicial(self, d0: float):
        self.dist_inicial = max(1e-6, float(d0))

    def paso(self, dist_prev: float, dist_act: float, choco: bool, llego_meta: bool) -> float:
        # Progreso normalizado hacia meta 
        delta = (dist_prev - dist_act) / self.dist_inicial
        r = self.k_progreso * delta
        # Costo por tiempo/paso
        r -= self.k_tiempo
        # Choque
        if choco:
            r -= self.r_choque
        # Bonus por meta
        if llego_meta:
            r += self.r_meta
        return r
