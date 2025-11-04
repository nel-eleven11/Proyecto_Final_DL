# envs/grid_track.py
from __future__ import annotations
import numpy as np
import csv
from dataclasses import dataclass

# Definiciones de casillas:
# 0 -> pavimento (gris)
# 1 -> muro (amarillo)  (único elemento con el que se puede chocar)
# 2 -> afueras (verde)  (no afecta nada al coche)
# 3 -> aceite (negro)   (reduce velocidad 10%)
# 4 -> terracería       (reduce velocidad 5%)
# 5 -> boost (azul)     (aumenta temporalmente +5%)

TILE_PAVIMENTO = 0
TILE_MURO = 1
TILE_AFUERAS = 2
TILE_ACEITE = 3
TILE_TERRACERIA = 4
TILE_BOOST = 5

@dataclass
class GridTrack:
    """Carga y expone una pista desde un CSV.

    El CSV debe contener enteros en {0..5}. 
    Se asume que el coche inicia cerca del borde izquierdo sobre pavimento.
    """
    grid: np.ndarray  # (alto, ancho) con ints de tiles

    @classmethod
    def from_csv(cls, path: str) -> 'GridTrack':
        filas = []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if not row:
                    continue
                # limpiar espacios por si los hay
                fila = [int(x.strip()) for x in row if x is not None and x.strip() != ""]
                filas.append(fila)

        grid = np.array(filas, dtype=np.int32)
        assert grid.ndim == 2, "El CSV debe tener forma 2D (alto × ancho)"
        return cls(grid=grid)


    @property
    def alto(self) -> int:
        return int(self.grid.shape[0])

    @property
    def ancho(self) -> int:
        return int(self.grid.shape[1])

    def tile_en(self, y: int, x: int) -> int:
        """Devuelve el tipo de tile en (y, x). Si está fuera del grid, devuelve TILE_AFUERAS (2)."""
        if y < 0 or y >= self.alto or x < 0 or x >= self.ancho:
            return TILE_AFUERAS
        return int(self.grid[y, x])

    def rect_toca_muro(self, x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
        """Verifica si un rectángulo en coordenadas de unidad del grid toca alguna celda MURO.
        Aproximación: muestreamos celdas cubiertas por el rectángulo y buscamos TILE_MURO.
        """
        # Convertimos a celdas (enteros) con margen
        xi0 = int(np.floor(x_min))
        yi0 = int(np.floor(y_min))
        xi1 = int(np.ceil(x_max))
        yi1 = int(np.ceil(y_max))
        for yi in range(yi0, yi1+1):
            for xi in range(xi0, xi1+1):
                if self.tile_en(yi, xi) == TILE_MURO:
                    # Chequeo extra por si el rect no llega realmente a cubrir la celda completa,
                    # para simplicidad lo consideramos colisión si toca la celda.
                    return True
        return False