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
# 6 -> salida 'S' (blanco)
# 7 -> meta   'M' (ajedrez blanco/negro)

TILE_PAVIMENTO  = 0
TILE_MURO       = 1
TILE_AFUERAS    = 2
TILE_ACEITE     = 3
TILE_TERRACERIA = 4
TILE_BOOST      = 5
TILE_SALIDA     = 6
TILE_META       = 7

@dataclass
class GridTrack:
    """Carga y expone una pista desde un CSV que puede contener enteros o tokens 'S'/'M'."""
    grid: np.ndarray  # (alto, ancho) con ints de tiles

    @classmethod
    def from_csv(cls, path: str) -> 'GridTrack':
        filas = []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if not row:
                    continue
                fila = []
                for x in row:
                    tok = x.strip()
                    if tok.upper() == 'S':
                        fila.append(TILE_SALIDA)
                    elif tok.upper() == 'M':
                        fila.append(TILE_META)
                    else:
                        fila.append(int(tok))
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
        """¿El rectángulo toca algún MURO? (muestreo por celdas cubiertas)"""
        xi0 = int(np.floor(x_min))
        yi0 = int(np.floor(y_min))
        xi1 = int(np.ceil(x_max))
        yi1 = int(np.ceil(y_max))
        for yi in range(yi0, yi1 + 1):
            for xi in range(xi0, xi1 + 1):
                if self.tile_en(yi, xi) == TILE_MURO:
                    return True
        return False

    def rect_toca_meta(self, x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
        """¿El rectángulo toca alguna casilla de META?"""
        xi0 = int(np.floor(x_min))
        yi0 = int(np.floor(y_min))
        xi1 = int(np.ceil(x_max))
        yi1 = int(np.ceil(y_max))
        for yi in range(yi0, yi1 + 1):
            for xi in range(xi0, xi1 + 1):
                if self.tile_en(yi, xi) == TILE_META:
                    return True
        return False

    def centros_meta(self) -> list[tuple[float, float]]:
        """Devuelve centros (x+0.5, y+0.5) de todas las casillas META."""
        ys, xs = np.where(self.grid == TILE_META)
        return [(float(x) + 0.5, float(y) + 0.5) for y, x in zip(ys, xs)]

    def spawn_desde_salida(self, car_largo_x: float, car_alto_y: float) -> tuple[float, float]:
        """Calcula el (x,y) inicial:
        - Asume EXACTAMENTE 2 casillas 'S' apiladas verticalmente en la MISMA columna.
        - El coche mira al ESTE y su PARTE TRASERA se coloca en el borde izquierdo de esas 'S'.
        - El centro (y) del coche se fija a la mitad entre ambas 'S'.
        """
        ys, xs = np.where(self.grid == TILE_SALIDA)
        assert len(xs) == 2, "La pista debe contener exactamente 2 casillas 'S' de salida."
        # Verificar misma columna
        assert xs[0] == xs[1], "Las casillas 'S' deben estar en la misma columna."
        col = int(xs[0])
        y_min, y_max = int(min(ys)), int(max(ys))
        # Centro vertical entre ambas S 
        y_c = (y_min + y_max + 1) / 2.0
        # Parte trasera en el borde izquierdo de S => rear_x = col
        # Centro = rear_x + car_largo_x/2
        x_c = float(col) + car_largo_x / 2.0
        return x_c, y_c
