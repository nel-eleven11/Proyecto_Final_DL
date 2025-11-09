# envs/sensors.py
from __future__ import annotations
import numpy as np
from .grid_track import GridTrack, TILE_AFUERAS

def patch_egocentrico(track: GridTrack, x_c: float, y_c: float, ancho: int, alto: int) -> np.ndarray:
    """Extrae un parche (alto × ancho) delante del coche (mirando hacia +X).

    Simplificación: el parche está anclado con el coche en el centro vertical (y) y el borde izquierdo
    en x_c (posición del coche). Incluye también un poco "detrás" si ancho es mayor que el avance.

    Devuelve un tensor one-hot con canales=6 (0..5).
    """
    # Coordenadas enteras de celdas
    # Ubicamos la esquina sup izq del patch de modo que el coche esté en la columna 2 del parche
    # para ver más "hacia adelante" que hacia atrás.
    offset_x = -2
    x0 = int(np.floor(x_c + offset_x))
    y0 = int(np.floor(y_c - alto // 2))

    patch = np.full((alto, ancho), fill_value=TILE_AFUERAS, dtype=np.int32)
    for dy in range(alto):
        for dx in range(ancho):
            xi = x0 + dx
            yi = y0 + dy
            patch[dy, dx] = track.tile_en(yi, xi)

    # One-hot 6 canales
    C = 6
    """
    
    oh = np.zeros((C, alto, ancho), dtype=np.float32)
    for c in range(C):
        oh[c] = (patch == c).astype(np.float32)
    return oh """

    oh = np.zeros((6, alto, ancho), dtype=np.float32)
    for c in range(6):
        oh[c] = (patch == c).astype(np.float32)
    oh_hwc = np.transpose(oh, (1, 2, 0)).astype(np.float32)  # -> (H, W, C)
    return oh_hwc
