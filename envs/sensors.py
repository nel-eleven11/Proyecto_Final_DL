# envs/sensors.py
from __future__ import annotations
import numpy as np
from .grid_track import GridTrack, TILE_AFUERAS

def patch_egocentrico(track: GridTrack, x_c: float, y_c: float, ancho: int, alto: int) -> np.ndarray:
    """Extrae un parche (alto × ancho) delante del coche (mirando hacia +X).
    Devuelve tensor one-hot en formato (H, W, C) con C=8 (incluye S y M).
    """
    offset_x = -2
    x0 = int(np.floor(x_c + offset_x))
    y0 = int(np.floor(y_c - alto // 2))

    patch = np.full((alto, ancho), fill_value=TILE_AFUERAS, dtype=np.int32)
    for dy in range(alto):
        for dx in range(ancho):
            xi = x0 + dx
            yi = y0 + dy
            patch[dy, dx] = track.tile_en(yi, xi)

    C = 8  
    oh = np.zeros((C, alto, ancho), dtype=np.float32)
    for c in range(C):
        oh[c] = (patch == c).astype(np.float32)
    # -> (H, W, C)
    return np.transpose(oh, (1, 2, 0)).astype(np.float32)
