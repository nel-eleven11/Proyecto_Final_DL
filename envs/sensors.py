# envs/sensors.py
from __future__ import annotations
import numpy as np
from .grid_track import GridTrack, TILE_AFUERAS

# 0=E, 1=S, 2=O, 3=N
def _rotar_local_a_mundo(forward: int, lateral: int, dir_card: int) -> tuple[int, int]:
    # E: (f, l) -> (dx=f, dy=l)
    if dir_card == 0:
        return forward, lateral
    # S: (f, l) -> (dx=-l, dy=f)
    if dir_card == 1:
        return -lateral, forward
    # O: (f, l) -> (dx=-f, dy=-l)
    if dir_card == 2:
        return -forward, -lateral
    # N: (f, l) -> (dx=l, dy=-f)
    return lateral, -forward

def patch_egocentrico(track: GridTrack, x_c: float, y_c: float,
                      dir_card: int, ancho: int, alto: int, back_margin: int = 3) -> np.ndarray:
    """Extrae parche (alto x ancho) egocéntrico orientado por 'dir_card'.
    'back_margin' celdas hacia atrás y el resto hacia adelante.
    Devuelve one-hot (H, W, C) con C=8 (incluye S y M)."""
    patch = np.full((alto, ancho), fill_value=TILE_AFUERAS, dtype=np.int32)
    for i in range(alto):
        forward = i - back_margin
        for j in range(ancho):
            lateral = j - (ancho // 2)
            dx, dy = _rotar_local_a_mundo(forward, lateral, dir_card)
            xi = int(np.floor(x_c + dx))
            yi = int(np.floor(y_c + dy))
            patch[i, j] = track.tile_en(yi, xi)

    C = 8
    oh = np.zeros((C, alto, ancho), dtype=np.float32)
    for c in range(C):
        oh[c] = (patch == c).astype(np.float32)
    return np.transpose(oh, (1, 2, 0)).astype(np.float32)  # (H, W, C)
