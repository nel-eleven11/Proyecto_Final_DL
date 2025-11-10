# envs/renderer.py
from __future__ import annotations
import pygame
import numpy as np
from .grid_track import (
    TILE_PAVIMENTO, TILE_MURO, TILE_AFUERAS, TILE_ACEITE, TILE_TERRACERIA, TILE_BOOST,
    TILE_SALIDA, TILE_META
)

# Colores
COLOR_PAV   = (128, 128, 128)
COLOR_MURO  = (255, 215, 0)
COLOR_AF    = (34, 139, 34)
COLOR_ACE   = (20, 20, 20)
COLOR_TERR  = (139, 90, 43)
COLOR_BOOST = (0, 191, 255)
COLOR_AUTO  = (220, 20, 60)
COLOR_FONDO = (25, 25, 35)
COLOR_BORDE = (60, 60, 70)
COLOR_BLANCO = (240, 240, 240)
COLOR_NEGRO  = (15, 15, 15)

def color_de_tile(t: int):
    return {
        TILE_PAVIMENTO: COLOR_PAV,
        TILE_MURO: COLOR_MURO,
        TILE_AFUERAS: COLOR_AF,
        TILE_ACEITE: COLOR_ACE,
        TILE_TERRACERIA: COLOR_TERR,
        TILE_BOOST: COLOR_BOOST,
        TILE_SALIDA: COLOR_BLANCO,  # S blanco
    }.get(t, (255, 255, 255))

class Renderer:
    def __init__(self, pix_por_unidad: int = 36):
        pygame.init()
        self.ppu = int(pix_por_unidad)
        self.screen = None
        self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _draw_meta_ajedrez(self, surface, rect: pygame.Rect):
        n = 4
        cw = rect.w // n
        ch = rect.h // n
        for iy in range(n):
            for ix in range(n):
                color = COLOR_BLANCO if (ix + iy) % 2 == 0 else COLOR_NEGRO
                sub = pygame.Rect(rect.x + ix * cw, rect.y + iy * ch, cw, ch)
                pygame.draw.rect(surface, color, sub)

    def draw(self, grid: np.ndarray, x: float, y: float,
             car_largo_x: float, car_alto_y: float, dir_card: int, fps: int = 60):
        alto, ancho = grid.shape
        W = ancho * self.ppu
        H = alto * self.ppu
        if self.screen is None:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("RL Racer (DQN)")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close(); return

        self.screen.fill(COLOR_FONDO)

        # Tiles
        for yi in range(alto):
            for xi in range(ancho):
                tile_type = int(grid[yi, xi])
                rect = pygame.Rect(xi*self.ppu, yi*self.ppu, self.ppu, self.ppu)
                if tile_type == TILE_META:
                    self._draw_meta_ajedrez(self.screen, rect)
                else:
                    pygame.draw.rect(self.screen, color_de_tile(tile_type), rect)
                if tile_type in (TILE_PAVIMENTO, TILE_MURO):
                    pygame.draw.rect(self.screen, COLOR_BORDE, rect, 1)

        # Coche
        x_px = x * self.ppu
        y_px = y * self.ppu
        w_px = car_largo_x * self.ppu
        h_px = car_alto_y * self.ppu

        # Ajuste por orientación: para N/S intercambiamos dimensiones
        if dir_card in (1, 3):  # S o N
            w_px, h_px = h_px, w_px

        rect_auto = pygame.Rect(int(x_px - w_px/2), int(y_px - h_px/2), int(w_px), int(h_px))
        pygame.draw.rect(self.screen, COLOR_AUTO, rect_auto, border_radius=8)

        # Parabrisas según frente: E(derecha), S(abajo), O(izquierda), N(arriba)
        if dir_card == 0:   # E
            parab = pygame.Rect(int(rect_auto.right - w_px*0.35), int(y_px - h_px*0.25),
                                int(w_px*0.25), int(h_px*0.5))
        elif dir_card == 2: # O
            parab = pygame.Rect(int(rect_auto.left + w_px*0.10), int(y_px - h_px*0.25),
                                int(w_px*0.25), int(h_px*0.5))
        elif dir_card == 1: # S
            parab = pygame.Rect(int(x_px - w_px*0.25), int(rect_auto.bottom - h_px*0.35),
                                int(w_px*0.5), int(h_px*0.25))
        else:               # N
            parab = pygame.Rect(int(x_px - w_px*0.25), int(rect_auto.top + h_px*0.10),
                                int(w_px*0.5), int(h_px*0.25))
        pygame.draw.rect(self.screen, (100, 120, 160), parab, border_radius=4)

        pygame.display.flip()
        self.clock.tick(fps)
