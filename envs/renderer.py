# envs/renderer.py
from __future__ import annotations
import pygame
import numpy as np
from .grid_track import TILE_PAVIMENTO, TILE_MURO, TILE_AFUERAS, TILE_ACEITE, TILE_TERRACERIA, TILE_BOOST

# Colores (R, G, B)
COLOR_PAV = (128, 128, 128)   # gris
COLOR_MURO = (255, 215, 0)    # amarillo dorado
COLOR_AF = (0, 128, 0)        # verde
COLOR_ACE = (0, 0, 0)         # negro
COLOR_TERR = (139, 69, 19)    # café
COLOR_BOOST = (30, 144, 255)  # azul
COLOR_AUTO = (220, 20, 60)    # rojo

def color_de_tile(t: int):
    return {
        TILE_PAVIMENTO: COLOR_PAV,
        TILE_MURO: COLOR_MURO,
        TILE_AFUERAS: COLOR_AF,
        TILE_ACEITE: COLOR_ACE,
        TILE_TERRACERIA: COLOR_TERR,
        TILE_BOOST: COLOR_BOOST,
    }.get(t, (255, 255, 255))

class Renderer:
    def __init__(self, pix_por_unidad: int = 20):
        pygame.init()
        self.ppu = int(pix_por_unidad)
        self.screen = None
        self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def draw(self, grid: np.ndarray, x: float, y: float, car_largo_x: float, car_alto_y: float, fps: int = 60):
        alto, ancho = grid.shape
        W = ancho * self.ppu
        H = alto * self.ppu
        if self.screen is None:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("RL Racer (DQN)")

        # Eventos para poder cerrar ventana con ESC
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                return

        # Fondo
        self.screen.fill((40, 40, 40))

        # Dibujar grid
        for yi in range(alto):
            for xi in range(ancho):
                color = color_de_tile(int(grid[yi, xi]))
                rect = pygame.Rect(xi*self.ppu, yi*self.ppu, self.ppu, self.ppu)
                pygame.draw.rect(self.screen, color, rect)

        # Dibujar auto como rectángulo
        x_px = x * self.ppu
        y_px = y * self.ppu
        w_px = car_largo_x * self.ppu
        h_px = car_alto_y * self.ppu
        # Convertir centro (x,y) a esquina sup izq
        rect_auto = pygame.Rect(int(x_px - w_px/2), int(y_px - h_px/2), int(w_px), int(h_px))
        pygame.draw.rect(self.screen, COLOR_AUTO, rect_auto, border_radius=6)

        pygame.display.flip()
        self.clock.tick(fps)