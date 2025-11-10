# envs/renderer.py
from __future__ import annotations
import pygame
import numpy as np
from .grid_track import TILE_PAVIMENTO, TILE_MURO, TILE_AFUERAS, TILE_ACEITE, TILE_TERRACERIA, TILE_BOOST

# Colores (R, G, B)
COLOR_PAV = (128, 128, 128)   # gris
COLOR_MURO = (255, 215, 0)    # amarillo dorado
COLOR_AF = (34, 139, 34)      # verde
COLOR_ACE = (20, 20, 20)      # negro 
COLOR_TERR = (139, 90, 43)    # café
COLOR_BOOST = (0, 191, 255)   # azul
COLOR_AUTO = (220, 20, 60)    # rojo
COLOR_FONDO = (25, 25, 35)    # fondo oscuro azulado
COLOR_BORDE = (60, 60, 70)    # borde de tiles
COLOR_BOTON_X = (200, 50, 50) # rojo para la X
COLOR_BOTON_HOVER = (255, 80, 80) # rojo más claro al pasar mouse

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
    def __init__(self, pix_por_unidad: int = 36):
        pygame.init()
        self.ppu = int(pix_por_unidad)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.boton_x_rect = None
        self.mouse_sobre_x = False

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def draw(self, grid: np.ndarray, x: float, y: float, car_largo_x: float, car_alto_y: float, fps: int = 60):
        alto, ancho = grid.shape
        W = ancho * self.ppu
        H = alto * self.ppu
        
        HUD_HEIGHT = 50
        WINDOW_HEIGHT = H + HUD_HEIGHT
        
        if self.screen is None:
            self.screen = pygame.display.set_mode((W, WINDOW_HEIGHT))
            pygame.display.set_caption("RL Racer (DQN)")
            icon = pygame.Surface((32, 32))
            icon.fill(COLOR_AUTO)
            pygame.display.set_icon(icon)

        mouse_pos = pygame.mouse.get_pos()
        
        boton_size = 40
        self.boton_x_rect = pygame.Rect(W - boton_size - 5, 5, boton_size, boton_size)
        self.mouse_sobre_x = self.boton_x_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.mouse_sobre_x:
                    self.close()
                    return

        # Fondo
        self.screen.fill(COLOR_FONDO)

        # Dibujar HUD
        pygame.draw.rect(self.screen, (40, 40, 50), pygame.Rect(0, 0, W, HUD_HEIGHT))

        color_x = COLOR_BOTON_HOVER if self.mouse_sobre_x else COLOR_BOTON_X
        pygame.draw.rect(self.screen, color_x, self.boton_x_rect, border_radius=8)
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (self.boton_x_rect.left + 12, self.boton_x_rect.top + 12),
                        (self.boton_x_rect.right - 12, self.boton_x_rect.bottom - 12), 3)
        pygame.draw.line(self.screen, (255, 255, 255),
                        (self.boton_x_rect.right - 12, self.boton_x_rect.top + 12),
                        (self.boton_x_rect.left + 12, self.boton_x_rect.bottom - 12), 3)

        for yi in range(alto):
            for xi in range(ancho):
                tile_type = int(grid[yi, xi])
                color = color_de_tile(tile_type)
                
                rect = pygame.Rect(xi*self.ppu, yi*self.ppu + HUD_HEIGHT, self.ppu, self.ppu)
                
                pygame.draw.rect(self.screen, color, rect)

                if tile_type == TILE_PAVIMENTO or tile_type == TILE_MURO:
                    pygame.draw.rect(self.screen, COLOR_BORDE, rect, 1)
                if tile_type == TILE_BOOST:
                    for offset in range(3):
                        start_x = xi * self.ppu + 5 + offset * 8
                        pygame.draw.line(self.screen, (255, 255, 255),
                                       (start_x, yi*self.ppu + HUD_HEIGHT + self.ppu//3),
                                       (start_x + 6, yi*self.ppu + HUD_HEIGHT + self.ppu//3), 2)
                        pygame.draw.line(self.screen, (255, 255, 255),
                                       (start_x, yi*self.ppu + HUD_HEIGHT + 2*self.ppu//3),
                                       (start_x + 6, yi*self.ppu + HUD_HEIGHT + 2*self.ppu//3), 2)
                
                elif tile_type == TILE_ACEITE:
                    center_x = xi * self.ppu + self.ppu // 2
                    center_y = yi * self.ppu + HUD_HEIGHT + self.ppu // 2
                    pygame.draw.circle(self.screen, (40, 40, 40), (center_x, center_y), 8)
                    pygame.draw.circle(self.screen, (60, 60, 60), (center_x - 3, center_y - 3), 4)

        x_px = x * self.ppu
        y_px = (y * self.ppu) + HUD_HEIGHT
        w_px = car_largo_x * self.ppu
        h_px = car_alto_y * self.ppu

        sombra_offset = 4
        rect_sombra = pygame.Rect(int(x_px - w_px/2 + sombra_offset), 
                                 int(y_px - h_px/2 + sombra_offset), 
                                 int(w_px), int(h_px))
        pygame.draw.rect(self.screen, (0, 0, 0, 128), rect_sombra, border_radius=8)
        

        rect_auto = pygame.Rect(int(x_px - w_px/2), int(y_px - h_px/2), int(w_px), int(h_px))
        pygame.draw.rect(self.screen, COLOR_AUTO, rect_auto, border_radius=8)
        
        ventana_color = (100, 100, 120)
        ventana_rect = pygame.Rect(int(x_px - w_px/4), int(y_px - h_px/4), 
                                   int(w_px/2), int(h_px/2))
        pygame.draw.rect(self.screen, ventana_color, ventana_rect, border_radius=4)
        

        pygame.draw.rect(self.screen, (255, 255, 255), rect_auto, 2, border_radius=8)

        pygame.display.flip()
        self.clock.tick(fps)