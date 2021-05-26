import pygame


class Widget():
    def __init__(self, window, x, y, width, height, text, back_ground_color=(255, 255, 255), text_color=(0, 0, 0), font_name='Arial', font_size=24):
        self.window = window
        self.window_rect = window.get_rect()
        self.width, self.height = width, height
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.text = text
        self.set_color(back_ground_color, text_color)
        self.set_font(font_name, font_size)

    def set_color(self, button_color, text_color):
        self.back_ground_color = button_color
        self.text_color = text_color

    def set_font(self, font_name, font_size):
        self.font = pygame.font.SysFont(font_name, font_size)

    def draw(self):
        pass

    def apply(self):
        pass
