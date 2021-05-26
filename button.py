import pygame
from helper_func import *
from widget import Widget

class Button(Widget):

    def __init__(self, window, x, y, width, height, text, func, button_color, text_color, font_name, font_size):
        super().__init__(window, x, y, width, height, text, button_color, text_color, font_name, font_size)
        self.apply()
        self.func = func
        self.pressed = False

    def apply(self):
        self.text_image = self.font.render(self.text, True, self.text_color, self.back_ground_color)
        self.text_image_rect = self.text_image.get_rect()
        self.text_image_rect.center = self.rect.center

    def draw(self):
        amount = 0
        if(self.pressed):
            amount = 20

        self.window.fill(darker(self.back_ground_color, amount), self.rect)
        self.window.blit(darker_surface(self.text_image, amount), self.text_image_rect)

    def press(self):
        self.pressed = True

    def release(self):
        self.pressed = False

    def on_click(self, **kwargs):
        self.func(**kwargs)
