import pygame
from widget import Widget


class Label(Widget):
    def __init__(self, window, x, y, width, height, text, label_color, text_color, font_name, font_size, line_space=20):
        super().__init__(window, x, y, width, height, text, label_color, text_color, font_name, font_size)
        self.line_space = line_space
        self.apply()

    def apply(self):
        self.text_image = []
        self.text_image_rect = []
        for text in self.text:
            text_image = self.font.render(text, True, self.text_color, self.back_ground_color)
            self.text_image.append(text_image)
            self.text_image_rect.append(text_image.get_rect())

        i = 1
        for rect in self.text_image_rect:
            rect.centerx = self.rect.centerx
            rect.centery = self.rect.top + self.line_space * i
            i += 1

        self.line = list(zip(self.text_image, self.text_image_rect))

    def set_text(self, text):
        self.text = text

    def get_text(self):
        return self.text

    def draw(self):
        self.window.fill(self.back_ground_color, self.rect)
        for image, rect in self.line:
            self.window.blit(image, rect)
