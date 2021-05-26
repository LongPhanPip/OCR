import pygame
import sys
from window import Window
from button import Button
from label import Label
import numpy as np
from helper_func import *
import cv2


class Paint():
    def __init__(self):
        pygame.init()

        self.window_init()
        self.image_init()
        self.mouse_init()
        self.widget_init()

    def window_init(self):
        width = 750
        height = int(12 / 16 * width)
        back_ground_color = (29, 191, 134)
        title = 'AI'
        self.is_run = True

        self.display = Window(width, height, back_ground_color, title)

        self.window = pygame.display.set_mode((self.display.width, self.display.height))
        pygame.display.set_caption(self.display.title)

    def image_init(self):
        self.create_draw_surface()

    def create_draw_surface(self):
        self.draw_surface_width = self.draw_surface_height = 400
        self.pixels_per_cell = int(400 / 20)
        self.draw_surface = pygame.Surface((self.draw_surface_width, self.draw_surface_height))
        pixelarray = pygame.PixelArray(self.draw_surface)
        pixelarray[:] = (255, 255, 255)

    def mouse_init(self):
        self.x_old = 0
        self.y_old = 0
        self.blur_amount = 50
        self.last_time = pygame.time.get_ticks()

    def widget_init(self):
        button_width = 120
        button_height = 50
        label_width = 120
        label_height = 50
        self.save_button = Button(self.window, 450, 50, button_width, button_height, 'Save', save_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))
        self.clear_button = Button(self.window, 600, 50, button_width, button_height, 'Clear', clear_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))
        self.train_button = Button(self.window, 450, 150, button_width, button_height, 'Train', train_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))
        self.predict_button = Button(self.window, 600, 150, button_width, button_height, 'Predict', predict_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))
        self.plot_button = Button(self.window, 525, 250, button_width, button_height, 'Plot', plot_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))

        self.true_button = Button(self.window, 450, 475, button_width, button_height, 'True', true_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))

        self.false_button = Button(self.window, 600, 475, button_width, button_height, 'False', false_func, font_name='Futura', font_size=30, text_color=(255, 255, 255), button_color=(54, 52, 52))

        self.button_group = []
        self.button_group.append(self.save_button)
        self.button_group.append(self.clear_button)
        self.button_group.append(self.train_button)
        self.button_group.append(self.predict_button)
        self.button_group.append(self.plot_button)
        self.button_group.append(self.true_button)
        self.button_group.append(self.false_button)

        self.label = Label(self.window, 50, 475, label_width, label_height, ['Label:'], font_name='Futura', font_size=30, label_color=self.display.back_ground_color, text_color=(255, 255, 255))
        self.num_label = Label(self.window, 150, 475, 50, label_height, [''], font_name='Futura', font_size=30, label_color=self.display.back_ground_color, text_color=(255, 255, 255))
        self.warning_label = Label(self.window, 485, 350, 200, label_height * 2, [''], font_name='Futura', font_size=30, label_color=self.display.back_ground_color, text_color=(33, 33, 33))

        self.label_group = []
        self.label_group.append(self.label)
        self.label_group.append(self.num_label)
        self.label_group.append(self.warning_label)

    def check_insurface(self):
        pos = pygame.mouse.get_pos()

        return (pos[0] >= 0 and pos[1] >= 0 and pos[0] < self.draw_surface.get_rect().right and pos[1] < self.draw_surface.get_rect().bottom)

    def set_color(self, draw_surface, x, y, amount):
        color = np.array(draw_surface.get_at((x * self.pixels_per_cell, y * self.pixels_per_cell)), dtype=np.int64)
        color -= [amount, amount, amount, 0]
        if((color < 0).any()):
            color = [0, 0, 0, 255]

        for i in range(self.pixels_per_cell):
            for j in range(self.pixels_per_cell):
                draw_surface.set_at((x * self.pixels_per_cell + i, y * self.pixels_per_cell + j), color)

    def blur_effect(self, draw_surface, x, y):
        self.set_color(draw_surface, x, y, 255)
        self.set_color(draw_surface, np.max([x - 1, 0]), y, self.blur_amount * np.random.randint(1, 4))
        self.set_color(draw_surface, np.min([x + 1, self.pixels_per_cell - 1]), y, self.blur_amount * np.random.randint(1, 4))
        self.set_color(draw_surface, x, np.max([y - 1, 0]), self.blur_amount * np.random.randint(1, 4))
        self.set_color(draw_surface, x, np.min([y + 1, self.pixels_per_cell - 1]), self.blur_amount * np.random.randint(1, 4))
        self.set_color(draw_surface, np.max([x - 1, 0]), np.min([y + 1, self.pixels_per_cell - 1]), self.blur_amount * np.random.randint(0, 3))
        self.set_color(draw_surface, np.min([x + 1, self.pixels_per_cell - 1]), np.max([y - 1, 0]), self.blur_amount * np.random.randint(0, 3))
        self.set_color(draw_surface, np.max([x - 1, 0]), np.max([y - 1, 0]), self.blur_amount * np.random.randint(0, 3))
        self.set_color(draw_surface, np.min([x + 1, self.pixels_per_cell - 1]), np.min([y + 1, self.pixels_per_cell - 1]), self.blur_amount * np.random.randint(0, 3))

    def paint(self):
        button = pygame.mouse.get_pressed()
        if(button[0] == 1 and self.check_insurface()):
            pos = pygame.mouse.get_pos()
            x_pos = int(pos[0] / self.pixels_per_cell)
            y_pos = int(pos[1] / self.pixels_per_cell)

            self.blur_effect(self.draw_surface, x_pos, y_pos)

            if self.x_old != x_pos or self.y_old != y_pos:
                if (pygame.time.get_ticks() - self.last_time) < 50:

                    if self.x_old == x_pos:
                        if(self.y_old < y_pos):
                            y_b = self.y_old
                            y_e = y_pos
                        else:
                            y_b = y_pos
                            y_e = self.y_old

                        for y in range(y_b, y_e):
                            self.blur_effect(self.draw_surface, x_pos, y)

                    elif self.y_old == y_pos:
                        if(self.x_old < x_pos):
                            x_b = self.x_old
                            x_e = x_pos
                        else:
                            x_b = x_pos
                            x_e = self.x_old

                        for x in range(x_b, x_e):
                            self.blur_effect(self.draw_surface, x, y_pos)

                    else:
                        if(self.y_old < y_pos):
                            y_b = self.y_old
                            y_e = y_pos
                        else:
                            y_b = y_pos
                            y_e = self.y_old

                        if(self.x_old < x_pos):
                            x_b = self.x_old
                            x_e = x_pos
                        else:
                            x_b = x_pos
                            x_e = self.x_old

                        for x in range(x_b, x_e):
                            for y in range(y_b, y_e):
                                self.blur_effect(self.draw_surface, x, y)

            self.x_old = x_pos
            self.y_old = y_pos
            self.last_time = pygame.time.get_ticks()

    def click_button(self, button):
        mouse = pygame.mouse.get_pressed()
        if(mouse[0] == 1 and in_button(button)):
            button.press()

        if(mouse[0] == 0):
            if in_button(button) and button.pressed:
                button.on_click(surface=self.draw_surface, target=self.num_label, warning=self.warning_label)
            button.release()

    def check_event(self):
        self.paint()

        for button in self.button_group:
            self.click_button(button)

        mouse = pygame.mouse.get_pressed()
        if(mouse[2] == 1):
            pygame.time.delay(50)
            save_func(self.draw_surface, target=self.num_label, warning=self.warning_label)
            clear_func(self.draw_surface, target=self.num_label)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                self.num_label.set_text([event.unicode])
                self.num_label.apply()

    def draw(self):
        self.window.fill(self.display.back_ground_color)
        self.window.blit(self.draw_surface, (0, 0))
        for button in self.button_group:
            button.draw()
        for label in self.label_group:
            label.draw()

    def update(self):
        pass

    def run(self):
        while self.is_run:

            self.check_event()
            self.update()
            self.draw()
            pygame.display.update()


app = Paint()
app.run()
