import pygame
import numpy as np
import cv2
import pandas as pd
import os
import json
from neuralnet import *


def save_log(target, i):
    sam = read_log('sample_number')
    pre = read_log('predict_number')
    true = read_log('true_number')
    if target == 'sample_number':
        sam = i
    elif target == 'predict_number':
        pre = i
    elif target == 'true_number':
        true.append(i)
    data = {'sample_number': sam, 'predict_number': pre, 'true_number': true}
    with open('log.json', 'w') as f:
        json.dump(data, f)


def reset_log():
    data = {'sample_number': 0, 'predict_number': 0, 'true_number': [0]}
    with open('log.json', 'w') as f:
        json.dump(data, f)


def read_log(key):
    if not os.path.exists('log.json'):
        reset_log()

    with open('log.json', 'r') as f:
        data = json.load(f)
    return data[key]


def is_drawn(surface):
    pixel = pygame.surfarray.array3d(surface)
    array = np.array(pixel, dtype=np.int64)
    return not (array == 255).all()


def save_func(surface, **kwargs):
    if not os.path.exists('images'):
        os.mkdir('images')
    target = kwargs['target'].get_text()
    warning_label = kwargs['warning']
    if not is_drawn(surface) or target == ['']:
        warning_label.set_text(['Warning!!', 'Haven\'t drawn', 'or set label'])
        warning_label.apply()
    else:
        warning_label.set_text([''])
        warning_label.apply()
        number = read_log('sample_number')
        filename = os.path.join('images', str(number) + '_image' + '.png')
        number += 1
        save_log('sample_number', number)

        pygame.image.save(surface, filename)
        im = cv2.imread(filename)
        imresize = cv2.resize(im, (20, 20))
        grayscale = cv2.cvtColor(imresize, cv2.COLOR_BGR2GRAY).reshape(1, 400)

        data = pd.DataFrame(grayscale, index=[number])
        target = pd.DataFrame({'target': target}, index=[number])
        data.to_csv(os.path.join('images', 'data.csv'), mode='a', header=False)
        target.to_csv(os.path.join('images', 'target.csv'), mode='a', header=False)


def clear_func(surface, **kwargs):
    target = kwargs['target']
    if(target != None):
        target.set_text([''])
        target.apply()
    surface.fill((255, 255, 255))


def predict_func(surface, **kwargs):
    warning_label = kwargs['warning']
    num_label = kwargs['target']

    if not os.path.exists('predicts'):
        os.mkdir('predicts')

    if not os.path.exists('model.json'):
        warning_label.set_text(["Haven't train model!!"])
        warning_label.apply()
    else:
        if not is_drawn(surface):
            warning_label.set_text(["Haven't drawn!!"])
            warning_label.apply()
        else:
            warning_label.set_text('')
            warning_label.apply()
            model = load_model('model.json')
            predict_number = read_log('predict_number')
            predict_number += 1
            filename = os.path.join('predicts', 'images' + str(predict_number) + '.png')
            save_log('predict_number', predict_number)
            pygame.image.save(surface, filename)
            im = cv2.imread(filename)
            imresize = cv2.resize(im, (20, 20))
            grayscale = cv2.cvtColor(imresize, cv2.COLOR_BGR2GRAY).reshape(1, 400)
            result = model.predict(grayscale)
            num_label.set_text(str(int(result)))
            num_label.apply()


def train_func(surface, **kwargs):
    X = pd.read_csv('images/data.csv', index_col=0).to_numpy() / 255
    y = pd.read_csv('images/target.csv', index_col=0)['target'].to_numpy()
    index = np.random.permutation(len(X) - 1)
    X = X[index]
    y = y[index]

    model = NeuralNetwork(learning_rate=0.0001)
    model.add(Layer(400))
    model.add(Layer(100, activation='relu'))
    model.add(Layer(100, activation='relu'))
    model.add(Layer(10, activation='softmax'))
    model.summary()

    model.fit(X[:40000], y[:40000], epochs=20, batch_size=10, validation_data=(X[40000:], y[40000:]))
    model.save('model.json')
    model.plot()


def plot_func(surface, **kwargs):
    plt.style.use('seaborn-whitegrid')
    true_num = read_log('true_number')[1:]
    times = np.arange(len(true_num)) + 1
    array = np.array(true_num) / times
    plt.plot(array, c='orange', label='Prediction')
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Prediction')
    plt.show()


def true_func(surface, **kwargs):
    num_label = kwargs['target']
    warning_label = kwargs['warning']
    if is_drawn(surface) and num_label.get_text != '':
        warning_label.set_text('')
        warning_label.apply()
        true_num = read_log('true_number')
        num = true_num[-1] + 1
        save_log('true_number', num)
        clear_func(surface, **kwargs)
    else:
        warning_label.set_text(["Haven't predicted!!"])
        warning_label.apply()


def false_func(surface, **kwargs):
    num_label = kwargs['target']
    warning_label = kwargs['warning']
    if is_drawn(surface) and num_label.get_text != '':
        warning_label.set_text('')
        warning_label.apply()
        true_num = read_log('true_number')
        save_log('true_number', true_num[-1])
        clear_func(surface, **kwargs)
    else:
        warning_label.set_text(["Haven't predicted"])
        warning_label.apply()


def in_button(button):
    pos = pygame.mouse.get_pos()

    return (pos[0] - button.rect.left >= 0) and (button.rect.right - pos[0] >= 0) and (pos[1] - button.rect.top >= 0) and (button.rect.bottom - pos[1] >= 0)


def darker_surface(surface, amount):
    global once
    pixel = pygame.surfarray.array3d(surface)
    color = np.array(pixel, dtype=np.int64)
    color = np.where(color > amount, color - amount, 0)

    return pygame.surfarray.make_surface(color)


def darker(color, amount):
    col = np.array(color)
    col -= np.full(3, amount)
    return tuple(col)
