import os

import matplotlib.pyplot as plt
import tensorflow as tf

PATH = "D:/Chocolate/datasets/edges2shoes"
IMG_HEIGHT = 256
IMG_WIDTH = 256
BUFFER_SIZE = 5000
BATCH_SIZE = 32


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    # 对原始数据集进行裁剪
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    real_image = tf.cast(real_image, tf.float32)
    input_image = tf.cast(input_image, tf.float32)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    """
    随机裁剪
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    """
    图像归一化 [-1, 1]
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function
def random_jitter(input_image, real_image):
    """
    随机抖动

    首先将图像调整为更大的高度和宽度
    然后随机裁剪为目标尺寸并随机水平翻转图像
    目的：避免过拟合
    """
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)  # 随机裁剪 to 256 x 256 x 3

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    # 随机水平翻转图像
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


def load_train_image(image_file):
    input_image, real_image = load_image(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_test_image(image_file):
    input_image, real_image = load_image(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


class Datasets(object):
    def __init__(self, datasets_path=PATH, batch_size=BATCH_SIZE):
        print(os.path.realpath(datasets_path))

        self.train_dataset = tf.data.Dataset.list_files(datasets_path + '/train/*.jpg')
        self.train_dataset = self.train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_dataset = self.train_dataset.shuffle(BUFFER_SIZE).batch(batch_size)

        self.test_dataset = tf.data.Dataset.list_files(datasets_path + '/test/*.jpg')
        self.test_dataset = self.test_dataset.map(load_test_image)
        self.test_dataset = self.test_dataset.batch(batch_size)
