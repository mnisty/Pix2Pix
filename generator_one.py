import pathlib

import numpy as np
import tensorflow as tf

from Pix2Pix import Pix2Pix

PATH = "D:/Chocolate/datasets/edges2shoes"
IMG_HEIGHT = 256
IMG_WIDTH = 256


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    # 对原始数据集进行裁剪
    w = tf.shape(image)[1]
    w = w // 2
    image = image[:, :w, :]

    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=0)
    return image


if __name__ == '__main__':
    model = Pix2Pix("./training_checkpoints")

    # 多例测试
    test_dir = pathlib.Path(PATH + "./test")
    all_test_images_path = list(test_dir.glob("*.jpg"))

    input_images = [load_image(str(path)) for path in all_test_images_path]
    for input_image in input_images:
        model.generator_one(input_image)
