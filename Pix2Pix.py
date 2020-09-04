import os

import matplotlib.pyplot as plt
import tensorflow as tf

OUTPUT_CHANNELS = 3
LAMBDA = 100


def downsample(filters, size, apply_batchnorm=True):
    """
    下采样层
    下采样层使用 Conv2D 卷积层来实现
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """
    上采样层
    上采样层使用 Conv2DTranspose 转置卷积来实现
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result


def generator():
    """
    生成器 模型

    经过下采样，得到类似于 DCGAN 生成器的初始输入
    上采样类似于 DCGAN 中的转置卷积，由初始输入生成图像

    """
    # 输入层
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # => (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # => (bs, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # => (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # => (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')  # => (bs, 256, 256, 3)

    x = inputs  # 输入训练集
    # 对训练集进行下采样，获取边缘特征
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)  # 记录每一步下采样提取的特征

    skips = reversed(skips[:-1])  # 将结果逆置，以便与上采样结果建立对应关系

    # 对下采样的结果进行上采样，并建立跳过连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # 将每一步上采样提取的特征与对应的下采样提取的特征建立跳过连接
        # 目的：特征融合
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator():
    """
    判别器 模型

    判别器类似于 CNN
    其中加入了零填充
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    # 输入层
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    # 输入图像与目标图像进行特征融合
    x = tf.keras.layers.concatenate([inp, tar])  # => (bs, 256, 256, 6)

    # 进行下采样 实为卷积提取特征
    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # => (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # 零填充 => (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)  # => (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)  # 批归一化
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # 零填充 => (bs, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # => (bs, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(discriminator_generated_output, generator_output, target):
    """
    生成器损失

    tf.ones_like(): 创建一个和输入 tensor 维度一样，元素都为 1 的新 tensor
    tf.reduce_mean(): 主要用作降维或者计算 tensor 的平均值

    """
    # 计算判别器对伪造图像的输出与全 1 tensor 的交叉熵
    gan_loss = loss_object(tf.ones_like(discriminator_generated_output), discriminator_generated_output)
    # 计算伪造图像与真实图像差的 tensor 的均值
    l1_loss = tf.reduce_mean(tf.abs(target - generator_output))
    # 计算总损失
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(discriminator_real_output, discriminator_generated_output):
    # 分别计算判别器对真实图像和伪造图像与全 1 tensor 的交叉熵
    real_loss = loss_object(tf.ones_like(discriminator_real_output), discriminator_real_output)
    generated_loss = loss_object(tf.zeros_like(discriminator_generated_output), discriminator_generated_output)
    # 计算总损失
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generate_images(dir_name, model, epoch, test_input, test_target):
    """
    训练过程中生成并保存图像
    """
    prediction = model(test_input, training=True)

    display_list = [test_input[0], test_target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    plt.figure(figsize=(9, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # pixel values to [0, 1]
        plt.axis('off')
    plt.savefig('./{:s}/image_at_epoch_{:04d}.png'.format(dir_name, epoch))
    plt.close()
    print('\nImage (image_at_epoch_{:04d}.png) has been saved'.format(epoch))


class Pix2Pix(object):
    """
    Pix2Pix

    图像到图像的生成
    """
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def __init__(self, checkpoint_dir='./training_checkpoints'):
        self.generator = generator()
        self.discriminator = discriminator()

        # 设置模型保存检查点
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        # 载入模型
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    @tf.function
    def train_step(self, input_image, target_image):
        """
        自定义训练循环
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_output = self.generator(input_image, training=True)  # 生成器生成伪造图像
            discriminator_real_output = self.discriminator([input_image, target_image], training=True)  # 判别器对真实图像进行鉴定
            discriminator_generated_output = self.discriminator([input_image, generator_output], training=True)  # 判别器对伪造图像进行鉴定
            # 计算生成器损失
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(discriminator_generated_output, generator_output, target_image)
            # 计算判别器损失
            disc_loss = discriminator_loss(discriminator_real_output, discriminator_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    def train(self, datasets, epochs):
        for epoch in range(epochs):
            print("Epoch:", epoch + 1)
            progress_bar = tf.keras.utils.Progbar(1557)
            # 开始训练
            for n, (input_image, target) in datasets.train_dataset.enumerate():
                self.train_step(input_image, target)
                progress_bar.update(int(n))  # 每训练一步，输出一个指示符

            # 生成并保存本批次训练结果图像
            for example_input, example_target in datasets.test_dataset.take(1):
                generate_images("generated", self.generator, epoch + 1, example_input, example_target)
            
            # 每个批次结束保存一次模型
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def generator_one(self, input_image):
        """
        生成一张图像
        """
        output = self.generator(input_image, training=False)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(input_image[0] * 0.5 + 0.5)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Output")
        plt.imshow(output[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()
        plt.close()
