from datasets import Datasets
from Pix2Pix import Pix2Pix

PATH = "D:/Chocolate/datasets/edges2shoes"  # 指定数据集位置
EPOCHS = 32  # 指定训练批次


if __name__ == '__main__':
    my_datasets = Datasets(datasets_path=PATH)  # 加载数据集
    model = Pix2Pix("./training_checkpoints")  # 指定模型位置
    model.train(my_datasets, epochs=EPOCHS)  # 开始执行训练
