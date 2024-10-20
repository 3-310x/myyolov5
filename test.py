import os

def train_yolov5s():
    # 配置参数
    data_yaml = 'D:/project/yolov5/sampleSAEshapes/data.yaml'  # 数据集配置文件的路径
    weights = 'D:/project/yolov5/models/SKAyolov5s.yaml'   # 预训练权重文件
    img_size = 640           # 输入图像尺寸
    batch_size = 32          # 批次大小
    epochs = 75             # 训练轮数
    # 构建训练命令
    command = f'python train.py --img {img_size} --batch {batch_size} --epochs {epochs} ' \
              f'--data {data_yaml} --weights {weights} --device 0'

    # 执行训练命令
    os.system(command)

if __name__ == '__main__':
    train_yolov5s()
