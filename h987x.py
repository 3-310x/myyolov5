import cv2
import numpy as np
import random

# 创建一个 800x480 的黄色背景图像
width, height = 800, 480
background = np.full((height, width, 3), (0, 255, 255), dtype=np.uint8)  # 黄色背景

# 图形的边长或直径设定
area_size = 5000  # 图形的面积设定（单位：像素）
side_length = int(np.sqrt(area_size))  # 正方形和菱形的边长
radius = int(np.sqrt(area_size / np.pi))  # 圆形的半径


# 函数生成正方形
def draw_square(img, center, side, color):
    top_left = (center[0] - side // 2, center[1] - side // 2)
    bottom_right = (center[0] + side // 2, center[1] + side // 2)
    cv2.rectangle(img, top_left, bottom_right, color, -1)


# 函数生成圆形
def draw_circle(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1)


# 函数生成菱形
def draw_diamond(img, center, side, color):
    half_side = side // 2
    pts = np.array([
        [center[0], center[1] - half_side],  # 上
        [center[0] + half_side, center[1]],  # 右
        [center[0], center[1] + half_side],  # 下
        [center[0] - half_side, center[1]]  # 左
    ], np.int32)
    cv2.fillPoly(img, [pts], color)


# 在随机位置绘制图形
shapes = []
for _ in range(2):  # 6个图形，每种2个
    shapes.append(("square", random.randint(side_length // 2, width - side_length // 2),
                   random.randint(side_length // 2, height - side_length // 2)))
    shapes.append(("circle", random.randint(radius, width - radius), random.randint(radius, height - radius)))
    shapes.append(("diamond", random.randint(side_length // 2, width - side_length // 2),
                   random.randint(side_length // 2, height - side_length // 2)))

for shape, x, y in shapes:
    center = (x, y)
    if shape == "square":
        draw_square(background, center, side_length, (255, 0, 0))  # 蓝色正方形
    elif shape == "circle":
        draw_circle(background, center, radius, (0, 0, 255))  # 红色圆形
    elif shape == "diamond":
        draw_diamond(background, center, side_length, (0, 255, 0))  # 绿色菱形

# 随机在形状周围放置“青春”两个字
font = cv2.FONT_HERSHEY_SIMPLEX
for shape, x, y in shapes:
    offset_x = random.randint(-50, 50)
    offset_y = random.randint(-50, 50)
    cv2.putText(background, '青春', (x + offset_x, y + offset_y), font, 1, (0, 0, 0), 2)

# 保存图片
cv2.imwrite('random_shapes_with_text.jpg', background)
