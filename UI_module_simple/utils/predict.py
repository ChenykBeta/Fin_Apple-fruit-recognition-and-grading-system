from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
import cv2
import numpy as np
import random


def put_chinese_text(image, text, position, text_color, text_size, font_path):
    img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_path, text_size)
    position = (position[0], position[1] - text_size)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=text_color)
    image = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
    return image


def save_modified_results(source, output_path, labels, positions, confidences, label_colors, font_path, text_size=30):
    # 加载图像
    image = cv2.imread(source)

    # 将原始图像复制一份，避免修改原图
    modified_image = np.copy(image)

    # 遍历每个目标
    for label, position, confidence in zip(labels, positions, confidences):
        # 提取目标位置坐标
        x_min, y_min, x_max, y_max = position

        # 获取标签对应的颜色
        color = label_colors[label]
        cv_color = (color[2], color[1], color[0])
        # 在图像上绘制矩形框
        cv2.rectangle(modified_image, (x_min, y_min), (x_max, y_max), cv_color, 3)

        # 构造标签字符串，包含类别和置信度
        label_text = f'{label} {confidence:.2f}'

        # 在图像上绘制目标标签 (英文）
        # cv2.putText(modified_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 在图像上绘制目标标签 (中文）

        modified_image = put_chinese_text(modified_image, label_text, (x_min, y_min),
                                          color, text_size=text_size,  font_path=font_path)

    # 保存修改结果图像
    cv2.imwrite(output_path, modified_image)


def detect(label_name, defects, source, output_path, label_colors, font_path='SimHei.ttf', text_size=30):
    labels_plt = []
    positions_plt = []
    confidences_plt = []

    for defect in defects:
        # 格式转换，先转为cpu格式，然后转为numpy格式
        defect = defect.cpu().numpy()
        # 最终输出list
        boxes = defect.boxes.data.tolist()
        # 遍历每个框
        for box in boxes:
            labels_plt.append(label_name[int(box[5])])
            positions_plt.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            confidences_plt.append(box[4])

    # 保存修改结果图像
    save_modified_results(source, output_path, labels_plt, positions_plt,
                          confidences_plt, label_colors, font_path, text_size)


if __name__ == '__main__':
    # font_path = '../simsun.ttc'
    font_path = '../SimHei.ttf'
    text_size = 30
    model = YOLO('../runs/detect/train/weights/best.pt')
    label_name = ['狗', '人', '火车', '沙发', '椅子', '车',
                           '盆栽', '餐桌', '马', '猫',
                           '牛', '公交车', '自行车', '飞机', '摩托',
                           '屏幕', '鸟', '瓶子', '船', '羊']
    base_path = 'C:/UserData/DZF/Code/Python/Object_Detection/DataSet/Remote/all_images/'
    source = base_path + '1.jpg'
    defects = model.predict(source=source, mode="predict", save=False, conf=0.5, device='0')
    # 保存结果路径
    output_path = base_path + 'aircraft_32.jpg'

    common_colors = [(0, 0, 0),  # 黑色
                     (255, 250, 250),  # 雪白
                     (0, 0, 255),  # 蓝色
                     (255, 0, 0),  # 红色
                     (255, 215, 0),  # 金黄色
                     (51, 161, 201),  # 孔雀蓝
                     (255, 127, 80)]  # 珊瑚色
    while len(common_colors) < len(label_name):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        common_colors.append(color)
    # 初始化颜色索引
    color_index = 0
    # 定义一个空字典来存储标签和颜色的关联
    label_colors = {}
    for one_name in label_name:
        # 获取下一个颜色
        color = common_colors[color_index]
        # 将标签与颜色进行关联
        label_colors[one_name] = color
        # 更新颜色索引
        color_index = color_index + 1

    detect(label_name, defects, source, output_path, label_colors,
           font_path=font_path, text_size=text_size)

