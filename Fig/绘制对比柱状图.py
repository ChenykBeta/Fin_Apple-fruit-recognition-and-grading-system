import matplotlib.pyplot as plt
import numpy as np
import os
import random


# # 生成互不相似的随机颜色
# def generate_distinct_colors(num_colors):
#     colors = []
#     for i in range(num_colors):
#         color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
#         while color in colors:
#             color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
#         colors.append(color)
#     return colors

def generate_distinct_colors(num_colors):
    base_colors = ["#FFA500", "#008000", "#FF0000", "#00FFFF"]
    colors = []
    for i in range(num_colors):
        colors.append(base_colors[i % len(base_colors)])
    return colors


def compare_performance_test(models, mAP50, precision, recall, f1_score):
    # 指标列表和颜色
    metrics = [mAP50, precision, recall, f1_score]
    metric_names = ['mAP50', 'Precision', 'Recall', 'F1 Score']
    colors = generate_distinct_colors(len(metric_names))
    # 绘制柱状图
    plt.figure(figsize=(10, 6))

    bar_width = 0.2
    index = np.arange(len(models))

    for i, (metric, color, metric_name) in enumerate(zip(metrics[::-1], colors[::-1], metric_names[::-1])):
        plt.barh(index + i * bar_width, metric, bar_width, label=metric_name, color=color)

    # 在每个柱状图上方添加数值标签
    # for i, metric in enumerate(metrics[::-1]):
    #     for j, value in enumerate(metric):
    #         plt.text(value + 0.01, j + i * bar_width, round(value, 2), va='center')

    # 设置Y轴标签和刻度
    plt.yticks(index + bar_width, models)
    plt.xlabel('Metrics')
    plt.title('Model Performance Comparison')

    # 保持图注顺序不变
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))  # 图注放到图像外面

    # 设置X轴坐标范围为0到1
    plt.xlim(0.3, 1)

    # # 显示图形
    # plt.show()
    # 保存路径
    plt.savefig(os.path.join(r'./', "Test_performance_compare.png"), dpi=300, bbox_inches='tight')


def get_data(my_data):
    models = []  # 可以修改为实际的模型名称
    mAP50 = []  # 准确率
    precision = []  # 精确率
    recall = []  # 召回率
    f1_score = []  # F1值
    for one_data in my_data:
        models.append(one_data['name'])
        mAP50.append(one_data['mAP50'])
        precision.append(one_data['precision'])
        recall.append(one_data['recall'])
        f1_score.append(one_data['f1_score'])
    return models, mAP50, precision, recall, f1_score


if __name__ == '__main__':
    my_data = [

        {'name': 'YoloV8_Improve',
         'mAP50': 0.892,
         'precision': 0.842,
         'recall': 0.828,
         'f1_score': 2 * 0.842 * 0.828 / (0.842 + 0.828)
         },

        {'name': 'YoloV8',
         'mAP50': 0.856,
         'precision': 0.835,
         'recall': 0.801,
         'f1_score': 2 * 0.835 * 0.801 / (0.835 + 0.801)
         },

        {'name': 'YoloV5',
         'mAP50': 0.8,
         'precision': 0.804,
         'recall': 0.775,
         'f1_score': 2 * 0.804 * 0.775 / (0.804 + 0.775)
         },

        {'name': 'Faster_RCNN',
         'mAP50': 0.7925,
         'precision': 0.6934,
         'recall': 0.7999,
         'f1_score': 2 * 0.6934 * 0.7999 / (0.6934 + 0.7999)
         },

    ]
    # 模型名称和对应的各项指标
    models, mAP50, precision, recall, f1_score = get_data(my_data)
    compare_performance_test(models, mAP50, precision, recall, f1_score)
