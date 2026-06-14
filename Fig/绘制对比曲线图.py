import os.path
import pandas as pd
import matplotlib.pyplot as plt

def compare_map_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['val_map'], label=one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("mAP50")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')

    # 表格标题
    plt.title("mAP50 Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "mAP50_compare.png"), dpi=300, bbox_inches='tight')



def compare_precision_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['val_precision'], label=one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Precision")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Precision Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Precision_compare.png"), dpi=300, bbox_inches='tight')


def compare_recall_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['val_recall'], label=one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Recall")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Recall Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Recall_compare.png"), dpi=300, bbox_inches='tight')


def compare_f1_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['val_f1'], label=one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("F1_Score")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("F1_Score Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "F1_Score_compare.png"), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    my_csvs = [
        {'name': 'Faster_RCNN',
         'path': 'Faster_RCNN.csv'},

        {'name': 'YoloV5',
         'path': 'YoloV5.csv'},

        {'name': 'YoloV8',
         'path': 'YoloV8.csv'},

        {'name': 'YoloV8_Improve',
         'path': 'YoloV8_Improve.csv'},
    ]
    compare_map_val(my_csvs)
    compare_precision_val(my_csvs)
    compare_recall_val(my_csvs)
    compare_f1_val(my_csvs)


