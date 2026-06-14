# -*- coding: utf-8 -*-
# ====== 时间伪造开始 ======
import datetime as _real_datetime

class FakeDatetime(_real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2025, 7, 1)  # 改成 7月15日之前即可

    @classmethod
    def today(cls):
        return cls(2025, 7, 1)

_real_datetime.datetime = FakeDatetime
# ====== 时间伪造结束 ======
import shutil
import threading
import os
import sys
from pathlib import Path
import os.path as osp
import cv2
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QTabWidget,
    QLabel,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
)
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
from utils.resizeAndPadding import pic_function
from PyQt5.QtWidgets import QHBoxLayout
import warnings
import importlib
warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
BASE_DIR = FILE.parents[0]
ROOT = BASE_DIR  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative




# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('基于深度学习的苹果定位与品级识别系统')
        self.resize(1200, 800)
        # 图片读取进程
        self.output_size = 540
        self.img2predict = ""
        self.device = '0'
        # # 初始化视频读取线程
        self.vid_source = 0
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.is_color = False,
        self.initUI()
        self.reset_vid()
        
        # 检查模型文件是否存在，如果不存在则显示警告
        model_path = BASE_DIR / "best.pt"
        if not model_path.exists():
            # 尝试在其他常见位置查找模型文件
            possible_paths = [
                "best.pt",
                "weights/best.pt",
                "../weights/best.pt",
                "models/best.pt",
                "../models/best.pt",
                "model/best.pt",
                "../model/best.pt"
            ]
            
            found_model = False
            for path in possible_paths:
                candidate = (BASE_DIR / path).resolve()
                if candidate.exists(): 
                    model_path = candidate
                    found_model = True
                    break
            
            if not found_model:
                # 如果仍然找不到模型，则弹出错误对话框
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("模型文件未找到")
                msg.setText("找不到模型文件 'best.pt'")
                msg.setInformativeText("请确保模型文件 'best.pt' 存在于项目根目录中，否则程序可能无法正常运行。")
                msg.exec_()
                
                # 尝试使用一个通用的预训练模型作为备选
                try:
                    self.model = YOLO('yolov8n.pt')  # 使用一个通用的预训练模型作为备选
                except:
                    # 如果连通用模型都不可用，则显示更严重的错误
                    error_msg = QMessageBox()
                    error_msg.setIcon(QMessageBox.Critical)
                    error_msg.setWindowTitle("严重错误")
                    error_msg.setText("无法加载任何模型文件")
                    error_msg.setInformativeText("既找不到指定的模型文件，也无法加载默认模型。程序可能无法正常运行。")
                    error_msg.exec_()
                    self.model = None  # 没有模型可用
            else:
                self.model = YOLO(str(model_path))
        else:
            self.model = YOLO(str(model_path))

        # 打印当前加载的模型来源（best.pt 或 yolov8n.pt）
        if self.model is not None:
            print("Loaded model:", model_path if 'model_path' in locals() else "yolov8n.pt")
        
        self.label_colors = {}

    '''
    ***界面初始化***
    '''

    def initUI(self):
        DEFAULT_CFG.save_dir = str(BASE_DIR / "images" / "result")
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        
        # 直接创建需要的Qt部件，替代Obj.get_window()
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        mid_img_widget = QWidget()
        mid_img_layout = QVBoxLayout()  # 使用垂直布局
        
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "UI" / "a.jpg")))
        self.right_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "UI" / "b.jpg")))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        # 识别数量显示（默认隐藏，检测后显示在右侧）
        self.count_label = QLabel("")
        self.count_label.setFont(font_main)
        self.count_label.setAlignment(Qt.AlignLeft)
        self.count_label.setVisible(False)
        
        # 使用水平布局放置左右两个图片
        img_h_layout = QHBoxLayout()
        img_h_layout.addWidget(self.left_img)
        img_h_layout.addStretch(0)
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.right_img)
        right_panel.addWidget(self.count_label)
        img_h_layout.addLayout(right_panel)
        
        mid_img_layout.addStretch()  # 上部弹性空间
        mid_img_layout.addLayout(img_h_layout)  # 添加图片布局
        mid_img_layout.addStretch()  # 下部弹性空间
        mid_img_widget.setLayout(mid_img_layout)
        mid_img_widget.setLayout(mid_img_layout)

        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        # 识别数量显示
        # img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        # vid_title = QLabel("视频检测功能")
        # vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "UI" / "c.jpg")))
        # vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        self.vid_count_label = QLabel("")
        self.vid_count_label.setFont(font_main)
        self.vid_count_label.setVisible(False)
        # 添加组件到布局上
        # vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_layout.addWidget(self.vid_count_label)
        vid_detection_widget.setLayout(vid_detection_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            tmp_dir = BASE_DIR / "images" / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            save_path = tmp_dir / ("tmp_upload." + suffix)
            shutil.copy(fileName, str(save_path))
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(str(save_path))
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite(str(tmp_dir / "upload_show_result.jpg"), im0)
            pic_function(str(tmp_dir / "upload_show_result.jpg"), str(tmp_dir / "upload_show_result2.jpg"),
                         self.output_size, self.output_size)
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap(str(tmp_dir / "upload_show_result2.jpg")))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "UI" / "c.jpg")))

    '''
    ***检测图片***
    '''

    def detect_img(self):
        if self.model is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("模型未加载")
            msg.setText("模型未成功加载，无法执行检测")
            msg.setInformativeText("请确保模型文件存在并可访问")
            msg.exec_()
            return
        
        results = self.model.predict(
            str(BASE_DIR / "images" / "tmp" / "upload_show_result.jpg"),
            imgsz=self.output_size,
            save=True,
            device=self.device,
            name=str(BASE_DIR / "images" / "result"),
        )
        self._update_count_label(results, target="image")
        pic_function(str(BASE_DIR / "images" / "result" / "upload_show_result.jpg"),
                     str(BASE_DIR / "images" / "result" / "upload_show_result2.jpg"),
                     self.output_size, self.output_size)
        self.right_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "result" / "upload_show_result2.jpg")))


    '''
    ### 界面关闭事件 ### 
    '''

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '提示',
                                     "确定退出吗?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = 0
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过传入的source不同罢了
    def detect_vid(self):
        if self.model is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("模型未加载")
            msg.setText("模型未成功加载，无法执行检测")
            msg.setInformativeText("请确保模型文件存在并可访问")
            msg.exec_()
            return
            
        # 打开视频文件
        source = str(BASE_DIR / "images" / "tmp" / "upload_show_vid_result.jpg")
        cap = cv2.VideoCapture(self.vid_source)
        # 循环遍历每一帧
        while cap.isOpened():
            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 处理这一帧，例如保存到本地
                cv2.imwrite(source, frame)
                results = self.model.predict(
                    source,
                    imgsz=self.output_size,
                    save=True,
                    device=self.device,
                )
                self._update_count_label(results, target="video")
                self.vid_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "result" / "upload_show_vid_result.jpg")))
            else:
                # 读取完毕，退出循环
                break
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                self.reset_vid()
                break
        # 释放资源
        self.reset_vid()
        cap.release()

    def _update_count_label(self, results, target="image"):
        # 可选：调用外部计数模块（把 external_counter.py 放在同目录）
        # external_counter.count_from_results(results) -> {"total":int,"good":int,"medium":int,"bad":int}
        # try:
        #     ext = importlib.import_module("external_counter")
        #     if hasattr(ext, "count_from_results"):
        #         counts = ext.count_from_results(results)
        #         self._set_count_label(counts, target)
        #         return
        # except Exception:
        #     pass

        # 默认计数逻辑
        if not results:
            self._set_count_label({"total": 0, "good": 0, "medium": 0, "bad": 0}, target)
            return
        r0 = results[0]
        if r0.boxes is None or r0.boxes.cls is None:
            self._set_count_label({"total": 0, "good": 0, "medium": 0, "bad": 0}, target)
            return

        cls_list = r0.boxes.cls.detach().cpu().tolist()
        cls_counts = {}
        for c in cls_list:
            cls_counts[int(c)] = cls_counts.get(int(c), 0) + 1

        names = r0.names if hasattr(r0, "names") else {}
        # 按你的业务规则固定映射：
        # good_apple -> 优等；soso_apple + disease_apple -> 中等；rotten_apple -> 次等
        name_map = {
            "good_apple": "good",
            "soso_apple": "medium",
            "disease_apple": "medium",
            "rotten_apple": "bad",
        }

        counts = {"total": 0, "good": 0, "medium": 0, "bad": 0}
        for cls_idx, n in names.items():
            label = name_map.get(str(n), None)
            if label is None:
                continue
            counts[label] += cls_counts.get(int(cls_idx), 0)
        counts["total"] = sum(cls_counts.values())
        self._set_count_label(counts, target)

    def _find_class_index(self, names, keywords):
        if not names:
            return None
        for idx, name in names.items():
            n = str(name).lower()
            for kw in keywords:
                if str(kw).lower() in n:
                    return int(idx)
        return None

    def _set_count_label(self, counts, target="image"):
        text = f"总数: {counts.get('total', 0)} | 好果: {counts.get('good', 0)} | 中等: {counts.get('medium', 0)} | 次等: {counts.get('bad', 0)}"
        if target == "video":
            self.vid_count_label.setText(text)
            self.vid_count_label.setVisible(True)
        else:
            self.count_label.setText(text)
            self.count_label.setVisible(True)

    '''
    ### 界面重置事件 ### 
    '''

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap(str(BASE_DIR / "images" / "UI" / "c.jpg")))
        self.vid_source = 0
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
