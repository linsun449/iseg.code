import sys

import torch
from PIL import Image
from PyQt5.QtCore import QDir, Qt, QPoint, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QBrush, QColor, QPen

import interaction
import numpy as np
import torch.nn.functional as F

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsSceneMouseEvent, QMenu

import warnings


warnings.filterwarnings("ignore")


def generate_distinct_colors():
    colors = [(0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
              (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128),
              (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
              (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64), (0, 128, 64), (128, 128, 64),
              (0, 0, 192), (128, 0, 192), (0, 128, 192), (128, 128, 192), (64, 0, 64), (192, 0, 64),
              (64, 128, 64), (192, 128, 64), (64, 0, 192), (192, 0, 192), (64, 128, 192), (192, 128, 192),
              (0, 64, 64), (128, 64, 64), (0, 192, 64), (128, 192, 64), (0, 64, 192), (128, 64, 192),
              (0, 192, 192), (128, 192, 192), (64, 64, 64), (192, 64, 64), (64, 192, 64), (192, 192, 64)]
    return colors


def iterative_mean_threshold(img, max_iterations=10, convergence_threshold=0.001):
    mean_value = img.mean()  # 计算当前平均值
    for _ in range(max_iterations):
        # 分割图像为低于和高于当前均值的两部分
        below_mean = img[img <= mean_value].mean()
        above_mean = img[img >= mean_value].mean()

        new_threshold = below_mean * 0.9 + above_mean * 0.1

        # 检查收敛条件
        if abs(new_threshold - mean_value) < convergence_threshold:
            break
        mean_value = new_threshold  # 更新前一次阈值
    return mean_value


class Message:
    def __init__(self, mess_type, pos):
        self.mess_type = mess_type
        self.pos = pos


class ProcessSignals(QObject):
    signal_process = pyqtSignal(Message)
    show_process = pyqtSignal(str)


class Loader(QObject):

    def __init__(self, message):
        super().__init__()
        self.process_message = message
        self.iSeg = None

    def load_model(self):
        # 加载模型
        from ui.arguments import init_args
        from ui.interaction_iSeg import interaction_iSeg
        config = init_args()
        self.iSeg = interaction_iSeg(config)
        self.iSeg = self.iSeg.cuda() if torch.cuda.is_available() else self.iSeg
        self.iSeg.on_test_start()
        self.process_message.show_process.emit("Loaded iSeg!")


class ClickableQGraphicsScene(QGraphicsScene):
    def __init__(self, message, type_id=1, parent=None):
        super(ClickableQGraphicsScene, self).__init__(parent)
        self.process_message = message
        self.type_id = type_id
        self.pos = None
        self.pos_list = []

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.LeftButton and self.type_id > 0:
            pos = event.scenePos()
            mouse_pos = self.views()[0].mapFromScene(pos)
            scene_pos = self.views()[0].mapToScene(mouse_pos)
            if scene_pos.x() < 0 or scene_pos.y() < 0: return
            if scene_pos.x() >= self.width() or scene_pos.y() >= self.height(): return
            self.pos = QPoint(int(scene_pos.x()), int(scene_pos.y()))
            self.pos_list = [self.pos]
            self.process_message.signal_process.emit(Message(0,
                                                             QPoint(int(scene_pos.x()), int(scene_pos.y()))))

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self.type_id == 2 and self.pos is not None:
            pos = event.scenePos()
            mouse_pos = self.views()[0].mapFromScene(pos)
            scene_pos = self.views()[0].mapToScene(mouse_pos)
            if scene_pos.x() < 0 or scene_pos.y() < 0: return
            if scene_pos.x() >= self.width() or scene_pos.y() >= self.height(): return
            self.pos_list.append(QPoint(int(scene_pos.x()), int(scene_pos.y())))
            self.process_message.signal_process.emit(Message(1, self.pos_list[-1]))
        elif self.type_id == 3 and self.pos is not None:
            pos = event.scenePos()
            mouse_pos = self.views()[0].mapFromScene(pos)
            scene_pos = self.views()[0].mapToScene(mouse_pos)
            if scene_pos.x() < 0 or scene_pos.y() < 0: return
            if scene_pos.x() >= self.width() or scene_pos.y() >= self.height(): return
            self.process_message.signal_process.emit(Message(3,
                                                             QPoint(int(scene_pos.x()), int(scene_pos.y()))))

    def mouseReleaseEvent(self, event):
        if self.type_id == 2 and self.pos is not None and event.button() == Qt.LeftButton:
            self.pos = None
            self.process_message.signal_process.emit(Message(2, QPoint(0, 0)))
        elif self.type_id == 3 and self.pos is not None and event.button() == Qt.LeftButton:
            self.pos = None
            pos = event.scenePos()
            mouse_pos = self.views()[0].mapFromScene(pos)
            scene_pos = self.views()[0].mapToScene(mouse_pos)
            if scene_pos.x() < 0 or scene_pos.y() < 0: return
            if scene_pos.x() >= self.width() or scene_pos.y() >= self.height(): return
            self.process_message.signal_process.emit(Message(4,
                                                             QPoint(int(scene_pos.x()), int(scene_pos.y()))))


class iSeg(interaction.Ui_iSeg):

    def __init__(self):
        self.click_pos = None
        self.image = None
        self.init_cross = None
        self.final_cross = None
        self.mask = None
        self.overlay = None

        self.process_signals = ProcessSignals()

        self.scene_image = ClickableQGraphicsScene(self.process_signals, type_id=1)
        self.scene_init = QGraphicsScene()
        self.scene_final = QGraphicsScene()
        self.scene_mask = QGraphicsScene()
        self.scene_overlay = QGraphicsScene()

        self.type_id = 1  # 0:Text, 1: Point, 2:Line

        self.iSeg = None

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        self.init_all()
        self.btn_load.clicked.connect(self.path_clicked)
        self.submit.clicked.connect(self.submit_clicked)
        self.process_signals.signal_process.connect(self.processHandler)
        self.process_signals.show_process.connect(self.showHandler)

        self.com_type.setCurrentIndex(1)
        self.com_type.currentIndexChanged.connect(self.typeChanged)
        self.lineEdit.setEnabled(False)
        self.submit.setEnabled(False)

        self.slider_iter.valueChanged.connect(self.sliderChanged)

        self.show_label.setText("Loading iSeg...")
        self.loder = Loader(self.process_signals)
        self.thread = QThread(MainWindow)
        self.loder.moveToThread(self.thread)
        self.thread.started.connect(self.loder.load_model)
        self.thread.start()

    def sliderChanged(self):
        self.label_iter.setText(f"Iter:{int(self.slider_iter.value())}")

    def submit_clicked(self):
        self.iSegProcess(None)

    def typeChanged(self):
        self.type_id = self.com_type.currentIndex()
        self.scene_image.type_id = self.type_id
        self.scene_image.pos = None
        if self.type_id == 0:
            self.lineEdit.setEnabled(True)
            self.submit.setEnabled(True)
        else:
            self.lineEdit.setEnabled(False)
            self.submit.setEnabled(False)
        if self.image is not None:
            y, x = self.image.shape[:2]
            # 显示图像
            frame = QImage(self.image, x, y, 3 * x, QImage.Format_RGB888)
            self.init_all()
            self.scene_image.addPixmap(QPixmap.fromImage(frame))

    def path_clicked(self):
        if self.iSeg is None: return
        fileDialog = QFileDialog()
        fileDialog.setWindowTitle('Select image file')
        fileDialog.setFileMode(QFileDialog.AnyFile)

        fileDialog.setDirectory(QDir("."))
        fileDialog.setNameFilter("Image (*.png *.jpg)")
        file_path = fileDialog.exec_()
        if file_path and fileDialog.selectedFiles():
            self.init_all()
            image_path = fileDialog.selectedFiles()[0]
            self.btn_load.setText(image_path)
            image = Image.open(image_path).convert('RGB')
            self.image = np.asarray(image)
            y, x = self.image.shape[:2]
            # 显示图像
            frame = QImage(self.image, x, y, 3 * x, QImage.Format_RGB888)
            self.scene_image.clear()
            self.scene_image.addPixmap(QPixmap.fromImage(frame))

            self.iSeg.self_attn = None
            self.iSeg.cross_attn = None
        else:
            self.btn_load.setText("UpLoad image")

    def init_all(self):
        self.scene_image = ClickableQGraphicsScene(self.process_signals, type_id=self.type_id)
        self.scene_init = QGraphicsScene()
        self.scene_final = QGraphicsScene()
        self.scene_mask = QGraphicsScene()
        self.scene_overlay = QGraphicsScene()

        self.graph_image.setScene(self.scene_image)
        self.graph_image.setMouseTracking(True)
        self.graph_image.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.graph_final.setScene(self.scene_final)
        self.graph_final.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.graph_init.setScene(self.scene_init)
        self.graph_init.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.graph_mask.setScene(self.scene_mask)
        self.graph_mask.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.graph_overlay.setScene(self.scene_overlay)
        self.graph_overlay.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def processHandler(self, message):
        mess_type, message = message.mess_type, message.pos
        if self.image is None: return
        if mess_type in [0, 1, 3]:
            self.click_pos = message if mess_type in [0, 1] else self.click_pos

            y, x = self.image.shape[:2]
            # 显示图像
            frame = QImage(self.image, x, y, 3 * x, QImage.Format_RGB888)
            if not mess_type == 1:
                self.scene_image.clear()
                self.scene_image.addPixmap(QPixmap.fromImage(frame))
            if mess_type in [0, 1]:
                self.scene_image.addEllipse(message.x() - 3.0, message.y() - 3.0, 6.0, 6.0,
                                            QPen(QColor(255, 0, 0)), QBrush(QColor(255, 0, 0)))
            else:
                x1, x2 = (self.click_pos.x(), message.x()) if self.click_pos.x() < message.x() else \
                    (message.x(), self.click_pos.x())
                y1, y2 = (self.click_pos.y(), message.y()) if self.click_pos.y() < message.y() else \
                    (message.y(), self.click_pos.y())

                self.scene_image.addRect(x1 - 3, y1 - 3, x2 - x1, y2 - y1,
                                         QPen(QColor(255, 0, 0), 6, Qt.SolidLine), )  # QBrush(QColor(255, 0, 0))

        if ((mess_type == 0 and self.type_id == 1) or (mess_type == 2 and self.type_id == 2)
                or (mess_type == 4 and self.type_id == 3)):
            self.iSegProcess(message)

    def iSegProcess(self, click_pos: QPoint):
        if self.iSeg.self_attn is None or self.type_id == 0:
            self.process_signals.show_process.emit("processing...")
            # 数据准备
            img = torch.tensor(self.image / 255, device=self.iSeg.device)
            img = img.half() if self.iSeg.half else img
            img = img.permute(2, 0, 1)[None]
            # 送入网络
            if self.type_id == 0:
                text = self.lineEdit.text().strip()
                if "".__eq__(text):
                    return
                mask = self.iSeg.stable_diffusion.tokenizer(text.split(";"), padding="max_length").attention_mask
                mask = [sum(m) - 2 for m in mask]
                sel_idx, start_ids = [], []
                for idx, l in enumerate(mask):
                    start_ids.append(start_ids[idx - 1] + 1 + mask[idx] if idx > 0 else 4)
                    sel_idx.append([start_ids[-1] + sel_id for sel_id in range(mask[idx])])
                text = text.replace(";", " and ")
                self.iSeg.test_step((img, f"a photography of {text} and other object and background",
                                     sel_idx), 0)
            else:
                self.iSeg.test_step((img, " ", [[0]]), 0)
        self_attn = self.iSeg.self_attn.clone()

        if self.type_id == 0:
            cross_attn = self.iSeg.cross_attn.clone()
            cross_attn = cross_attn.permute(0, 2, 1).reshape(1, -1, 64, 64)
            cross_attn -= cross_attn.amin(dim=(-2, -1), keepdim=True)
            cross_attn /= cross_attn.amax(dim=(-2, -1), keepdim=True)
        elif self.type_id == 1:  # Point
            cross_attn = torch.zeros((1, 1, 64, 64), device=self.iSeg.device)
            cross_attn[:, :, click_pos.y() * 64 // self.image.shape[0], click_pos.x() * 64 // self.image.shape[1]] = 1
        elif self.type_id == 2:
            cross_attn = torch.zeros((1, 1, 64, 64), device=self.iSeg.device)
            for click_pos in self.scene_image.pos_list:
                cross_attn[:, :, click_pos.y() * 64 // self.image.shape[0],
                click_pos.x() * 64 // self.image.shape[1]] = 1
            ###################################
            idx = torch.nonzero(cross_attn.flatten(), as_tuple=False).squeeze()
            cross_attn_ = torch.eye(4096, device=self.iSeg.device)[:, idx]
            cross_attn = cross_attn_.reshape(1, 64, 64, -1).permute(0, 3, 1, 2)
            ###################################
        elif self.type_id == 3:
            cross_attn = torch.zeros((1, 1, 64, 64), device=self.iSeg.device)
            x1, x2 = (self.click_pos.x(), click_pos.x()) if self.click_pos.x() < click_pos.x() else \
                (click_pos.x(), self.click_pos.x())
            x1, x2 = x1 * 64 // self.image.shape[1], x2 * 64 // self.image.shape[1]
            y1, y2 = (self.click_pos.y(), click_pos.y()) if self.click_pos.y() < click_pos.y() else \
                (click_pos.y(), self.click_pos.y())
            y1, y2 = y1 * 64 // self.image.shape[0], y2 * 64 // self.image.shape[0]
            cross_attn[:, :, y1: y2, x1: x2] = 1
        else:
            return None

        self.init_cross = F.interpolate(cross_attn.sum(dim=-3, keepdims=True),
                                        size=(self.image.shape[:2]), mode='nearest')[0, 0]
        cross_attn = cross_attn.permute(0, 2, 3, 1).reshape(1, 4096, -1)
        # iteration
        cross_attn = cross_attn - cross_attn.amin(dim=-2, keepdim=True)
        cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)  # 归一化

        self_attn /= self_attn.sum(dim=-1, keepdim=True)

        self_attn /= torch.amax(self_attn, dim=-2, keepdim=True)
        ent = float(self.textEdit_entself.text())
        self_attn += torch.where(self_attn == 0, 0, ent * (torch.log10(torch.e * self_attn)))
        self_attn = torch.clamp(self_attn, min=0, max=1)
        self_attn /= self_attn.sum(dim=-1, keepdim=True)

        iter_ = self.slider_iter.value()
        for _ in range(iter_):
            cross_attn = torch.bmm(self_attn, cross_attn)
            cross_attn -= cross_attn.amin(dim=-2, keepdim=True)
            cross_attn /= cross_attn.sum(dim=-2, keepdim=True)

        cross_attn = cross_attn / cross_attn.amax(dim=-2, keepdim=True)
        cross_attn = cross_attn.mean(-1, keepdim=True)
        cross_attn = cross_attn / cross_attn.amax(dim=-2, keepdim=True)
        cross_attn = cross_attn.permute(0, 2, 1).reshape(1, -1, 64, 64)
        cross_attn = F.interpolate(cross_attn, size=(self.image.shape[:2]),
                                   mode='bilinear', align_corners=False)[0]
        self.final_cross = cross_attn.clone()
        mask = torch.cat((torch.ones_like(cross_attn[[0], :, :]) * 0.4, cross_attn), dim=0)
        self.mask = mask.argmax(dim=0).squeeze()
        eroded, bound = self.iSeg.get_boundry_and_eroded_mask(self.mask.cpu())
        overlay = self.iSeg.get_colored_segmentation(torch.tensor(eroded), torch.tensor(bound),
                                                torch.tensor(self.image).permute(2, 0, 1),
                                                generate_distinct_colors()).cpu().numpy()
        self.overlay = np.uint8(overlay)
        self.show_result()

    def show_result(self):

        def show(graph, scene, img, direct=False, type_=0):
            import cv2
            if direct:
                heat_map = img
            else:
                type_ = cv2.COLORMAP_SPRING if type_ == 1 else cv2.COLORMAP_JET
                heat_map = self.iSeg.show_cam_on_image(img, type_)
            y, x = heat_map.shape[:2]
            my, mx = graph.size().height(), graph.size().width()
            scale = my / y if my / y < mx / x else mx / x
            heat_map = cv2.resize(heat_map, (int(scale * x), int(scale * y)))
            y, x = heat_map.shape[:2]
            frame = QImage(np.ascontiguousarray(heat_map), x, y, 3 * x, QImage.Format_RGB888)
            scene.clear()
            scene.addPixmap(QPixmap.fromImage(frame))

        # final
        show(self.graph_final, self.scene_final, self.final_cross[0])
        show(self.graph_init, self.scene_init, self.init_cross)
        show(self.graph_mask, self.scene_mask, self.mask, type_=1)
        show(self.graph_overlay, self.scene_overlay, self.overlay, direct=True)
        self.show_label.setText("finished!")

    def showHandler(self, message):
        self.show_label.setText(message)
        if "Loaded iSeg!".__eq__(message):
            self.iSeg = self.loder.iSeg
            del self.loder
            del self.thread


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = iSeg()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
