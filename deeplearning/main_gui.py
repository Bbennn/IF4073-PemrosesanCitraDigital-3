import os
import sys

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget, QHBoxLayout, QFrame
)

# =====================================================================
#  LOAD MODEL MODULES
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from segmentation_model_pretrained import SegmentationModelPreTrained
from segmentation_model import SegmentationModel


# =====================================================================
#  WORKER THREAD (AGAR GUI TIDAK FREEZE)
# =====================================================================
class InferenceWorker(QThread):
    finished = pyqtSignal(object, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, logic_model, input_image):
        super().__init__()
        self.logic = logic_model
        self.input_image = input_image

    def run(self):
        try:
            if self.logic.model is None:
                self.logic.load_model()

            result_image, predictions = self.logic.predict(self.input_image)
            self.finished.emit(result_image, predictions)

        except Exception as e:
            self.error.emit(str(e))


# =====================================================================
#  MAIN GUI
# =====================================================================
class SemanticSegApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Deep Learning Segmentation (PyQt6)")
        self.setGeometry(100, 100, 1300, 700)
        self.setStyleSheet("background-color: #2c3e50;")

        # default logic = custom model
        self.logic = SegmentationModel()
        self.current_image = None

        self.use_pretrained = False  # toggle state

        self.init_ui()

    # -----------------------------------------------------------------
    #  UI
    # -----------------------------------------------------------------
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # -------------------------------------------------------------
        #  TOP BUTTONS
        # -------------------------------------------------------------
        btn_layout = QHBoxLayout()

        # TOGGLE BUTTON
        self.btn_toggle = QPushButton("Mode: CUSTOM")
        self.btn_toggle.setFixedHeight(50)
        self.btn_toggle.setStyleSheet(
            "background-color: #9b59b6; color: white; font-size: 18px; font-weight: bold; border-radius: 5px;")
        self.btn_toggle.clicked.connect(self.toggle_model)
        btn_layout.addWidget(self.btn_toggle)

        # LOAD IMAGE
        self.btn_load = QPushButton("Buka Gambar")
        self.btn_load.setFixedHeight(50)
        self.btn_load.setStyleSheet(
            "background-color: #3498db; color: white; font-size: 20px; font-weight: bold; border-radius: 5px;")
        self.btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load)

        # START SEGMENTATION
        self.btn_process = QPushButton("Mulai Segmentasi")
        self.btn_process.setFixedHeight(50)
        self.btn_process.setStyleSheet(
            "background-color: #e74c3c; color: white; font-size: 20px; font-weight: bold; border-radius: 5px;")
        self.btn_process.clicked.connect(self.start_segmentation)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)

        main_layout.addLayout(btn_layout)

        # -------------------------------------------------------------
        # STATUS LABEL
        # -------------------------------------------------------------
        self.lbl_status = QLabel("Status: Siap. Silakan buka gambar.")
        self.lbl_status.setStyleSheet("color: #ecf0f1; font-size: 12px; padding: 5px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.lbl_status)

        # -------------------------------------------------------------
        # IMAGE AREA
        # -------------------------------------------------------------
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # LEFT IMAGE
        self.lbl_img_input = QLabel("Input Image")
        self.lbl_img_input.setStyleSheet("background-color: #bdc3c7; border: 2px solid #7f8c8d;")
        self.lbl_img_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_input.setFixedSize(500, 450)
        content_layout.addWidget(self.lbl_img_input)

        # RIGHT IMAGE
        self.lbl_img_output = QLabel("Result Image")
        self.lbl_img_output.setStyleSheet("background-color: #bdc3c7; border: 2px solid #7f8c8d;")
        self.lbl_img_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_output.setFixedSize(500, 450)
        content_layout.addWidget(self.lbl_img_output)

        # -------------------------------------------------------------
        # LABEL LEGEND
        # -------------------------------------------------------------
        self.legend_scroll_area = QScrollArea()
        self.legend_scroll_area.setWidgetResizable(True)
        self.legend_scroll_area.setFixedWidth(250)
        self.legend_scroll_area.setStyleSheet("background-color: #34495e; border: 1px solid #7f8c8d;")

        legend_content_widget = QWidget()
        self.legend_layout = QVBoxLayout(legend_content_widget)
        self.legend_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.legend_scroll_area.setWidget(legend_content_widget)

        content_layout.addWidget(self.legend_scroll_area)

        self.update_legend(None)

    # =================================================================
    #  MODEL TOGGLE
    # =================================================================
    def toggle_model(self):
        self.use_pretrained = not self.use_pretrained

        if self.use_pretrained:
            self.logic = SegmentationModelPreTrained()
            self.btn_toggle.setText("Mode: PRETRAINED")
            self.btn_toggle.setStyleSheet(
                "background-color: #16a085; color:white; font-size:18px; font-weight:bold; border-radius:5px;")
            self.lbl_status.setText("Model diganti: PRETRAINED")
        else:
            self.logic = SegmentationModel()
            self.btn_toggle.setText("Mode: CUSTOM")
            self.btn_toggle.setStyleSheet(
                "background-color: #9b59b6; color:white; font-size:18px; font-weight:bold; border-radius:5px;")
            self.lbl_status.setText("Model diganti: CUSTOM")

    # =================================================================
    #  LOAD IMAGE
    # =================================================================
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_name:
            try:
                pil_img = Image.open(file_name).convert("RGB")
                max_size = 1000
                if pil_img.width > max_size or pil_img.height > max_size:
                    pil_img.thumbnail((max_size, max_size))

                self.current_image = pil_img
                self.display_image(self.current_image, self.lbl_img_input)
                self.lbl_img_output.clear()
                self.lbl_img_output.setText("Result Image")

                self.btn_process.setEnabled(True)
                self.btn_process.setStyleSheet(
                    "background-color: #2ecc71; color: white; font-size: 14px; font-weight: bold; border-radius: 5px;")

                self.lbl_status.setText(f"Gambar dimuat: {os.path.basename(file_name)}")
                self.update_legend(None)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal memuat gambar: {str(e)}")

    # =================================================================
    #  RUN SEGMENTATION
    # =================================================================
    def start_segmentation(self):
        if self.current_image is None:
            return

        self.lbl_status.setText("Sedang memproses...")
        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)

        self.worker = InferenceWorker(self.logic, self.current_image)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    # =================================================================
    #  RESULT CALLBACKS
    # =================================================================
    def on_inference_finished(self, result_pil_image, predictions_array):
        self.display_image(result_pil_image, self.lbl_img_output)
        self.lbl_status.setText("Segmentasi selesai!")
        self.btn_process.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.update_legend(predictions_array)
        QMessageBox.information(self, "Sukses", "Proses segmentasi berhasil.")

    def on_inference_error(self, error_msg):
        self.lbl_status.setText("Terjadi kesalahan.")
        self.btn_process.setEnabled(True)
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "Error Inference", error_msg)
        self.update_legend(None)

    # =================================================================
    #  DISPLAY IMAGE
    # =================================================================
    def display_image(self, pil_image, qlabel_widget):
        im2 = pil_image.copy()
        if im2.mode != "RGB":
            im2 = im2.convert("RGB")

        qim = ImageQt(im2)
        pixmap = QPixmap.fromImage(qim)

        scaled_pixmap = pixmap.scaled(
            qlabel_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        qlabel_widget.setPixmap(scaled_pixmap)

    # =================================================================
    #  LEGEND
    # =================================================================
    def update_legend(self, predictions_array):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        title_label = QLabel("Deteksi Objek:")
        title_label.setStyleSheet("color:white; font-size:14px; font-weight:bold; margin-bottom:5px;")
        self.legend_layout.addWidget(title_label)

        if predictions_array is None:
            no_obj_label = QLabel("Belum ada objek terdeteksi.")
            no_obj_label.setStyleSheet("color:#bdc3c7; font-size:11px;")
            self.legend_layout.addWidget(no_obj_label)
            return

        unique_classes = np.unique(predictions_array)
        detected_objects = []

        for class_id in unique_classes:
            if class_id == 0:
                continue

            label_name = self.logic.VOC_LABELS.get(class_id, f"Unknown ({class_id})")
            color_rgb = self.logic.VOC_COLORS[class_id]

            legend_item_frame = QFrame()
            legend_item_frame.setStyleSheet(
                "background-color:#3e526a; border-radius:3px; padding:3px;")
            legend_item_layout = QHBoxLayout(legend_item_frame)
            legend_item_layout.setContentsMargins(5, 2, 5, 2)

            color_box = QLabel()
            color_box.setFixedSize(20, 15)
            color_box.setStyleSheet(
                f"background-color: rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]}); "
                "border:1px solid white;")
            legend_item_layout.addWidget(color_box)

            text_label = QLabel(label_name.capitalize())
            text_label.setStyleSheet("color:white; font-size:12px;")
            legend_item_layout.addWidget(text_label)

            legend_item_layout.addStretch()
            detected_objects.append(legend_item_frame)

        if not detected_objects:
            no_obj_label = QLabel("Tidak ada objek yang dikenal model.")
            no_obj_label.setStyleSheet("color:#bdc3c7; font-size:11px;")
            self.legend_layout.addWidget(no_obj_label)
        else:
            for item in detected_objects:
                self.legend_layout.addWidget(item)


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SemanticSegApp()
    window.show()
    sys.exit(app.exec())