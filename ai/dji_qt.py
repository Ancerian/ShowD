import cv2
import time
import numpy as np
from queue import Queue, Empty
from collections import Counter

from djitellopy import Tello, TelloSwarm
from mp.scan import search_tello

from mp.run import recognize_action, classes

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QSizePolicy, QPushButton, QMessageBox, QStatusBar, QStyle, QFormLayout, QLineEdit, QSlider
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

class TelloControlThread(QThread):
    command_executed = pyqtSignal(str)  # Signal to emit when command is executed
    error_occurred = pyqtSignal(str)    # Signal to emit when error occurs
    connection_status = pyqtSignal(bool)  # Signal to emit connection status
    
    def __init__(self):
        super().__init__()
        self.tello = None
        self.height = 100
        self.command_queue = Queue()
        self.running = True


        #standard limits
        self.max_height = 250
        self.min_height = 40
        self.move_step = 20
        self.horizontal_step = 100
        
    def initialize_drone(self):
        try:
            self.tello = Tello()
            self.tello.connect(False)
            self.command_executed.emit("Drone connected")
            self.tello.takeoff()
            self.command_executed.emit("Takeoff successful")
            self.connection_status.emit(True)
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.connection_status.emit(False)
            self.tello = None
            return False

    def initialize_swarm(self):
        try:
            self.tello = TelloSwarm.fromIps(search_tello())
            self.tello.connect(False)
            self.command_executed.emit("Swarm connected")
            self.tello.takeoff()
            self.command_executed.emit("Takeoff successful")
            self.connection_status.emit(True)
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.connection_status.emit(False)
            self.tello = None
            return False
        
    def add_command(self, command):
        print("Queueing command:", command)
        self.command_queue.put((command, time.time()))
        
    def run(self):
        while self.running:
            try:
                command, timestamp = self.command_queue.get(timeout=0.1)  # Wait for 100ms for a command
                if time.time() - timestamp > 1:
                    print("Dequeued command expired:", command)
                    continue  # Skip expired commands

                print("Dequeued command:", command)

                if command == "takeoff":
                    self.initialize_drone()
                    self.command_executed.emit("Takeoff successful")
                elif command == "takeoff_swarm":
                    self.initialize_swarm()
                    self.command_executed.emit("Takeoff successful")
                elif command == "up":
                    if self.height < self.max_height:
                        self.tello.move_up(self.move_step)
                        self.height += self.move_step
                        self.command_executed.emit("Moved up")
                    else:
                        self.error_occurred.emit("Maximum height reached")
                elif command == "down":
                    if self.height > self.min_height:
                        self.tello.move_down(self.move_step)
                        self.height -= self.move_step
                        self.command_executed.emit("Moved down")
                    else:
                        self.error_occurred.emit("Already at minimum height")
                elif command == "left":
                    self.tello.move_left(self.horizontal_step)
                    self.command_executed.emit("Moved left")
                elif command == "right":
                    self.tello.move_right(self.horizontal_step)
                    self.command_executed.emit("Moved right")
                elif command == "land":
                    self.tello.land()
                    self.command_executed.emit("Landed")
                    break  # Stop the thread after landing
            except Empty:
                # No command available, continue
                pass
            except Exception as e:
                self.error_occurred.emit(str(e))
            
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self, with_drone=False, with_swarm=False):
        super().__init__()

        self.setWindowTitle("Action Recognition + Drone Control")
        self.resize(1100, 800)
        self.setMinimumSize(600, 400)
        self.setMaximumSize(1600, 1200)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create status bar
        self.statusBar().showMessage("Ready")

        # Современный белый фон
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8fafc, stop:1 #e3e9f3);
            }
        """)

        # Видео-рамка
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.Box)
        self.image_frame.setLineWidth(0)
        self.image_frame.setStyleSheet("""
            QFrame {
                background: #222;
                border-radius: 18px;
                border: 3px solid #b2becd;
                margin: 0px;
            }
        """)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: #111; border-radius: 12px;")
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(8, 8, 8, 8)
        image_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_layout)

        # Текст класса
        self.text_label = QLabel("Predicted class: ...")
        self.text_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color: #1a237e; background: #e3e9f3; border-radius: 10px; padding: 16px 0; margin-top: 18px; margin-bottom: 8px; font-weight: 700;")

        # Кнопка посадки дрона (только если с дроном)
        self.land_button = None
        self.with_drone = with_drone
        if self.with_drone:
            self.land_button = QPushButton("Land (Посадити дрона)")
            self.land_button.setStyleSheet("font-size: 22px; background: #e74c3c; color: white; border-radius: 10px; padding: 12px 0; font-weight: bold;")
            self.land_button.setMinimumHeight(48)
            self.land_button.clicked.connect(self.land_drone)

        # Кнопка ⚙️ для перехода на следующее окно
        self.settings_button = QPushButton("⚙️")
        self.settings_button.setToolTip("Настройки")
        self.settings_button.setStyleSheet("font-size: 24px; background: #3498db; border-radius: 10px; padding: 8px; font-weight: bold;")
        self.settings_button.setFixedSize(50, 50)
        self.settings_button.clicked.connect(self.open_settings_window)

        # Главный layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(32, 32, 32, 32)
        main_layout.setSpacing(18)
        main_layout.addWidget(self.image_frame, stretch=8)
        main_layout.addWidget(self.text_label, stretch=1)
        if self.land_button:
            main_layout.addWidget(self.land_button, stretch=0)
        main_layout.addWidget(self.settings_button, alignment=Qt.AlignRight)  # Кнопка справа
        self.central_widget.setLayout(main_layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Дрон и контроллер дрона
        self.tello_control = TelloControlThread()
        self.tello_control.command_executed.connect(lambda msg: self.statusBar().showMessage(msg, 2000))
        self.tello_control.error_occurred.connect(lambda err: QMessageBox.warning(self, "Drone Error", err))
        self.tello_control.connection_status.connect(self._handle_drone_connection)
        self.predictions_buffer = []
        self.frames_to_accumulate = 10
        
        if self.with_drone:
            self.tello_control.start()
            if with_swarm:
                self.tello_control.add_command("takeoff_swarm")
            else:
                self.tello_control.add_command("takeoff")
    def open_settings_window(self):
        self.settings_window = SettingsWindow(self.tello_control)
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()
            
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        response = recognize_action(imgRGB.copy())
        if response:
            prediction, imgRGB_draw = response
        else:
            prediction = None
            imgRGB_draw = imgRGB

        h, w, ch = imgRGB_draw.shape
        bytes_per_line = ch * w
        qt_img = QImage(imgRGB_draw.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Адаптивное масштабирование изображения с отступами
        label_w = max(1, self.image_label.width() - 16)
        label_h = max(1, self.image_label.height() - 16)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            label_w, label_h,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        if prediction is not None:
            self.text_label.setText(f"Predicted class: {classes[int(prediction[0][0])]}")
            if self.with_drone:
                self.predictions_buffer.append(classes[int(prediction[0][0])])
                if len(self.predictions_buffer) == self.frames_to_accumulate:
                    most_common = Counter(self.predictions_buffer).most_common(1)[0][0]
                    self.tello_control.add_command(most_common)
                    self.predictions_buffer = []
        else:
            self.text_label.setText("Predicted class: ...")

    def land_drone(self):
        self.tello_control.add_command("land")
        QMessageBox.information(self, "Дрон", "Дрон посаджено.")
        # закрити вікно
        self.close()

    def _handle_drone_connection(self, connected):
        """Handle drone connection status changes"""
        if not connected and self.land_button:
            self.land_button.hide()
            self.with_drone = False

    def closeEvent(self, event):
        self.cap.release()
        if self.with_drone:
            try:
                self.tello_control.add_command("land")  # Send land command
                self.tello_control.stop()
                self.tello_control.wait()  # Wait for thread to finish
            except Exception:
                pass
        event.accept()

class SettingsWindow(QWidget):
    def __init__(self, tello_control):
        super().__init__()
        self.setWindowTitle("Settings")
        self.resize(400, 300)
        self.tello_control = tello_control

        layout = QVBoxLayout()
        form = QFormLayout()

        def make_slider_input(label, min_val, max_val, current_val):
            container = QHBoxLayout()

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(current_val)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval((max_val - min_val) // 10)

            input_box = QLineEdit(str(current_val))
            input_box.setFixedWidth(60)
            input_box.setAlignment(Qt.AlignCenter)

            # Связь: ползунок → поле
            slider.valueChanged.connect(lambda val: input_box.setText(str(val)))
            # Связь: поле → ползунок
            input_box.textChanged.connect(lambda text: slider.setValue(int(text)) if text.isdigit() else None)

            container.addWidget(slider)
            container.addWidget(input_box)
            return container, slider, input_box

        # Создаём все параметры
        self.max_height_box, self.max_height_slider, self.max_height_input = make_slider_input(
            "Макс. висота", 100, 500, self.tello_control.max_height
        )
        self.min_height_box, self.min_height_slider, self.min_height_input = make_slider_input(
            "Мін. висота", 10, 200, self.tello_control.min_height
        )
        self.move_step_box, self.move_step_slider, self.move_step_input = make_slider_input(
            "Крок вгору/вниз", 5, 100, self.tello_control.move_step
        )
        self.horizontal_step_box, self.horizontal_step_slider, self.horizontal_step_input = make_slider_input(
            "Крок вліво/вправо", 10, 300, self.tello_control.horizontal_step
        )

        # Добавляем их в форму
        form.addRow("Макс. висота:", self.max_height_box)
        form.addRow("Мін. висота:", self.min_height_box)
        form.addRow("Крок вгору/вниз:", self.move_step_box)
        form.addRow("Крок вліво/вправо:", self.horizontal_step_box)

        layout.addLayout(form)

        # Кнопка сохранения
        save_button = QPushButton("💾 Зберегти і вийти")
        save_button.setStyleSheet("font-size: 18px; background: #27ae60; color: white; border-radius: 10px; padding: 8px;")
        save_button.clicked.connect(self.apply_changes_and_close)
        layout.addWidget(save_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def apply_changes_and_close(self):
        """Применить изменения и закрыть окно"""
        try:
            self.tello_control.max_height = int(self.max_height_input.text())
            self.tello_control.min_height = int(self.min_height_input.text())
            self.tello_control.move_step = int(self.move_step_input.text())
            self.tello_control.horizontal_step = int(self.horizontal_step_input.text())
        except ValueError:
            QMessageBox.warning(self, "Помилка", "Введіть лише числа.")
            return

        QMessageBox.information(self, "Збережено", "Параметри оновлено.")
        self.close()
if __name__ == "__main__":
    import sys

    class StartWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Вибір режиму")
            self.resize(480, 390)
            self.setMinimumSize(400, 350)

            # Стиль
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor("#f0f0f0"))
            self.setPalette(palette)
            self.setAutoFillBackground(True)

            # Заголовок
            label = QLabel("Оберіть режим роботи:")
            label.setFont(QFont("Segoe UI", 20, QFont.Bold))
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #1a237e; margin-bottom: 12px;")

            # Список пристроїв
            self.devices_label = QLabel("Доступні пристрої:")
            self.devices_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
            self.devices_label.setStyleSheet("color: #333; margin-top: 8px; margin-bottom: 2px;")
            self.devices_list = QLabel()
            self.devices_list.setFont(QFont("Consolas", 12))
            self.devices_list.setStyleSheet("background: #e3e9f3; border-radius: 8px; padding: 8px 12px; color: #222; min-height: 32px;")
            self.devices_list.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.refresh_devices()

            # Кнопки вибору режиму
            btn_drone = QPushButton("З дроном")
            btn_drone.setStyleSheet("font-size: 20px; background: #27ae60; color: white; border-radius: 10px; padding: 12px 0; font-weight: bold;")
    
            btn_swarm = QPushButton("З роєм дронів")
            btn_swarm.setStyleSheet("font-size: 20px; background: #27ae60; color: white; border-radius: 10px; padding: 12px 0; font-weight: bold;")
    
            btn_nodrone = QPushButton("Без дрона")
            btn_nodrone.setStyleSheet("font-size: 20px; background: #2980b9; color: white; border-radius: 10px; padding: 12px 0; font-weight: bold;")

            btn_drone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn_swarm.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn_nodrone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


            btn_drone.clicked.connect(self.start_with_drone)
            btn_swarm.clicked.connect(self.start_with_swarm)
            btn_nodrone.clicked.connect(self.start_without_drone)

            # Кнопка обновления списка
            btn_refresh = QPushButton("Оновити список")
            btn_refresh.setStyleSheet("font-size: 14px; background: #b2becd; color: #222; border-radius: 8px; padding: 4px 0;")
            btn_refresh.clicked.connect(self.refresh_devices)

            vbox = QVBoxLayout()
            vbox.setContentsMargins(32, 28, 32, 28)
            vbox.setSpacing(12)
            vbox.addWidget(label)
            vbox.addWidget(self.devices_label)
            vbox.addWidget(self.devices_list)
            vbox.addWidget(btn_refresh)
            vbox.addSpacing(8)
            vbox.addStretch()  
            vbox.addWidget(btn_drone)
            vbox.addWidget(btn_swarm)
            vbox.addWidget(btn_nodrone)
            self.setLayout(vbox)


           


        def refresh_devices(self):
            try:
                ips = search_tello()
                if not ips:
                    self.devices_list.setText("Не знайдено жодного пристрою.")
                else:
                    self.devices_list.setText("\n".join(str(ip) for ip in ips))
            except Exception as e:
                self.devices_list.setText(f"Помилка пошуку: {e}")

        def start_with_drone(self):
            self.hide()
            self.main = MainWindow(with_drone=True)
            self.main.show()
            self.main.raise_()
            self.main.activateWindow()

        def start_with_swarm(self):
            self.hide()
            self.main = MainWindow(with_drone=True, with_swarm=True)
            self.main.show()
            self.main.raise_()
            self.main.activateWindow()

        def start_without_drone(self):
            self.hide()
            self.main = MainWindow(with_drone=False)
            self.main.show()
            self.main.raise_()
            self.main.activateWindow()

    app = QApplication(sys.argv)
    start = StartWindow()
    start.show()
    sys.exit(app.exec_())
