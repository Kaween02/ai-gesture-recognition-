import sys

import moderngl
from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtGui import QShortcut, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QStyle,
    QSystemTrayIcon,
)

from speech_worker import SpeechWorker
from voice_worker import VoiceWorker
import math
import time

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel

from hand_worker import HandWorker
from PySide6.QtWidgets import QFileDialog, QMessageBox
from workspace_renderer import WorkspaceRenderer
from voice_actions import VoiceActions

class WorkspaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ultron — Workspace")
        self.resize(1200, 760)

        self.renderer = WorkspaceRenderer(self)
        self.setCentralWidget(self.renderer)
        toolbar = self.addToolBar("Model controls")
        toolbar.setMovable(False)

        load_action = toolbar.addAction("Load 3D model")
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.choose_model)

        wireframe_action = toolbar.addAction("Wireframe")
        wireframe_action.setCheckable(True)
        wireframe_action.toggled.connect(
            self.renderer.set_wireframe
        )

        reset_action = toolbar.addAction("Reset view")
        reset_action.triggered.connect(
            self.renderer.reset_view
        )

        toolbar.setStyleSheet("""
            QToolBar {
                background: #111820;
                border: none;
                spacing: 8px;
                padding: 8px;
            }

            QToolButton {
                color: #eeeeee;
                background: #202b36;
                border: 1px solid #344454;
                border-radius: 5px;
                padding: 7px 12px;
            }

            QToolButton:hover {
                border-color: #fe5119;
            }

            QToolButton:checked {
                background: #a53611;
                border-color: #fe5119;
            }
        """)

        self.statusBar().showMessage(
            "Voice starting | Esc: hide | Ctrl+Shift+Q: quit"
        )

        self.hand_label = QLabel("Hand: camera off")
        self.statusBar().addPermanentWidget(self.hand_label)

        self.hand = HandWorker(self)
        self.hand.failed.connect(
            self.show_hand_error,
            Qt.ConnectionType.QueuedConnection,
        )

        self.hand_error = None
        self.previous_hand = None
        self.pinching = False
        self.last_sample = 0.0
        self.opened_at = 0.0

        self.hand_timer = QTimer(self)
        self.hand_timer.setInterval(33)
        self.hand_timer.timeout.connect(self.update_hand)

        self.hide_shortcut = QShortcut(
            Qt.Key.Key_Escape, self
        )
        self.hide_shortcut.activated.connect(self.hide)

    @Slot(str)
    def show_hand_error(self, message):
        self.hand_error = message
        self.previous_hand = None
        self.hand_label.setText("Hand: camera error")
        self.hand_label.setToolTip(message)
        print("Hand tracking error:", message)

    def showEvent(self, event):
        super().showEvent(event)

        self.opened_at = time.monotonic()
        self.previous_hand = None
        self.pinching = False

        self.hand.set_enabled(True)
        self.hand_timer.start()

    def hideEvent(self, event):
        self.hand_timer.stop()
        self.hand.set_enabled(False)

        self.previous_hand = None
        self.pinching = False
        self.hand_label.setText("Hand: camera off")

        super().hideEvent(event)

    def update_hand(self):
        if self.hand_error:
            self.hand_label.setText("Hand: camera error")
            return

        sample = self.hand.snapshot()
        now = time.monotonic()

        if (
            sample is None
            or sample[0] < self.opened_at
            or now - sample[0] > 0.3
        ):
            self.previous_hand = None
            self.pinching = False
            self.hand_label.setText("Hand: show one hand")
            return

        captured_at, x, y, ratio = sample

        # Process each camera result only once.
        if captured_at <= self.last_sample:
            return

        self.last_sample = captured_at
        was_pinching = self.pinching

        # Separate thresholds prevent rapid mode switching.
        if self.pinching:
            self.pinching = ratio < 0.40
        else:
            self.pinching = ratio < 0.25

        if (
            self.previous_hand is None
            or self.pinching != was_pinching
        ):
            self.previous_hand = (x, y)
            self.hand_label.setText(
                "Hand: pinch + move vertically to zoom"
                if self.pinching
                else "Hand: move horizontally to rotate"
            )
            return

        previous_x, previous_y = self.previous_hand

        # Smooth small tracking fluctuations.
        smooth_x = previous_x + (x - previous_x) * 0.35
        smooth_y = previous_y + (y - previous_y) * 0.35

        dx = smooth_x - previous_x
        dy = smooth_y - previous_y

        self.previous_hand = (smooth_x, smooth_y)

        if self.pinching:
            # Moving upward reduces camera distance.
            self.renderer.distance = max(
                4.0,
                min(
                    18.0,
                    self.renderer.distance + dy * 20.0,
                ),
            )
            self.hand_label.setText("Hand: zoom")
        else:
            self.renderer.hand_angle = (
                self.renderer.hand_angle + dx * 7.0
            ) % math.tau
            self.hand_label.setText("Hand: rotate")

        self.renderer.update()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        
    @Slot()
    def choose_model(self):
        # Pause gesture updates while choosing/loading a file.
        self.hand_timer.stop()
        self.previous_hand = None

        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Open 3D model",
                "",
                "3D models (*.glb *.obj *.stl)",
            )

            if not filename:
                return

            self.statusBar().showMessage("Loading 3D model...")

            face_count = self.renderer.load_model(filename)

            self.setWindowTitle(
                f"Ultron — {self.renderer.model_name}"
            )
            self.statusBar().showMessage(
                f"Loaded: {self.renderer.model_name} | "
                f"{face_count:,} triangles",
                8000,
            )

        except Exception as error:
            QMessageBox.warning(
                self,
                "Could not load model",
                str(error),
            )

        finally:
            # Require a fresh hand position after the dialog.
            self.previous_hand = None
            self.pinching = False
            self.opened_at = time.monotonic()

            if self.isVisible():
                self.hand_timer.start()

class JarvisApplication(QObject):
    def __init__(self, app):
        super().__init__(app)

        self.app = app
        self.shutting_down = False
        self.quit_finished = False

        self.workspace = WorkspaceWindow()

        self.quit_shortcut = QShortcut(
            "Ctrl+Shift+Q",
            self.workspace,
        )
        self.quit_shortcut.activated.connect(self.quit)

        icon = app.style().standardIcon(
            QStyle.StandardPixmap.SP_ComputerIcon
        )
        self.workspace.setWindowIcon(icon)

        self.tray = QSystemTrayIcon(icon, app)
        self.tray.setToolTip("Ultron — starting")

        self.tray_menu = QMenu()
        self.tray_menu.addAction(
            "Open workspace",
            self.open_workspace,
        )
        self.tray_menu.addAction(
            "Close workspace",
            self.close_workspace,
        )
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(
            "Quit Ultron",
            self.quit,
        )

        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()

        self.voice = VoiceWorker(self)

        # Both workers use the same playback event.
        self.speech = SpeechWorker(
            self.voice.speaking_event,
            self,
        )
        self.actions = VoiceActions(self)

        self.voice.command_received.connect(
            self.handle_voice_command,
            Qt.ConnectionType.QueuedConnection,
        )
        self.voice.transcript_received.connect(
            self.show_transcript,
            Qt.ConnectionType.QueuedConnection,
        )
        self.voice.status_changed.connect(
            self.show_voice_status,
            Qt.ConnectionType.QueuedConnection,
        )
        self.voice.failed.connect(
            self.show_voice_error,
            Qt.ConnectionType.QueuedConnection,
        )
        self.speech.failed.connect(
            self.show_speech_error,
            Qt.ConnectionType.QueuedConnection,
        )

        self.voice.finished.connect(
            self.on_workers_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self.speech.finished.connect(
            self.on_workers_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        self.workspace.hand.finished.connect(
            self.on_workers_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        self.workspace.hand.start()
        self.speech.start()
        self.voice.start()

    @Slot()
    def open_workspace(self):
        if self.shutting_down:
            return

        self.workspace.showNormal()
        self.workspace.raise_()
        self.workspace.activateWindow()

    @Slot()
    def close_workspace(self):
        self.workspace.hide()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_workspace()

    @Slot(str)
        # @Slot(str)
    def handle_voice_command(self, command):
        if self.shutting_down:
            return

        if self.voice.speaking_event.is_set():
            return

        print("Command:", command)
        self.actions.handle(command)
    @Slot(str)
    def show_transcript(self, text):
        if self.shutting_down:
            return

        print("Heard:", text or "[no speech detected]")

    @Slot(str)
    def show_voice_status(self, status):
        if self.shutting_down:
            return

        self.tray.setToolTip(f"Ultron — {status}")
        self.workspace.statusBar().showMessage(
            f"Voice: {status} | Esc: hide | Ctrl+Shift+Q: quit"
        )

    @Slot(str)
    def show_voice_error(self, message):
        print("Voice error:", message)

        if self.shutting_down:
            return

        self.tray.setToolTip("Ultron — voice error")
        self.workspace.statusBar().showMessage(
            f"Voice error: {message}"
        )

    @Slot(str)
    def show_speech_error(self, message):
        print("Speech output error:", message)

        if self.shutting_down:
            return

        self.tray.setToolTip("Ultron — speech output error")
        self.workspace.statusBar().showMessage(
            f"Speech output error: {message}"
        )

    @Slot()
    def quit(self):
        if self.shutting_down:
            return

        self.shutting_down = True

        self.tray.setToolTip("Ultron — stopping")
        self.workspace.statusBar().showMessage(
            "Stopping voice, speech and camera workers..."
        )

        self.workspace.hand_timer.stop()
        self.workspace.hand.set_enabled(False)

        self.workspace.hand.requestInterruption()
        self.voice.requestInterruption()
        self.speech.requestInterruption()

        self.on_workers_finished()

    @Slot()
    def on_workers_finished(self):
        if not self.shutting_down or self.quit_finished:
            return

        if (
            self.voice.isRunning()
            or self.speech.isRunning()
            or self.workspace.hand.isRunning()
        ):
            return

        self.quit_finished = True

        self.workspace.renderer.release_gpu()
        self.tray.hide()
        self.app.quit()
def main():
    graphics_format = QSurfaceFormat()
    graphics_format.setVersion(3, 3)
    graphics_format.setProfile(
        QSurfaceFormat.OpenGLContextProfile.CoreProfile
    )
    graphics_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(graphics_format)

    app = QApplication(sys.argv)
    app.setApplicationName("Ultron")
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("Windows system tray is unavailable.")
        return 1

    # Keep the controller alive throughout the event loop.
    controller = JarvisApplication(app)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())