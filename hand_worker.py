import math
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "hand_landmarker.task"
)


class HandWorker(QThread):
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.enabled = threading.Event()
        self.lock = threading.Lock()
        self.latest = None

    def set_enabled(self, enabled):
        if enabled:
            self.enabled.set()
        else:
            self.enabled.clear()
            self.store(None)

    def store(self, value):
        with self.lock:
            self.latest = value

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        camera = None

        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Missing hand model: {MODEL_PATH}"
                )

            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )

            with vision.HandLandmarker.create_from_options(
                options
            ) as detector:
                last_timestamp = -1
                failed_reads = 0

                while not self.isInterruptionRequested():
                    if not self.enabled.is_set():
                        if camera is not None:
                            camera.release()
                            camera = None

                        self.store(None)
                        self.msleep(50)
                        continue

                    if camera is None:
                        camera = cv2.VideoCapture(
                            0, cv2.CAP_DSHOW
                        )

                        if not camera.isOpened():
                            raise RuntimeError(
                                "Cannot open webcam. Close other "
                                "camera apps and check Windows "
                                "camera permissions."
                            )

                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        failed_reads = 0

                    ok, frame = camera.read()
                    captured_at = time.monotonic()

                    if not ok:
                        self.store(None)
                        failed_reads += 1

                        if failed_reads >= 20:
                            raise RuntimeError(
                                "Webcam stopped returning frames."
                            )

                        self.msleep(50)
                        continue

                    failed_reads = 0

                    if (
                        self.isInterruptionRequested()
                        or not self.enabled.is_set()
                    ):
                        continue

                    # Mirror the image for natural hand movement.
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )

                    timestamp = max(
                        last_timestamp + 1,
                        int(captured_at * 1000),
                    )
                    last_timestamp = timestamp

                    result = detector.detect_for_video(
                        image, timestamp
                    )

                    if not result.hand_landmarks:
                        self.store(None)
                    else:
                        hand = result.hand_landmarks[0]
                        height, width = frame.shape[:2]

                        def distance(a, b):
                            return math.hypot(
                                (hand[a].x - hand[b].x) * width,
                                (hand[a].y - hand[b].y) * height,
                            )

                        # Normalize finger gap by palm size.
                        palm_size = max(distance(0, 9), 1.0)
                        pinch_ratio = distance(4, 8) / palm_size

                        palm_indices = (0, 5, 9, 13, 17)

                        x = sum(
                            hand[i].x for i in palm_indices
                        ) / len(palm_indices)

                        y = sum(
                            hand[i].y for i in palm_indices
                        ) / len(palm_indices)

                        self.store(
                            (captured_at, x, y, pinch_ratio)
                        )

                    self.msleep(15)

        except Exception as error:
            self.failed.emit(
                f"{type(error).__name__}: {error}"
            )

        finally:
            self.store(None)

            if camera is not None:
                camera.release()