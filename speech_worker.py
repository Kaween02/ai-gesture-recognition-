import queue

import pythoncom
import win32com.client
from PySide6.QtCore import QThread, Signal


class SpeechWorker(QThread):
    failed = Signal(str)

    def __init__(self, speaking_event, parent=None):
        super().__init__(parent)
        self.speaking_event = speaking_event
        self.messages = queue.Queue(maxsize=3)

    def say(self, text):
        if self.isInterruptionRequested() or not self.isRunning():
            return

        try:
            self.messages.put_nowait(text)
        except queue.Full:
            print("Speech queue full: reply skipped.")

    def run(self):
        voice = None
        com_initialized = False

        try:
            pythoncom.CoInitialize()
            com_initialized = True

            # Create and use the COM object in this worker thread.
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.Rate = 0
            voice.Volume = 85

            while not self.isInterruptionRequested():
                try:
                    text = self.messages.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.isInterruptionRequested():
                    break

                self.speaking_event.set()

                try:
                    print("Ultron:", text)

                    # 1 = asynchronous; 16 = treat input as plain text.
                    voice.Speak(text, 1 | 16)

                    while not voice.WaitUntilDone(50):
                        if self.isInterruptionRequested():
                            # Purge speech still playing.
                            voice.Speak("", 2)
                            break

                    # Keep the microphone gate closed briefly for audio tail.
                    if not self.isInterruptionRequested():
                        self.msleep(300)

                finally:
                    self.speaking_event.clear()

        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")

        finally:
            self.speaking_event.clear()
            voice = None

            if com_initialized:
                pythoncom.CoUninitialize()