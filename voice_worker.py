import math
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from PySide6.QtCore import QThread, Signal
from scipy.signal import resample_poly

from voice_test import MODEL_DIRECTORY, identify_command
from voice_test import (
    MODEL_DIRECTORY,
    VOICE_CONTEXT,
    identify_command,
)

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

PRE_ROLL_FRAMES = 10       # Preserve 200 ms before speech.
START_FRAMES = 3           # Require 60 ms of consecutive speech.
END_SILENCE_FRAMES = 30    # Finish after 600 ms of silence.
MIN_VOICED_FRAMES = 8      # Ignore very short bursts.
MAX_PHRASE_FRAMES = 400   # Limit phrases to about 8 seconds.

BUFFER_FRAMES = 150       # Queue up to 3 seconds of microphone audio.
MAX_RESULT_AGE = 5.0      # Discard stale recognition results.


class VoiceWorker(QThread):
    command_received = Signal(str)
    transcript_received = Signal(str)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Share this event with SpeechWorker.
        self.speaking_event = threading.Event()

    def run(self):
        try:
            self.status_changed.emit("Loading voice model")

            model = WhisperModel(
                "base.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1,
                download_root=str(MODEL_DIRECTORY),
            )

            if self.isInterruptionRequested():
                return

            self.listen(model)

        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")

    def listen(self, model):
        microphone = sd.query_devices(kind="input")
        device_rate = int(microphone["default_samplerate"])

        if device_rate % 50 != 0:
            raise ValueError(
                "Unsupported microphone rate for this checkpoint: "
                f"{device_rate}"
            )

        block_size = device_rate // 50

        divisor = math.gcd(device_rate, SAMPLE_RATE)
        up = SAMPLE_RATE // divisor
        down = device_rate // divisor

        audio_queue = queue.Queue(maxsize=BUFFER_FRAMES)
        audio_gap = threading.Event()

        def audio_callback(indata, frames, timing, status):
            # Do not capture Ultron's own speech.
            if self.speaking_event.is_set():
                audio_gap.set()
                return

            if status:
                audio_gap.set()

            try:
                audio_queue.put_nowait(
                    (time.monotonic(), indata[:, 0].copy())
                )
            except queue.Full:
                audio_gap.set()

        vad = webrtcvad.Vad(2)
        pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

        phrase = []
        active = False
        start_count = 0
        silence_count = 0
        voiced_count = 0
        recovering = False
        playback_paused = False

        print("Microphone:", microphone["name"])

        with sd.InputStream(
            samplerate=device_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ) as stream:
            self.status_changed.emit("Listening")
            print("Continuous listener ready. Say: Hey Ultron")

            while not self.isInterruptionRequested():
                if not stream.active:
                    raise RuntimeError("Microphone stream stopped.")

                # Keep processing paused throughout speech playback.
                if self.speaking_event.is_set():
                    audio_gap.set()

                    if not playback_paused:
                        playback_paused = True
                        self.status_changed.emit("Ultron speaking")

                    self.msleep(50)
                    continue

                if audio_gap.is_set():
                    audio_gap.clear()

                    # Remove old audio after playback or an audio gap.
                    for _ in range(BUFFER_FRAMES):
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break

                    pre_roll.clear()
                    phrase = []
                    active = False
                    start_count = 0
                    silence_count = 0
                    voiced_count = 0
                    recovering = True

                    if playback_paused:
                        print("Reply finished; waiting for silence.")
                    else:
                        print(
                            "Audio buffer interrupted; "
                            "waiting for silence."
                        )

                    playback_paused = False
                    self.status_changed.emit("Waiting for silence")

                try:
                    captured_at, frame = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Playback may have started while waiting for a frame.
                if self.speaking_event.is_set():
                    audio_gap.set()
                    continue

                if device_rate != SAMPLE_RATE:
                    frame = resample_poly(frame, up, down)

                frame = np.ascontiguousarray(
                    frame,
                    dtype=np.float32,
                )

                if len(frame) != FRAME_SAMPLES:
                    raise RuntimeError(
                        f"Expected {FRAME_SAMPLES} samples, "
                        f"received {len(frame)}."
                    )

                pcm = (
                    np.clip(frame, -1.0, 1.0) * 32767
                ).astype("<i2").tobytes()

                is_speech = vad.is_speech(pcm, SAMPLE_RATE)

                # Require a pause before accepting another full phrase.
                if recovering:
                    silence_count = (
                        0 if is_speech else silence_count + 1
                    )

                    if silence_count >= END_SILENCE_FRAMES:
                        recovering = False
                        silence_count = 0
                        pre_roll.clear()
                        self.status_changed.emit("Listening")

                    continue

                if not active:
                    pre_roll.append((frame, is_speech))
                    start_count = (
                        start_count + 1 if is_speech else 0
                    )

                    if start_count >= START_FRAMES:
                        active = True
                        phrase = [item[0] for item in pre_roll]
                        voiced_count = sum(
                            item[1] for item in pre_roll
                        )
                        silence_count = 0
                        self.status_changed.emit("Hearing speech")

                    continue

                phrase.append(frame)

                if is_speech:
                    voiced_count += 1
                    silence_count = 0
                else:
                    silence_count += 1

                if len(phrase) >= MAX_PHRASE_FRAMES:
                    print(
                        "Phrase too long; discarded. "
                        "Pause and retry."
                    )

                    phrase = []
                    active = False
                    recovering = True
                    start_count = 0
                    silence_count = 0
                    voiced_count = 0
                    pre_roll.clear()

                    self.status_changed.emit("Waiting for a pause")
                    continue

                if silence_count < END_SILENCE_FRAMES:
                    continue

                audio = np.concatenate(phrase)
                enough_speech = (
                    voiced_count >= MIN_VOICED_FRAMES
                )

                phrase = []
                active = False
                start_count = 0
                silence_count = 0
                voiced_count = 0
                pre_roll.clear()

                if enough_speech:
                    self.recognize(
                        model,
                        audio,
                        captured_at,
                        audio_gap,
                    )

                if (
                    not self.isInterruptionRequested()
                    and not self.speaking_event.is_set()
                ):
                    self.status_changed.emit("Listening")

    def recognize(self, model, audio, captured_at, audio_gap):
        if (
            self.isInterruptionRequested()
            or self.speaking_event.is_set()
            or audio_gap.is_set()
        ):
            return

        if time.monotonic() - captured_at > MAX_RESULT_AGE:
            print("Stale audio discarded. Please repeat.")
            return

        self.status_changed.emit("Recognizing")
        started = time.monotonic()

        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=3,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
             initial_prompt=VOICE_CONTEXT,
        )

        text = " ".join(
            segment.text.strip() for segment in segments
        )

        # audio_gap also catches playback that started and finished
        # while transcription was running.
        if (
            self.isInterruptionRequested()
            or self.speaking_event.is_set()
            or audio_gap.is_set()
        ):
            return

        print(
            f"Recognition time: "
            f"{time.monotonic() - started:.2f}s"
        )
        self.transcript_received.emit(text)

        if time.monotonic() - captured_at > MAX_RESULT_AGE:
            print(
                "Stale recognition result discarded. "
                "Please repeat."
            )
            return

        command = identify_command(text)

        if command:
            self.command_received.emit(command)