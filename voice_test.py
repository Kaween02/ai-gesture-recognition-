import math
import re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent / "models" / "whisper"
)

RECORD_SECONDS = 4
WHISPER_SAMPLE_RATE = 16000

VOICE_CONTEXT = (
    "Ultron. Hey Ultron.open. Open workspace. Close workspace.close. "
    "Create a cube. Create a sphere. Create a robot. "
    "Load engine. Show wireframe. Show solid. "
    "Reset view. Clear workspace. Choose one. Cancel."
)


def normalize_words(text):
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    )


def identify_command(text):
    request = normalize_words(text)

    # Saying only the assistant's name still opens the workspace.
    if request in {"hey ultron", "ultron", "hey ultron open", "ultron open"}:
        return "OPEN_WORKSPACE"

    # Wake phrase is optional.
    if request.startswith("hey ultron "):
        request = request[len("hey ultron "):]
    elif request.startswith("ultron "):
        request = request[len("ultron "):]
    elif request.startswith("open "):
        request = request[len("open "):]

    commands = {
        "open workspace": "OPEN_WORKSPACE",
        "open the workspace": "OPEN_WORKSPACE",
        "close workspace": "CLOSE_WORKSPACE",
        "close the workspace": "CLOSE_WORKSPACE",

        "create cube": "CREATE_CUBE",
        "create a cube": "CREATE_CUBE",
        "create sphere": "CREATE_SPHERE",
        "create a sphere": "CREATE_SPHERE",
        "create robot": "CREATE_ROBOT",
        "create a robot": "CREATE_ROBOT",

        "show wireframe": "SHOW_WIREFRAME",
        "show wire frame": "SHOW_WIREFRAME",
        "show solid": "SHOW_SOLID",

        "reset view": "RESET_VIEW",
        "reset the view": "RESET_VIEW",

        "clear workspace": "CLEAR_WORKSPACE",
        "clear the workspace": "CLEAR_WORKSPACE",

        "cancel": "CANCEL",
    }

    if request in commands:
        return commands[request]

    match = re.fullmatch(r"load (.+)", request)

    if match:
        return f"LOAD_MODEL:{match.group(1)}"

    match = re.fullmatch(
        r"choose(?: number)? (one|two|three|four|five|[1-5])",
        request,
    )

    if match:
        number = match.group(1)

        numbers = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
        }

        return f"SELECT_MODEL:{numbers.get(number, number)}"

    return None

def record_command():
    microphone = sd.query_devices(kind="input")
    sample_rate = int(microphone["default_samplerate"])

    print(f"Listening for {RECORD_SECONDS} seconds...")

    recording = sd.rec(
        frames=RECORD_SECONDS * sample_rate,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )

    status = sd.wait()

    if status:
        raise RuntimeError(f"Audio capture problem: {status}")

    audio = recording[:, 0]

    if sample_rate != WHISPER_SAMPLE_RATE:
        divisor = math.gcd(
            sample_rate,
            WHISPER_SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            WHISPER_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )

    return np.ascontiguousarray(audio, dtype=np.float32)


def main():
    print("Loading speech model...")

    model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
        download_root=str(MODEL_DIRECTORY),
    )

    print("\nSpeech model ready.")
    print('Example: "Hey Ultron, create a cube."')
    print("This test prints commands; main.py performs them.")
    print("Enter q at the prompt to quit.")

    while True:
        response = input(
            "\nPress Enter to record, or type q: "
        ).strip()

        if response.lower() == "q":
            break

        audio = record_command()
        started = time.perf_counter()

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

        elapsed = time.perf_counter() - started

        print("Heard:", text or "[no speech detected]")
        print(f"Recognition time: {elapsed:.2f} seconds")

        command = identify_command(text)

        if command:
            print("Command:", command)
        else:
            print("No supported command matched.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sd.stop()
        print("\nVoice test stopped.")
    except Exception as error:
        sd.stop()
        print(
            f"\nVoice test failed: "
            f"{type(error).__name__}: {error}"
        )
        raise