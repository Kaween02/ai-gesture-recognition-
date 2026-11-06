import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
import sherpa_onnx
import sounddevice as sd
from scipy.signal import resample_poly


MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
)

WAKE_PHRASE = "ULTRON"
SAMPLE_RATE = 16000

# Starting values; we'll tune these from actual detection results.
KEYWORDS_THRESHOLD = 0.25
KEYWORDS_SCORE = 1.0


def model_file(name):
    path = MODEL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    return str(path)


def create_keyword_file():
    tokenizer = spm.SentencePieceProcessor(
        model_file=model_file("bpe.model")
    )

    pieces = tokenizer.encode(WAKE_PHRASE, out_type=str)

    if not pieces or "<unk>" in pieces:
        raise ValueError("Wake phrase contains unsupported tokens.")

    keywords_path = MODEL_DIR / "ultron_keywords.txt"
    keywords_path.write_text(
        " ".join(pieces) + "\n",
        encoding="utf-8",
    )

    print("Wake phrase:", WAKE_PHRASE)
    print("Tokens:", " ".join(pieces))
    return str(keywords_path)


def main():
    spotter = sherpa_onnx.KeywordSpotter(
        tokens=model_file("tokens.txt"),
        encoder=model_file(
            "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        ),
        decoder=model_file(
            "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        ),
        joiner=model_file(
            "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        ),
        keywords_file=create_keyword_file(),
        num_threads=1,
        provider="cpu",
        keywords_score=KEYWORDS_SCORE,
        keywords_threshold=KEYWORDS_THRESHOLD,
        num_trailing_blanks=1,
    )

    microphone = sd.query_devices(kind="input")
    device_rate = int(microphone["default_samplerate"])
    samples_per_read = int(device_rate * 0.1)

    divisor = math.gcd(device_rate, SAMPLE_RATE)
    up = SAMPLE_RATE // divisor
    down = device_rate // divisor

    keyword_stream = spotter.create_stream()
    detections = 0

    print("Microphone:", microphone["name"])
    print(f'\nListening for "{WAKE_PHRASE}". Ctrl+C stops the test.')

    with sd.InputStream(
        samplerate=device_rate,
        channels=1,
        dtype="float32",
    ) as microphone_stream:
        while True:
            samples, overflowed = microphone_stream.read(
                samples_per_read
            )

            if overflowed:
                print("Audio overflow: resetting detector.")
                keyword_stream = spotter.create_stream()
                continue

            audio = samples[:, 0]

            if device_rate != SAMPLE_RATE:
                audio = resample_poly(audio, up, down)

            audio = np.ascontiguousarray(audio, dtype=np.float32)
            keyword_stream.accept_waveform(SAMPLE_RATE, audio)

            while spotter.is_ready(keyword_stream):
                spotter.decode_stream(keyword_stream)
                result = spotter.get_result(keyword_stream)

                if result:
                    detections += 1
                    print(
                        f"WAKE DETECTED #{detections}: {result}",
                        flush=True,
                    )
                    spotter.reset_stream(keyword_stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nWake-word test stopped.")