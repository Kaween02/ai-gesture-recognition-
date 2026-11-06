import numpy as np
import sounddevice as sd


def main():
    microphone = sd.query_devices(kind="input")
    sample_rate = int(microphone["default_samplerate"])
    duration = 5

    print("Microphone:", microphone["name"])
    print("Sample rate:", sample_rate)

    input("\nPress Enter when ready to speak...")
    print('Recording for 5 seconds. Say: "Jarvis, open workspace."')

    audio = sd.rec(
        frames=duration * sample_rate,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )

    recording_status = sd.wait()
    if recording_status:
        print("Recording warning:", recording_status)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    level_db = 20 * np.log10(max(rms, 1e-10))

    print(f"\nPeak amplitude: {peak:.3f}")
    print(f"Average level: {level_db:.1f} dBFS")

    if peak < 0.001:
        print("Very little audio detected. Check your microphone.")
    elif peak >= 0.99:
        print("Possible clipping. Try speaking farther from the microphone.")

    print("\nPlaying your recording...")
    sd.play(audio, samplerate=sample_rate)
    playback_status = sd.wait()

    if playback_status:
        print("Playback warning:", playback_status)

    print("Test complete. Audio was kept in memory only.")


if __name__ == "__main__":
    try:
        main()
    except (sd.PortAudioError, ValueError) as error:
        print("Audio test failed:", error)
    except KeyboardInterrupt:
        sd.stop()
        print("\nTest cancelled.")