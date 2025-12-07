import argparse
import soundfile as sf
import os

def cut_wav(input_path, start_sec, end_sec, output_path=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data, sr = sf.read(input_path)

    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)

    if start_idx < 0 or end_idx > len(data) or start_idx >= end_idx:
        raise ValueError("Invalid start or end times")

    cut_audio = data[start_idx:end_idx]

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_cut_{start_sec}-{end_sec}.wav"

    sf.write(output_path, cut_audio, sr)
    print(f"Saved cut file to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut a segment from a WAV file")
    parser.add_argument("input_path", help="Path to input WAV")
    parser.add_argument("start", type=float, help="Start time (seconds)")
    parser.add_argument("end", type=float, help="End time (seconds)")
    parser.add_argument("--output", help="Optional output path")

    args = parser.parse_args()
    cut_wav(args.input_path, args.start, args.end, args.output)
