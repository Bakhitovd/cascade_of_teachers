import os
import argparse
import subprocess

def convert_m4a_to_wav(input_path, output_path=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"Converted: {input_path} -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .m4a to .wav using ffmpeg")
    parser.add_argument("input_path", help="Path to .m4a file")
    parser.add_argument("--output", help="Optional output .wav path")

    args = parser.parse_args()
    convert_m4a_to_wav(args.input_path, args.output)
