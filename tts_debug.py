import logging
from pathlib import Path

import soundfile as sf
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        from melo.api import TTS  # type: ignore[import]
    except Exception as e:
        logging.error(f"Failed to import MeloTTS (melo.api.TTS): {e}")
        return

    # For now we just test English TTS
    language = "EN_NEWEST"
    logging.info(f"Initializing MeloTTS with language={language}")
    tts = TTS(language=language, device=device)

    speaker_ids = tts.hps.data.spk2id
    logging.info(f"Available speakers: {list(speaker_ids.keys())}")
    if not speaker_ids:
        logging.error("No speakers found in MeloTTS model")
        return

    speaker_key = next(iter(speaker_ids.keys()))
    speaker_id = speaker_ids[speaker_key]
    logging.info(f"Using speaker: {speaker_key} -> id={speaker_id}")

    text = "This is a Melo TTS debug test for the teacher cascade."
    out_path = Path("tts_debug.wav")

    logging.info(f"Synthesizing text: {text!r} -> {out_path}")
    tts.tts_to_file(text, speaker_id, str(out_path), speed=1.0)
    logging.info(f"Wrote debug TTS audio to {out_path.resolve()}")


if __name__ == "__main__":
    main()
