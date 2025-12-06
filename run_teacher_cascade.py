#!/usr/bin/env python
"""
Offline teacher cascade runner for S2ST dataset generation.

Pipeline (per sample, RU→EN for now):

1. Load src.wav from manifest.
2. Whisper ASR → src.txt (ru_text_teacher).
3. M2M-100 (later GPT-4o) → tgt.txt (en_text_teacher).
4. ECAPA-TDNN → spk.npy (speaker embedding from src.wav).
5. OpenVoice → tgt.wav (EN speech in same voice).
6. EnCodec 24kHz 6kbps → tokens.npy (discrete speech tokens).
7. meta.json + DONE marker.

Directory structure (for direction == "ruen"):

output_root/
  data/
    parallel/
      ruen/
        <id>/
          src.wav
          src.txt
          tgt.wav
          tgt.txt
          spk.npy
          tokens.npy
          meta.json
          DONE
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm


# =========================
# Config / CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher cascade for S2ST dataset generation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (see example in spec).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["ru", "en"],
        required=True,
        help="Source language for this run (ru or en).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["ruen", "enru"],
        required=True,
        help="Translation direction (ruen or enru).",
    )
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="Shard index for parallel processing (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for models, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


# =========================
# Model wrappers (skeletons)
# =========================

class WhisperASR:
    """
    Thin wrapper around HuggingFace Whisper for offline ASR.

    Assumptions:
      - waveform is 1D float32 numpy array, mono
      - sr == 16000 (enforced by load_audio)
      - language is ISO-639-1 code: 'ru' or 'en'
    """

    def __init__(
        self,
        device: str,
        model_name: str = "openai/whisper-large-v3",
        use_fp16: bool = True,
    ):
        import transformers  # local import so module still imports if HF not installed

        self.device = torch.device(device)
        self.model_name = model_name
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        logging.info(f"[WhisperASR] Loading model {model_name} on {self.device} (fp16={self.use_fp16})")

        self.processor = transformers.WhisperProcessor.from_pretrained(model_name)
        self.model = transformers.WhisperForConditionalGeneration.from_pretrained(model_name)

        if self.use_fp16:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()

        # cache for decoder prompts
        self._forced_ids_cache = {}

    def _get_forced_decoder_ids(self, language: str, task: str = "transcribe"):
        """
        Cache forced decoder ids per (lang, task).
        language: 'ru' or 'en'
        """
        key = (language, task)
        if key in self._forced_ids_cache:
            return self._forced_ids_cache[key]

        forced_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task,
        )
        self._forced_ids_cache[key] = forced_ids
        return forced_ids

    @torch.inference_mode()
    def transcribe(self, waveform: np.ndarray, sr: int, language: str) -> str:
        """
        Run Whisper ASR on a single utterance.

        :param waveform: float32 numpy array [T], mono
        :param sr: sampling rate (expected 16000)
        :param language: 'ru' or 'en'
        :return: transcription string
        """
        if waveform.ndim != 1:
            raise ValueError(f"WhisperASR expects 1D mono audio, got shape {waveform.shape}")

        if sr != 16000:
            # In theory processor can resample, but your loader enforces 16k already.
            raise ValueError(f"WhisperASR expected sr=16000, got {sr}")

        # HF processor expects list/array and will do normalization
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device)
        if self.use_fp16:
            input_features = input_features.half()

        forced_ids = self._get_forced_decoder_ids(language=language, task="transcribe")

        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_ids,
            # you can tweak decoding settings here if needed:
            # max_new_tokens=128,
            # temperature=0.0,
        )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        return text

    def __call__(self, waveform: np.ndarray, sr: int, language: str) -> str:
        """Convenience alias."""
        return self.transcribe(waveform, sr, language)



class M2MTranslator:
    def __init__(self, device: str):
        self.device = torch.device(device)
        # TODO: uncomment & adjust once transformers installed
        #
        # from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        # self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
        # self.model = M2M100ForConditionalGeneration.from_pretrained(
        #     "facebook/m2m100_1.2B"
        # ).to(self.device)
        # self.model.eval()  

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        :param text: input text in src_lang
        :param src_lang: 'ru' or 'en'
        :param tgt_lang: 'en' or 'ru'
        """
        # TODO: implement actual M2M translation.
        # Example:
        #
        # self.tokenizer.src_lang = src_lang
        # encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        # generated = self.model.generate(
        #     **encoded,
        #     forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
        # )
        # out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        # return out
        raise NotImplementedError("M2MTranslator.translate not implemented")


class ECAPASpeakerEncoder:
    def __init__(self, device: str):
        self.device = torch.device(device)
        # TODO: uncomment once speechbrain installed
        #
        # from speechbrain.pretrained import SpeakerRecognition
        # self.rec = SpeakerRecognition.from_hparams(
        #     source="speechbrain/spkrec-ecapa-voxceleb",
        #     savedir="pretrained_models/spkrec-ecapa-voxceleb",
        # )
        # self.rec.to(self.device)

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        :param waveform: float32 numpy array [T]
        :param sr: sampling rate (expected 16k)
        :return: speaker embedding np.ndarray [D]
        """
        # TODO: implement ECAPA embedding.
        # Example:
        #
        # import torch
        # wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     emb = self.rec.encode_batch(wav_tensor).squeeze(0).squeeze(0)
        # return emb.cpu().numpy()
        raise NotImplementedError("ECAPASpeakerEncoder.embed not implemented")


class OpenVoiceSynth:
    def __init__(self, device: str):
        self.device = torch.device(device)
        # TODO: initialize OpenVoice model according to its repo instructions.
        # This will be project-specific.

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray,
        tgt_lang: str,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        :param text: target language text
        :param speaker_embedding: np.ndarray [D]
        :param tgt_lang: 'en' or 'ru'
        :param sample_rate: desired output sample rate
        :return: waveform float32 [T] at sample_rate
        """
        # TODO: implement OpenVoice synthesis (TTS/VC).
        raise NotImplementedError("OpenVoiceSynth.synthesize not implemented")


class EncodecWrapper:
    def __init__(self, device: str):
        self.device = torch.device(device)
        # TODO: uncomment once encodec installed
        #
        # from encodec import EncodecModel
        # from encodec.utils import convert_audio
        # self.model = EncodecModel.encodec_model_24khz()
        # self.model.set_target_bandwidth(6.0)  # kbps
        # self.model.to(self.device)
        # self.model.eval()
        # self.convert_audio = convert_audio

    def encode(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        :param waveform: float32 numpy array [T], mono
        :param sr: sample rate of waveform (e.g. 24000)
        :return: codes np.ndarray [num_codebooks, num_frames]
        """
        # TODO: implement EnCodec encoding.
        # Example:
        #
        # wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        # wav_24k = self.convert_audio(wav_tensor, sr, 24000, 1)
        # with torch.no_grad():
        #     encoded_frames = self.model.encode(wav_24k)
        # codes = torch.cat([f.codes for f in encoded_frames], dim=-1)  # [B, Q, T]
        # return codes.squeeze(0).cpu().numpy()
        raise NotImplementedError("EncodecWrapper.encode not implemented")


# =========================
# Utility helpers
# =========================

def load_audio(path: Path, target_sr: int = 16000) -> (np.ndarray, int):
    """Load mono audio as float32 numpy array [T]."""
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if sr != target_sr:
        # TODO: use librosa or torchaudio resample
        # e.g. librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        raise NotImplementedError("Resampling not implemented yet")

    return wav, target_sr


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_done_dir(out_dir: Path) -> bool:
    return (out_dir / "DONE").exists()


def mark_done(out_dir: Path):
    (out_dir / "DONE").touch()


# =========================
# Manifest loading / sharding
# =========================

def load_manifest(manifest_path: str, lang: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    logging.info(f"Loading manifest from {manifest_path}")
    df = pd.read_csv(manifest_path)
    df = df[df["lang"] == lang]

    min_dur = cfg["pipeline"].get("min_duration", 0.0)
    max_dur = cfg["pipeline"].get("max_duration", 1e9)
    if "duration_s" in df.columns:
        df = df[(df["duration_s"] >= min_dur) & (df["duration_s"] <= max_dur)]

    # Optional: use tier filtering for Golos
    tier_col = "tier"
    if tier_col in df.columns and "tier" in cfg["pipeline"]:
        allowed = cfg["pipeline"]["tier"]
        if isinstance(allowed, str):
            allowed = [allowed]
        df = df[df[tier_col].isin(allowed)]

    df = df.reset_index(drop=True)
    logging.info(f"Loaded {len(df)} rows after filtering.")
    return df


def shard_dataframe(df: pd.DataFrame, shard_idx: int, num_shards: int) -> pd.DataFrame:
    if num_shards <= 1:
        return df
    # simple round-robin sharding
    idxs = list(range(len(df)))
    shard_idxs = idxs[shard_idx::num_shards]
    shard_df = df.iloc[shard_idxs].reset_index(drop=True)
    logging.info(
        f"Sharding: total={len(df)}, num_shards={num_shards}, shard_idx={shard_idx}, shard_size={len(shard_df)}"
    )
    return shard_df


# =========================
# Core per-sample processing
# =========================

def process_sample(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
    models: Dict[str, Any],
    direction: str,
    data_root: Path,
    output_root: Path,
) -> None:
    """
    Process one row from manifest into a parallel/<direction>/<id>/ directory.
    """
    src_lang, tgt_lang = ("ru", "en") if direction == "ruen" else ("en", "ru")

    sample_id = str(row["id"])
    # Make filesystem-safe
    sample_id = sample_id.replace("/", "_").replace("\\", "_").replace(":", "_")

    out_dir = output_root / "data" / "parallel" / direction / sample_id
    if is_done_dir(out_dir):
        return

    ensure_dir(out_dir)

    audio_rel = row["audio_path"]
    audio_path = data_root / Path(audio_rel)
    audio_path = audio_path.resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 0) Load audio
    target_sr = 16000
    wav_src, sr_src = load_audio(audio_path, target_sr=target_sr)

    # 1) Whisper ASR
    whisper: WhisperASR = models["whisper"]
    src_text = whisper.transcribe(wav_src, sr_src, language=src_lang)

    # 2) Translation
    translator: M2MTranslator = models["translator"]
    tgt_text = translator.translate(src_text, src_lang=src_lang, tgt_lang=tgt_lang)

    # 3) Speaker embedding from src.wav
    spk_encoder: ECAPASpeakerEncoder = models["speaker"]
    spk_emb = spk_encoder.embed(wav_src, sr_src)

    # 4) OpenVoice synthesis in target language
    openvoice: OpenVoiceSynth = models["openvoice"]
    wav_tgt = openvoice.synthesize(
        text=tgt_text,
        speaker_embedding=spk_emb,
        tgt_lang=tgt_lang,
        sample_rate=24000,  # or whatever OpenVoice expects
    )
    sr_tgt = 24000

    # 5) EnCodec tokens from tgt.wav
    encodec: EncodecWrapper = models["encodec"]
    tokens = encodec.encode(wav_tgt, sr_tgt)

    # ---- Save outputs ----
    # src.wav (you can either copy original or re-save normalized)
    src_wav_out = out_dir / "src.wav"
    sf.write(str(src_wav_out), wav_src, sr_src)

    # src.txt
    with open(out_dir / "src.txt", "w", encoding="utf-8") as f:
        f.write(src_text + "\n")

    # tgt.wav
    tgt_wav_out = out_dir / "tgt.wav"
    sf.write(str(tgt_wav_out), wav_tgt, sr_tgt)

    # tgt.txt
    with open(out_dir / "tgt.txt", "w", encoding="utf-8") as f:
        f.write(tgt_text + "\n")

    # spk.npy
    np.save(out_dir / "spk.npy", spk_emb)

    # tokens.npy
    np.save(out_dir / "tokens.npy", tokens)

    # meta.json
    meta = {
        "id": sample_id,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "src_duration_s": float(row.get("duration_s", len(wav_src) / sr_src)),
        "tgt_duration_s": float(len(wav_tgt) / sr_tgt),
        "src_text_len": len(src_text),
        "tgt_text_len": len(tgt_text),
        "status": "ok",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # DONE marker
    mark_done(out_dir)


# =========================
# Model initialization
# =========================

def init_models(device: str) -> Dict[str, Any]:
    """
    Initialize and return all teacher models as a dict.
    """
    models = {
        "whisper": WhisperASR(device=device, model_name="openai/whisper-large-v3"),
        "translator": M2MTranslator(device=device),
        "speaker": ECAPASpeakerEncoder(device=device),
        "openvoice": OpenVoiceSynth(device=device),
        "encodec": EncodecWrapper(device=device),
    }
    return models


# =========================
# Main
# =========================

def main():
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    torch_device = args.device
    logging.info(f"Using device: {torch_device}")

    data_root = Path(cfg["data_root"]).expanduser().resolve()
    output_root = Path(cfg["output_root"]).expanduser().resolve()
    ensure_dir(output_root)

    # Select manifest based on lang
    manifest_path = cfg["manifests"][args.lang]
    df = load_manifest(manifest_path, lang=args.lang, cfg=cfg)
    df_shard = shard_dataframe(df, shard_idx=args.shard_idx, num_shards=args.num_shards)

    models = init_models(torch_device)

    errors_path = output_root / f"errors_{args.lang}_{args.direction}_shard{args.shard_idx}.log"
    num_processed = 0
    num_failed = 0

    with open(errors_path, "a", encoding="utf-8") as err_f:
        for _, row in tqdm(df_shard.iterrows(), total=len(df_shard), desc="Processing"):
            row_dict = row.to_dict()
            try:
                process_sample(
                    row=row_dict,
                    cfg=cfg,
                    models=models,
                    direction=args.direction,
                    data_root=data_root,
                    output_root=output_root,
                )
                num_processed += 1
            except Exception as e:
                num_failed += 1
                logging.exception(f"Error processing id={row_dict.get('id')}: {e}")
                err_f.write(f"{row_dict.get('id')}|{repr(e)}\n")
                err_f.flush()

    logging.info(
        f"Done. Processed={num_processed}, Failed={num_failed}, Shard={args.shard_idx}/{args.num_shards}"
    )

def debug_test_whisper(cfg: Dict[str, Any], args: argparse.Namespace):
    """
    Quick one-off Whisper test on a single sample from the manifest.
    """
    data_root = Path(cfg["data_root"]).expanduser().resolve()

    # 1) Load manifest for requested lang
    manifest_path = cfg["manifests"][args.lang]
    df = load_manifest(manifest_path, lang=args.lang, cfg=cfg)

    if len(df) == 0:
        logging.error("Manifest is empty after filtering; cannot test Whisper.")
        return

    # Just take the first row (or pick any index you want)
    row = df.iloc[0].to_dict()
    audio_rel = row["audio_path"]
    audio_path = (data_root / Path(audio_rel)).resolve()

    if not audio_path.exists():
        logging.error(f"Test audio file does not exist: {audio_path}")
        return

    logging.info(f"[Whisper DEBUG] Testing on {audio_path}")

    # 2) Load audio
    wav, sr = load_audio(audio_path, target_sr=16000)

    # 3) Init Whisper
    whisper = WhisperASR(device=args.device, model_name="openai/whisper-large-v3")

    # 4) Run ASR
    lang_code = args.lang  # 'ru' or 'en'
    text = whisper.transcribe(wav, sr, language=lang_code)

    logging.info(f"[Whisper DEBUG] Transcription ({lang_code}): {text}")
    print("==== WHISPER DEBUG RESULT ====")
    print(f"File: {audio_path}")
    print(f"Lang: {lang_code}")
    print("Text:", text)


if __name__ == "__main__":
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    # --- DEBUG: test Whisper only ---
    debug_test_whisper(cfg, args)
    # exit after debug
    raise SystemExit(0)

    # main()
