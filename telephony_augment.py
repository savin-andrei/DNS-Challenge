"""
Telephony-style augmentation utilities.
"""
import math
import os
import random
import tempfile
import subprocess

import numpy as np
from scipy import signal
import audioop


def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.999)
    if high <= low:
        high = min(low + 1e-3, 0.999)
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def bandlimit(audio, fs, lowcut, highcut, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.lfilter(b, a, audio)


def resample_to(audio, fs_in, fs_out):
    if fs_in == fs_out:
        return audio
    gcd = math.gcd(int(fs_in), int(fs_out))
    up = int(fs_out // gcd)
    down = int(fs_in // gcd)
    return signal.resample_poly(audio, up, down)


def _float_to_int16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _int16_to_float(audio_int16):
    return audio_int16.astype(np.float32) / 32767.0


def alaw_roundtrip(audio):
    pcm = _float_to_int16(audio)
    encoded = audioop.lin2alaw(pcm.tobytes(), 2)
    decoded = audioop.alaw2lin(encoded, 2)
    return _int16_to_float(np.frombuffer(decoded, dtype=np.int16))


def mulaw_roundtrip(audio):
    pcm = _float_to_int16(audio)
    encoded = audioop.lin2ulaw(pcm.tobytes(), 2)
    decoded = audioop.ulaw2lin(encoded, 2)
    return _int16_to_float(np.frombuffer(decoded, dtype=np.int16))


def quantize(audio, bits):
    if bits >= 16:
        return audio
    levels = (1 << bits) - 1
    audio = np.clip(audio, -1.0, 1.0)
    audio = np.round((audio + 1.0) * 0.5 * levels) / levels
    return audio * 2.0 - 1.0



def apply_gain_variation(audio, fs, max_db=3.0, segment_s=1.0, rng=None):
    if max_db <= 0:
        return audio
    rng = rng or random
    num_segments = max(1, int(len(audio) / max(int(segment_s * fs), 1)))
    gains_db = [rng.uniform(-max_db, max_db) for _ in range(num_segments + 1)]
    gains = np.interp(
        np.linspace(0, num_segments, num=len(audio), endpoint=False),
        np.arange(num_segments + 1),
        10 ** (np.array(gains_db) / 20.0),
    )
    return audio * gains.astype(np.float32)


def _ffmpeg_codec_roundtrip(audio, fs, codec):
    """Best-effort codec roundtrip via ffmpeg. Returns None on failure."""
    ffmpeg = "ffmpeg"
    if not shutil_which(ffmpeg):
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        in_wav = os.path.join(tmpdir, "in.wav")
        out_wav = os.path.join(tmpdir, "out.wav")
        _write_wav(in_wav, audio, fs)
        if codec == "amr_nb":
            codec_name = "amr_nb"
            bitrate = "12.2k"
            codec_sr = 8000
        elif codec == "amr_wb":
            codec_name = "amr_wb"
            bitrate = "12.65k"
            codec_sr = 16000
        else:
            return None
        amr_path = os.path.join(tmpdir, "tmp.amr")
        cmd_enc = [ffmpeg, "-y", "-i", in_wav, "-ar", str(codec_sr), "-ac", "1",
                   "-c:a", codec_name, "-b:a", bitrate, amr_path]
        cmd_dec = [ffmpeg, "-y", "-i", amr_path, "-ar", str(fs), "-ac", "1", out_wav]
        try:
            subprocess.run(cmd_enc, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(cmd_dec, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return _read_wav(out_wav)
        except Exception:
            return None


def _write_wav(path, audio, fs):
    import soundfile as sf
    sf.write(path, audio, fs)


def _read_wav(path):
    import soundfile as sf
    audio, _ = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)


def parse_codec_mix(codec_mix_str):
    if not codec_mix_str:
        return [("amr_nb", 1.0)]
    items = []
    for part in codec_mix_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, weight = part.split(":", 1)
            items.append((name.strip(), float(weight)))
        else:
            items.append((part, 1.0))
    return items if items else [("amr_nb", 1.0)]


def choose_codec(codec_mix, rng=None):
    rng = rng or random
    total = sum(weight for _, weight in codec_mix)
    if total <= 0:
        return codec_mix[0][0]
    pick = rng.random() * total
    acc = 0.0
    for name, weight in codec_mix:
        acc += weight
        if pick <= acc:
            return name
    return codec_mix[-1][0]


def apply_telephony_augmentation(audio, fs, cfg, rng=None):
    if not cfg or not cfg.get("enable", False):
        return audio

    rng = rng or random
    codec_mix = cfg.get("codec_mix", [("amr_nb", 1.0)])
    codec = choose_codec(codec_mix, rng=rng)

    if codec in ("amr_wb",):
        band_low = cfg.get("band_low_wb", 50.0)
        band_high = cfg.get("band_high_wb", 7000.0)
        target_sr = cfg.get("wb_target_sr", fs)
        quant_bits = cfg.get("quant_bits_wb", 10)
    else:
        band_low = cfg.get("band_low_nb", 300.0)
        band_high = cfg.get("band_high_nb", 3400.0)
        target_sr = cfg.get("nb_target_sr", 8000)
        quant_bits = cfg.get("quant_bits_nb", 8)

    audio_proc = bandlimit(audio, fs, band_low, band_high)

    if target_sr and target_sr != fs:
        audio_proc = resample_to(audio_proc, fs, target_sr)
        audio_proc = resample_to(audio_proc, target_sr, fs)

    if codec == "alaw":
        audio_proc = alaw_roundtrip(audio_proc)
    elif codec == "mulaw":
        audio_proc = mulaw_roundtrip(audio_proc)
    elif codec in ("amr_nb", "amr_wb"):
        if cfg.get("codec_backend") == "ffmpeg":
            ffmpeg_audio = _ffmpeg_codec_roundtrip(audio_proc, fs, codec)
            if ffmpeg_audio is not None:
                audio_proc = ffmpeg_audio
            else:
                audio_proc = quantize(audio_proc, quant_bits)
        else:
            audio_proc = quantize(audio_proc, quant_bits)
    else:
        audio_proc = quantize(audio_proc, quant_bits)

    gain_var_db = float(cfg.get("gain_variation_db", 0.0))
    if gain_var_db > 0:
        audio_proc = apply_gain_variation(
            audio_proc,
            fs,
            max_db=gain_var_db,
            segment_s=float(cfg.get("gain_variation_segment_s", 1.0)),
            rng=rng,
        )

    return np.clip(audio_proc, -1.0, 1.0)
