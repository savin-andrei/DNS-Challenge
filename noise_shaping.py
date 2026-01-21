# -*- coding: utf-8 -*-
"""
Time-varying noise shaping for more realistic temporal/frequency heterogeneity.
"""
import math
import random

import numpy as np
from scipy import signal

EPS = np.finfo(float).eps


def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.999)
    if high <= low:
        high = min(low + 1e-3, 0.999)
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def _butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    cut = min(max(cutoff / nyq, 1e-5), 0.999)
    b, a = signal.butter(order, cut, btype="low")
    return b, a


def _butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    cut = min(max(cutoff / nyq, 1e-5), 0.999)
    b, a = signal.butter(order, cut, btype="high")
    return b, a


def _filter_segment(seg, fs, ftype, params):
    if ftype == "lowpass":
        b, a = _butter_lowpass(params["cutoff"], fs)
    elif ftype == "highpass":
        b, a = _butter_highpass(params["cutoff"], fs)
    else:
        b, a = _butter_bandpass(params["low"], params["high"], fs)
    return signal.lfilter(b, a, seg)


def _linear_envelope(length, start_gain, end_gain):
    if length <= 1:
        return np.array([end_gain], dtype=np.float32)
    return np.linspace(start_gain, end_gain, num=length, endpoint=False, dtype=np.float32)


def apply_noise_shaping(noise, fs, cfg, rng=None):
    if not cfg or not cfg.get("enable", False):
        return noise
    if noise is None or len(noise) == 0:
        return noise

    rng = rng or random
    seg_min_s = float(cfg.get("seg_min_s", 0.5))
    seg_max_s = float(cfg.get("seg_max_s", 2.0))
    seg_min = max(1, int(seg_min_s * fs))
    seg_max = max(seg_min, int(seg_max_s * fs))
    fade_ms = float(cfg.get("fade_ms", 100.0))
    fade_len = max(0, int(fade_ms * fs / 1000.0))

    gain_db_min = float(cfg.get("gain_db_min", -25.0))
    gain_db_max = float(cfg.get("gain_db_max", -5.0))

    probs = cfg.get("filter_probs", {"bandpass": 0.5, "lowpass": 0.25, "highpass": 0.25})
    ftypes = list(probs.keys())
    weights = [probs[k] for k in ftypes]

    band_low_min = float(cfg.get("band_low_min", 200.0))
    band_low_max = float(cfg.get("band_low_max", 800.0))
    band_high_min = float(cfg.get("band_high_min", 2000.0))
    band_high_max = float(cfg.get("band_high_max", 6000.0))
    lp_min = float(cfg.get("lp_min", 1500.0))
    lp_max = float(cfg.get("lp_max", 6000.0))
    hp_min = float(cfg.get("hp_min", 100.0))
    hp_max = float(cfg.get("hp_max", 1000.0))

    out = np.zeros_like(noise, dtype=np.float32)
    weight_sum = np.zeros_like(noise, dtype=np.float32)

    pos = 0
    prev_gain = 10 ** (rng.uniform(gain_db_min, gain_db_max) / 20.0)
    while pos < len(noise):
        seg_len = rng.randint(seg_min, seg_max)
        end = min(pos + seg_len, len(noise))
        seg = noise[pos:end]

        ftype = rng.choices(ftypes, weights=weights, k=1)[0]
        if ftype == "lowpass":
            params = {"cutoff": rng.uniform(lp_min, lp_max)}
        elif ftype == "highpass":
            params = {"cutoff": rng.uniform(hp_min, hp_max)}
        else:
            low = rng.uniform(band_low_min, band_low_max)
            high = rng.uniform(max(low + 200.0, band_high_min), band_high_max)
            params = {"low": low, "high": high}

        seg_filt = _filter_segment(seg, fs, ftype, params)

        gain = 10 ** (rng.uniform(gain_db_min, gain_db_max) / 20.0)
        env = _linear_envelope(len(seg_filt), prev_gain, gain)
        seg_filt = seg_filt * env
        prev_gain = gain

        if fade_len > 0:
            fade = min(fade_len, len(seg_filt) // 2)
        else:
            fade = 0
        window = np.ones(len(seg_filt), dtype=np.float32)
        if fade > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, fade, endpoint=False))
            window[:fade] = ramp
            window[-fade:] = ramp[::-1]

        out[pos:end] += seg_filt * window
        weight_sum[pos:end] += window

        pos = end

    weight_sum = np.maximum(weight_sum, EPS)
    return out / weight_sum
