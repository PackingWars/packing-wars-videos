"""Briques sonores musicales (gamme pentatonique, reverb, nappe) pour Packing Wars."""
import numpy as np

PENT = [0, 3, 5, 7, 10]                   # La mineur pentatonique : A C D E G
ROOT = 110.0                              # La2


def ladder_freq(k, root=ROOT):
    octv, deg = divmod(int(k), 5)
    return root * 2 ** ((PENT[deg] + 12 * octv) / 12)


def pluck(f, sr, dur=1.1, bright=1.0):
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-t * 5.5) * (1 - np.exp(-t * 900))
    s = (np.sin(2 * np.pi * f * t) + 0.35 * bright * np.sin(4 * np.pi * f * t) * np.exp(-t * 8)
         + 0.15 * bright * np.sin(6 * np.pi * f * t) * np.exp(-t * 12))
    return (s * env).astype(np.float32)


def bell(f, sr, dur=1.8):
    t = np.arange(int(dur * sr)) / sr
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d)
            for r, a, d in ((1, 1.0, 2.2), (2.76, 0.45, 3.5), (5.4, 0.2, 6.0)))
    return (s * (1 - np.exp(-t * 700))).astype(np.float32)


def area_to_k(area, kmax=13, amin=6.0, amax=260.0):
    """Grosse piece -> note grave, petite piece -> note aigue."""
    x = 1 - np.log(max(area, amin) / amin) / np.log(amax / amin)
    return int(np.clip(round(x * kmax), 0, kmax))


def pad(sr, dur, times, levels):
    """Nappe douce sur un accord de La mineur, volume suivi par (times, levels)."""
    t = np.arange(int(dur * sr)) / sr
    s = sum(a * np.sin(2 * np.pi * f * t + ph)
            for f, a, ph in ((110.0, 1.0, 0.0), (164.81, 0.6, 1.0), (220.0, 0.5, 2.0), (261.63, 0.35, 3.0)))
    s = s * (0.82 + 0.18 * np.sin(2 * np.pi * 0.23 * t))
    return (s * np.interp(t, times, levels)).astype(np.float32)


def reverb(a, sr, seconds=1.5, mix=0.30, seed=11):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t * 3.0)
    k = int(0.012 * sr)
    ir[:k] *= np.linspace(0, 1, k)
    ir = np.convolve(ir, np.ones(6) / 6, mode="same")
    N = 1 << int(np.ceil(np.log2(len(a) + n)))
    wet = np.fft.irfft(np.fft.rfft(a, N) * np.fft.rfft(ir, N), N)[:len(a)]
    wet *= float(np.abs(a).max()) / max(1e-9, float(np.abs(wet).max()))
    return ((1 - mix) * a + mix * wet).astype(np.float32)


def midi_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)
