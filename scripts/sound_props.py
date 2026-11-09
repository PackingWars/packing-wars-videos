"""3 propositions de sons zen / ASMR pour le #04 : meme video, seul le son change. Mixage stereo (boite de gauche a
gauche, de droite a droite), gain qui s'adapte a la densite des poses, compression douce, niveau homogene."""
import sys
import os
import subprocess
import wave
import numpy as np
from scipy.signal import butter, sosfilt

sys.argv = ["proto_l2.py", "x", "1207"]
src = open("/home/claude/proto_l2.py").read().split("r = fr.Renderer(")[0]
g = {"__name__": "x"}
exec(src, g)
fr = g["fr"]
import fr_music as fm

SR = 44100
r = fr.Renderer(1207)
DUR = r.total_out / fr.FPS
N = int((DUR + 0.05) * SR)
S = len(fr.STRATS)
rng = np.random.default_rng(7)

ev = []                                                  # (t, boite, aire, progression 0..1)
for s in range(S):
    placed = [j for j, p in enumerate(r.plans[s]) if p]
    for k, j in enumerate(placed):
        p = r.plans[s][j]
        ev.append((r.tf(r.land[s, j]), s, p[2] * p[3], k / max(1, len(placed) - 1)))
ev.sort()
times = np.array([e[0] for e in ev])
dens = np.array([((times > t - 0.25) & (times < t + 0.25)).sum() for t in times]) / 0.5   # poses par seconde
T_END = r.tf(r.END)
FIN = [r.tf(f) for f in r.finish]
PENT = [0, 3, 5, 7, 10]


def scale_f(k, root):
    o, d = divmod(int(k), 5)
    return root * 2 ** ((PENT[d] + 12 * o) / 12)


def pan_gains(s):
    p = -0.55 if s % 2 == 0 else 0.55                    # colonne gauche / droite
    return np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)


class Bus:
    def __init__(self):
        self.L = np.zeros(N, np.float32)
        self.R = np.zeros(N, np.float32)

    def add(self, sig, t, gain, pan=(0.707, 0.707)):
        st = int(t * SR)
        if st < 0 or st >= N:
            return
        seg = sig[:N - st] * gain
        self.L[st:st + len(seg)] += seg * pan[0]
        self.R[st:st + len(seg)] += seg * pan[1]

    def master(self, rev_s, rev_mix, lp, target_rms=0.13):
        out = []
        for ch in (self.L, self.R):
            a = ch.copy()
            if lp:
                a = sosfilt(butter(2, lp, btype="lowpass", fs=SR, output="sos"), a)
            a = fm.reverb(a.astype(np.float32), SR, rev_s, rev_mix)
            out.append(a)
        a = np.stack(out, 1)
        a /= max(1e-9, np.abs(a).max())
        a = np.tanh(a * 1.8) / np.tanh(1.8)              # compression douce
        a *= target_rms / max(1e-9, np.sqrt((a ** 2).mean()))
        pk = np.abs(a).max()
        if pk > 0.92:
            a *= 0.92 / pk
        f = int(0.05 * SR)
        a[-f:] *= np.linspace(1, 0, f)[:, None]
        return a


_cache = {}


def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn().astype(np.float32)
    return _cache[key]


def env_t(d):
    return np.arange(int(d * SR)) / SR


def pad(bus, t0, t1, root, gain):
    t = env_t(t1 - t0)
    e = np.minimum(1, t / 1.5) * np.minimum(1, (t[-1] - t) / 1.5 + 1e-3)
    s = sum(a * np.sin(2 * np.pi * root * m * t + ph) for m, a, ph in ((1, 1.0, 0), (1.5, 0.5, 1), (2, 0.45, 2), (3, 0.12, 3)))
    s *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.2 * t)
    bus.add((s * e).astype(np.float32), t0, gain)


# ---------- A : KALIMBA (notes douces qui montent avec le remplissage) ----------
def kalimba(f):
    def mk():
        t = env_t(2.0)
        s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.6) + 0.22 * np.sin(2 * np.pi * f * 5.95 * t) * np.exp(-t * 14)
             + 0.1 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t * 6))
        return s * (1 - np.exp(-t * 500))
    return cached(("kal", round(f, 1)), mk)


def prop_A():
    b = Bus()
    base = [0, 2, 4, 1]                                  # chaque boite part d'une note differente
    for (t, s, area, prog), d in zip(ev, dens):
        k = 8 + base[s] + int(prog * 8) + (1 if area > 120 else 0) * -2
        g = 0.22 / np.sqrt(max(1.0, d / 3))
        b.add(kalimba(scale_f(k, 110.0)), t, g, pan_gains(s))
    pad(b, 0.3, T_END + 3, 55.0, 0.035)
    for s in range(S):
        for i, k in enumerate((13, 15, 17)):
            b.add(kalimba(scale_f(k, 110.0)), FIN[s] + 0.07 * i, 0.12, pan_gains(s))
    for i, k in enumerate((10, 12, 13, 15, 17, 20)):
        b.add(kalimba(scale_f(k, 110.0)), T_END + 0.09 * i, 0.2)
    return b.master(1.8, 0.32, 6000)


# ---------- B : BOIS ASMR (blocs de bois poses sur une table, aucune note) ----------
def wood(area, var):
    q = int(np.clip(np.log2(max(area, 4)) * 2, 2, 16))

    def mk():
        t = env_t(0.35)
        f = 900 * 0.86 ** q * (1 + 0.06 * (var - 1))
        body = np.sin(2 * np.pi * f * t) * np.exp(-t * 38) + 0.5 * np.sin(2 * np.pi * f * 2.31 * t) * np.exp(-t * 60)
        nz = sosfilt(butter(2, [300, 2200], btype="bandpass", fs=SR, output="sos"),
                     np.random.default_rng(q * 5 + var).standard_normal(len(t))) * np.exp(-t * 140)
        thump = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 22) * max(0, (q - 8) / 8)
        return (body * 0.9 + nz * 0.7 + thump) * (1 - np.exp(-t * 3000))
    return cached(("wood", q, var), mk)


def prop_B():
    b = Bus()
    for (t, s, area, prog), d in zip(ev, dens):
        g = 0.3 / np.sqrt(max(1.0, d / 3))
        b.add(wood(area, int(rng.integers(0, 3))), t + rng.uniform(0, 0.008), g, pan_gains(s))
    pad(b, 0.3, T_END + 3, 55.0, 0.03)
    for s in range(S):                                   # petit 'toc-toc' grave quand une boite a fini
        b.add(wood(200, 1), FIN[s], 0.25, pan_gains(s))
        b.add(wood(200, 2), FIN[s] + 0.14, 0.18, pan_gains(s))
    for i in range(14):                                  # pluie de petits blocs pour la victoire
        b.add(wood(rng.choice([8, 16, 30]), int(rng.integers(0, 3))), T_END + 0.05 + i * 0.07, 0.2 * (1 - i / 20),
              pan_gains(int(rng.integers(0, 2))))
    pad(b, T_END - 0.2, T_END + 3.5, 82.4, 0.06)
    return b.master(0.9, 0.2, 7000)


# ---------- C : GOUTTES D'EAU (plip accordes + pluie douce en fond) ----------
def drop(f):
    def mk():
        t = env_t(0.4)
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.9 * (1 - np.exp(-t * 55)))) / SR
        return np.sin(ph) * np.exp(-t * 16) * (1 - np.exp(-t * 900))
    return cached(("drop", round(f, 1)), mk)


def prop_C():
    b = Bus()
    for (t, s, area, prog), d in zip(ev, dens):
        k = 9 + int(prog * 7) + (s % 2) * 2 - (2 if area > 120 else 0)
        g = 0.2 / np.sqrt(max(1.0, d / 3))
        b.add(drop(scale_f(k, 110.0)), t, g, pan_gains(s))
    t = env_t(T_END + 3)                                 # pluie douce
    rain = sosfilt(butter(2, [800, 5000], btype="bandpass", fs=SR, output="sos"), rng.standard_normal(len(t)))
    rain *= 0.6 + 0.4 * np.convolve(np.abs(rng.standard_normal(len(t))), np.ones(4000) / 4000, mode="same")
    b.add((rain * np.minimum(1, t / 2)).astype(np.float32), 0.0, 0.012)
    pad(b, 0.3, T_END + 3, 55.0, 0.03)
    for s in range(S):
        b.add(drop(scale_f(15, 110.0)), FIN[s], 0.12, pan_gains(s))
    for i, k in enumerate((10, 12, 14, 15, 17, 19, 20)):
        b.add(drop(scale_f(k, 110.0)), T_END + 0.08 * i, 0.16)
    pad(b, T_END - 0.2, T_END + 3.5, 110.0, 0.05)
    return b.master(2.0, 0.35, 6500)


# ---------- D : ENTRE A ET C (kalimba douce + petite goutte a l'attaque, pluie tres legere) ----------
def prop_D():
    b = Bus()
    base = [0, 2, 4, 1]
    for (t, s, area, prog), d in zip(ev, dens):
        k = 8 + base[s] + int(prog * 8) - (2 if area > 120 else 0)
        g = 0.2 / np.sqrt(max(1.0, d / 3))
        f = scale_f(k, 110.0)
        b.add(kalimba(f), t, g, pan_gains(s))
        b.add(drop(f * 2), t, g * 0.45, pan_gains(s))
    t = env_t(T_END + 3)
    rain = sosfilt(butter(2, [800, 5000], btype="bandpass", fs=SR, output="sos"), rng.standard_normal(len(t)))
    rain *= 0.6 + 0.4 * np.convolve(np.abs(rng.standard_normal(len(t))), np.ones(4000) / 4000, mode="same")
    b.add((rain * np.minimum(1, t / 2)).astype(np.float32), 0.0, 0.007)
    pad(b, 0.3, T_END + 3, 55.0, 0.033)
    for s in range(S):
        for i, k in enumerate((13, 15, 17)):
            b.add(kalimba(scale_f(k, 110.0)), FIN[s] + 0.07 * i, 0.11, pan_gains(s))
            b.add(drop(scale_f(k + 5, 110.0)), FIN[s] + 0.07 * i, 0.05, pan_gains(s))
    for i, k in enumerate((10, 12, 13, 15, 17, 20)):
        b.add(kalimba(scale_f(k, 110.0)), T_END + 0.09 * i, 0.18)
        b.add(drop(scale_f(k + 5, 110.0)), T_END + 0.09 * i, 0.07)
    pad(b, T_END - 0.2, T_END + 3.5, 110.0, 0.04)
    return b.master(1.9, 0.33, 6500)


def write(a, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


video = "/home/claude/ep04_classic_zen_v2.mp4"
ONLY = os.environ.get("ONLY")
for name, fn in (("A_kalimba", prop_A), ("B_bois", prop_B), ("C_gouttes", prop_C), ("D_kalimba_gouttes", prop_D)):
    if ONLY and ONLY not in name:
        continue
    a = fn()
    wav = f"/home/claude/son_{name}.wav"
    write(a, wav)
    out = f"/home/claude/ep04_son_{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
    os.remove(wav)
    print(name, "rms", round(float(np.sqrt((a ** 2).mean())), 3), "pic", round(float(np.abs(a).max()), 2), out)
