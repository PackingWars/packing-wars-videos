"""Episode #07 : guerre a 4 sans MIDI, nouveau son zen/ASMR 'piano feutre' (felt piano facon lo-fi) en stereo,
avec un souffle tres doux en fond. Vitesses differentes, classement en direct, couronne, badges d'arrivee."""
import sys
import os
import subprocess
import wave
import numpy as np
from scipy.signal import butter, sosfilt
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import fr_music as fm

out = sys.argv[1] if len(sys.argv) > 1 else "ep07_felt.mp4"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SR = 44100
PENT = [0, 2, 4, 7, 9]                                   # pentatonique majeure (plus lumineuse, tres zen)
ROOT_MIDI = 50                                           # re
_c = {}


def felt(midi):
    """Piano feutre : attaque douce de marteau en feutre, harmoniques qui s'eteignent vite, son rond et chaud."""
    if midi not in _c:
        f = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.arange(int(2.6 * SR)) / SR
        s = np.zeros_like(t)
        for h, a, d in ((1, 1.0, 1.6), (2, 0.42, 3.2), (3, 0.16, 5.5), (4, 0.07, 8.0)):
            s += a * np.sin(2 * np.pi * f * h * 1.0007 ** h * t) * np.exp(-t * d)
        s += 0.5 * np.sin(2 * np.pi * f * 1.003 * t) * np.exp(-t * 1.9)        # leger desaccord : chaleur
        ham = sosfilt(butter(2, 900, btype="lowpass", fs=SR, output="sos"),
                      np.random.default_rng(midi).standard_normal(len(t))) * np.exp(-t * 160) * 0.35
        s = (s + ham) * (1 - np.exp(-t * 180))
        _c[midi] = sosfilt(butter(2, 2600, btype="lowpass", fs=SR, output="sos"), s).astype(np.float32)
    return _c[midi]


def note(k):
    o, d = divmod(int(k), 5)
    return ROOT_MIDI + 12 + PENT[d] + 12 * o


def synth_felt(r):
    dur = r.total_out / fr.FPS
    N = int((dur + 0.05) * SR)
    L, R = np.zeros(N, np.float32), np.zeros(N, np.float32)

    def add(sig, t, g, s=None):
        st = int(t * SR)
        if st < 0 or st >= N:
            return
        seg = sig[:N - st] * g
        pl, pr = (0.707, 0.707) if s is None else ((0.88, 0.47) if s % 2 == 0 else (0.47, 0.88))
        L[st:st + len(seg)] += seg * pl
        R[st:st + len(seg)] += seg * pr
    S = len(fr.STRATS)
    evs = []
    for s in range(S):
        placed = [j for j, p in enumerate(r.plans[s]) if p]
        for k, j in enumerate(placed):
            evs.append((r.tf(r.land[s, j]), s, k / max(1, len(placed) - 1), r.plans[s][j]))
    evs.sort(key=lambda e: e[0])
    tt = np.array([e[0] for e in evs])
    base = [0, 2, 1, 3]
    for t, s, prog, p in evs:                             # chaque bloc pose = une note, qui monte avec le remplissage
        d = ((tt > t - 0.25) & (tt < t + 0.25)).sum() / 0.5
        k = 5 + base[s] + int(prog * 8) - (2 if p[2] * p[3] > 150 else 0)
        add(felt(note(k)), t, 0.2 / np.sqrt(max(1.0, d / 3)), s)
    te = r.tf(r.END)
    t = np.arange(int((te + 3) * SR)) / SR                # souffle tres doux (bruit brun filtre) + nappe
    air = sosfilt(butter(1, 400, btype="lowpass", fs=SR, output="sos"), np.random.default_rng(3).standard_normal(len(t)))
    add((air / np.abs(air).max() * np.minimum(1, t / 2)).astype(np.float32), 0.0, 0.05)
    f0 = 440.0 * 2 ** ((ROOT_MIDI - 12 - 69) / 12)
    pad = sum(a * np.sin(2 * np.pi * f0 * m * t + ph) for m, a, ph in ((1, 1.0, 0), (1.5, 0.45, 1), (2, 0.4, 2), (2.5, 0.12, 3)))
    add((pad * np.minimum(1, t / 2) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.15 * t))).astype(np.float32), 0.3, 0.028)
    for s in range(S):
        for i, k in enumerate((10, 12, 14)):
            add(felt(note(k)), r.tf(r.finish[s]) + 0.08 * i, 0.12, s)
    for i, k in enumerate((0, 2, 4, 5, 7, 9, 10)):        # accord final arpege
        add(felt(note(k)), te + 0.1 * i, 0.2)
    chans = []
    for ch in (L, R):
        chans.append(fm.reverb(ch, SR, 2.2, 0.36))
    a = np.stack(chans, 1)
    a /= max(1e-9, np.abs(a).max())
    a = np.tanh(a * 1.6) / np.tanh(1.6)
    a *= 0.13 / max(1e-9, np.sqrt((a ** 2).mean()))
    a *= min(1.0, 0.92 / max(1e-9, np.abs(a).max()))
    f = int(0.05 * SR)
    a[-f:] *= np.linspace(1, 0, f)[:, None]
    return a


c = dict(lot.CONCEPTS[0])
lot.configure(c, 7, SEED)
colors = [s_[1] for s_ in fr.STRATS]
fr.STRATS[:] = [("SPIRAL", colors[0], "spiral", False), ("EXPLOSION", colors[1], "explosion", False),
                ("COLUMNS", colors[2], "columns", False), ("BIGGEST FIRST", colors[3], "bl", True)]
fr.PSIZE = (3, 14, 0.5)
if "--scan" in sys.argv:
    for sd in range(1500, 1650):
        pieces, plans = fr.simulate(sd)
        fin = np.array([sum(q[2] * q[3] for q in pl if q) / fr.CAP * 100 for pl in plans])
        o = np.sort(fin)[::-1]
        w = fr.STRATS[int(np.argmax(fin))][0]
        if o[0] - o[1] < 0.8 and w != "BIGGEST FIRST":
            print(sd, w, len(pieces), np.round(fin, 1).tolist(), round(o[0] - o[1], 2))
    sys.exit()
fr.SUB = "PACKING WARS #07 - Classic Duel"
fr.MELODY = fr.MUSICAL = fr.BRICKS = False
fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.LIVE_RANK = True
fr.SLOW_LAST = 4
fr.SPEEDS = [1.0, 0.8, 1.5, 1.2]                         # COLUMNS le plus rapide, EXPLOSION le plus lent
fr.synth = lambda r, sr=SR: (np.zeros(int((r.total_out / fr.FPS + 0.5) * sr), np.int16), sr)
r = fr.Renderer(SEED)
print("graine", SEED, "gagnant", fr.STRATS[r.winner][0], [round(x, 1) for x in r.final],
      "arrivees", {fr.STRATS[k][0]: v for k, v in r.rank.items()}, "duree", round(r.total_out / fr.FPS, 1), flush=True)
tmp = out + ".noaudio.mp4"
fb.render(r, tmp)
a = synth_felt(r)
wav = out + ".wav"
with wave.open(wav, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
os.remove(tmp)
os.remove(wav)
print("ok", out)
