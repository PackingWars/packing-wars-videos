"""Episode #05 v2 : 4 algos classiques a vitesses differentes, refrain d'Unravel ou chaque bloc pose = une note (gros blocs,
tempo d'origine puis acceleration, fin sur une phrase), classement en direct + couronne, badges d'arrivee,
fin 'les perdants s'ecroulent', son kalimba stereo."""
import sys
import os
import subprocess
import wave
import numpy as np
from scipy.signal import butter, sosfilt
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import fr_midi
import fr_music as fm

out = sys.argv[1] if len(sys.argv) > 1 else "ep07_evangelion.mp4"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
SPLIT_SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 32.0
SLOW_SEC = 8.0
fr_midi.PATH = "/home/claude/eva.mid"
START = 0
ROOT = 65.41                                             # do mineur
SR = 44100


def split_schedule(r):
    S, n = len(fr.STRATS), r.n
    ev = []
    for s in range(S):
        ks = [j for j in range(n) if r.plans[s][j]]
        for k, j in enumerate(ks):
            ev.append(((k + 1) / fr.SPEEDS[s], s, j))
    ev.sort()
    E = len(ev)
    div, tempos, tracks = fr_midi.parse(fr_midi.PATH)
    f = fr_midi.tick_to_sec(tempos, div)
    allnotes = [x for t in tracks for x in t["notes"]]
    mel = [(f(t), p) for t, d, p, v in fr_midi.monophonic([x for x in allnotes if x[2] >= 60])]
    bass = [(f(t), p) for t, d, p, v in allnotes if p < 55]
    seq = [(t - mel[0][0], p) for t, p in mel]            # depuis le tout debut du morceau
    bseq = [(t - mel[0][0], p) for t, p in bass]
    t0s = 0.0
    T = [t for t, p in seq]
    P = [p for t, p in seq]
    ioi = list(np.diff(T)) + [2.0]
    ends = [i for i in range(len(seq)) if ioi[i] >= 0.9 and i + 1 >= E]
    e = ends[0] if ends else E - 1
    rep = [i for i in range(1, e) if P[i] == P[i - 1] and T[i] - T[i - 1] <= 0.35]
    dn = e + 1 - E
    pick = set(rep) if dn >= len(rep) else set(rep[int(round(k))] for k in np.linspace(0, len(rep) - 1, dn)) if dn > 0 else set()
    idx = [i for i in range(e + 1) if i not in pick][:E]
    seq = seq[:e + 1]
    r.cadence = (len(seq) - 1, len(seq) - len(idx))
    times = np.array([T[i] for i in idx])
    times = times - times[0]
    notes = [P[i] for i in idx]
    O = times[-1]
    p1 = min(0.9, SLOW_SEC / O)

    def warp(rmax):
        pp = times / O
        rate = np.where(pp < p1, 1.0, 1 + (rmax - 1) * (pp - p1) / (1 - p1))
        return np.cumsum(np.diff(times, prepend=0.0) / rate)
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if warp(mid)[-1] > SPLIT_SEC else (lo, mid)
    land = 14 + warp(hi) * fr.FPS
    r.rmax = hi
    st = np.full((S, n), np.nan)
    note_of = {}
    for (v, s, j), lf, p in zip(ev, land, notes):
        st[s, j] = lf - fr.MELODY_DROP
        note_of[(s, j)] = p
    for s in range(S):
        row = st[s]
        last = row[~np.isnan(row)][0]
        for j in range(n):
            if np.isnan(row[j]):
                row[j] = last
            else:
                last = row[j]
    r.split_lh = [(float(np.interp(t - t0s, times, land)), p) for t, p in bseq if 0 <= t - t0s <= O]
    return st, note_of


# ------------------------------- son : kalimba stereo -------------------------------
_c = {}


def kalimba(f):
    k = round(f, 1)
    if k not in _c:
        t = np.arange(int(2.0 * SR)) / SR
        s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.6) + 0.22 * np.sin(2 * np.pi * f * 5.95 * t) * np.exp(-t * 14)
             + 0.1 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t * 6))
        _c[k] = (s * (1 - np.exp(-t * 500))).astype(np.float32)
    return _c[k]


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def synth_kalimba(r, sr=SR):
    dur = r.total_out / fr.FPS
    N = int((dur + 0.05) * SR)
    L, R = np.zeros(N, np.float32), np.zeros(N, np.float32)

    def add(sig, t, g, s=None):
        st = int(t * SR)
        if st < 0 or st >= N:
            return
        seg = sig[:N - st] * g
        pl, pr = (0.707, 0.707) if s is None else ((0.89, 0.45) if s % 2 == 0 else (0.45, 0.89))
        L[st:st + len(seg)] += seg * pl
        R[st:st + len(seg)] += seg * pr
    evs = sorted((r.tf(r.land[s, j]), s, m) for (s, j), m in r.note_of.items())
    tt = np.array([e[0] for e in evs])
    for t, s, m in evs:
        d = ((tt > t - 0.25) & (tt < t + 0.25)).sum() / 0.5
        add(kalimba(mf(m + 12 if m < 74 else m)), t, 0.22 / np.sqrt(max(1.0, d / 3)), s)
    for f, m in r.split_lh:
        add(kalimba(mf(m + 12)), r.tf(f), 0.07)
    te = r.tf(r.END)
    t = np.arange(int((te + 3) * SR)) / SR                # nappe chaude
    pad = sum(a * np.sin(2 * np.pi * ROOT / 2 * k * t + ph) for k, a, ph in ((1, 1.0, 0), (1.5, 0.5, 1), (2, 0.45, 2)))
    add((pad * np.minimum(1, t / 1.5) * 0.85).astype(np.float32), 0.3, 0.03)
    for s in range(len(fr.STRATS)):
        for i, m in enumerate((60, 63, 67)):
            add(kalimba(mf(m)), r.tf(r.finish[s]) + 0.07 * i, 0.1, s)
    for i, m in enumerate((48, 55, 60, 63, 67, 72)):      # accord final do mineur
        add(kalimba(mf(m)), te + 0.09 * i, 0.18)
    out_ = []
    for ch in (L, R):
        a = sosfilt(butter(2, 6000, btype="lowpass", fs=SR, output="sos"), ch)
        out_.append(fm.reverb(a.astype(np.float32), SR, 1.8, 0.32))
    a = np.stack(out_, 1)
    a /= max(1e-9, np.abs(a).max())
    a = np.tanh(a * 1.8) / np.tanh(1.8)
    a *= 0.13 / max(1e-9, np.sqrt((a ** 2).mean()))
    a *= min(1.0, 0.92 / max(1e-9, np.abs(a).max()))
    f = int(0.05 * SR)
    a[-f:] *= np.linspace(1, 0, f)[:, None]
    return a


# ------------------------------- configuration -------------------------------
c = dict(lot.CONCEPTS[0])
lot.configure(c, 7, SEED)
colors = [s_[1] for s_ in fr.STRATS]
POOL = {p[0]: p for p in fb.POOL}
fr.STRATS[:] = [(nm, colors[i], POOL[nm][1], POOL[nm][2]) for i, nm in
                enumerate(("RANDOM", "SMALL FIRST", "LONG FIRST", "BEST COMBO"))]
fr.LAZY_FRAC = 0.17
fr.PSIZE = (4, 18, 0.5)
if "--scan" in sys.argv:
    for sd in range(1100, 1400):
        pieces, plans = fr.simulate(sd)
        fin = np.array([sum(q[2] * q[3] for q in pl if q) / fr.CAP * 100 for pl in plans])
        o = np.sort(fin)[::-1]
        w = fr.STRATS[int(np.argmax(fin))][0]
        placed = sum(sum(1 for q in pl if q) for pl in plans)
        if o[0] - o[1] < 0.8 and 120 <= placed <= 160:
            print(sd, w, placed, np.round(fin, 1).tolist(), round(o[0] - o[1], 2))
    sys.exit()
fr.SUB = "PACKING WARS #07 - Classic Duel"
fr.MELODY_MODULE = "fr_midi"
fr.MELODY = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.MUSICAL = fr.BRICKS = False
fr.LIVE_RANK = True
fr.SLOW_LAST = 4
fr.SPEEDS = [1.5, 1.25, 1.1, 0.8]                        # RANDOM le plus rapide ... BEST COMBO le plus lent
fr.CUSTOM_ST = split_schedule
fr.synth = lambda r, sr=SR: (np.zeros(int((r.total_out / fr.FPS + 0.5) * sr), np.int16), sr)  # son mixe a part (stereo)
r = fr.Renderer(SEED)
print("graine", SEED, "cadence", r.cadence, "tempo fin x%.2f" % r.rmax, "gagnant", fr.STRATS[r.winner][0],
      [round(x, 1) for x in r.final], "arrivees", {fr.STRATS[k][0]: v for k, v in r.rank.items()},
      "duree", round(r.total_out / fr.FPS, 1), flush=True)
tmp = out + ".noaudio.mp4"
fb.render(r, tmp)
a = synth_kalimba(r)
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
