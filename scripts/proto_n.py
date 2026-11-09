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

out = sys.argv[1] if len(sys.argv) > 1 else "ep05_unravel_v2.mp4"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
SPLIT_SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 32.0
SLOW_SEC = 8.0
fr_midi.PATH = "/home/claude/unravel.mid"
START = int(sys.argv[4]) if len(sys.argv) > 4 else 187     # entree du refrain (184 = 3 notes d'elan avant)
ROOT = 98.0                                              # sol mineur
SR = 44100


def split_schedule(r):
    S, n = len(fr.STRATS), r.n
    ev = []
    for s in range(S):
        ks = [j for j in range(n) if r.plans[s][j]]
        for k, j in enumerate(ks):
            ev.append(((k + 1) / fr.SPEEDS[s], s, j))
    ev.sort()
    rh_all, lh_all = fr_midi._load()
    rh = rh_all[START:] + rh_all[:START]
    E = len(ev)
    T = [x[0] - rh[0][0] if x[0] >= rh[0][0] else x[0] + rh_all[-1][0] + 1 - rh[0][0] for x in rh]
    P = [x[2] for x in rh]
    ioi = list(np.diff(T)) + [2.0]
    idx = list(range(E))
    ends = [i for i in range(len(rh)) if ioi[i] >= 1.0 and i + 1 >= E]
    r.cadence = None
    if ends:                                             # finir sur une fin de phrase
        e = ends[0]
        rep = [i for i in range(1, e) if P[i] == P[i - 1] and T[i] - T[i - 1] <= 0.3]
        dn = e + 1 - E
        if len(rep) >= dn:
            pick = set(rep[int(k)] for k in np.linspace(len(rep) // 3, len(rep) - 1, dn)) if dn else set()
            if len(pick) == dn:
                idx = [i for i in range(e + 1) if i not in pick]
                r.cadence = (e, sorted(pick))
    times = np.array([T[i] for i in idx])
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
    t0 = rh_all[START][0]
    r.split_lh = [(float(np.interp(t - t0, times, land)), p) for t, p in lh_all if 0 <= t - t0 <= O]
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
        for i, m in enumerate((67, 70, 74)):
            add(kalimba(mf(m)), r.tf(r.finish[s]) + 0.07 * i, 0.1, s)
    for i, m in enumerate((55, 62, 67, 70, 74, 79)):      # accord final sol mineur
        add(kalimba(mf(m)), te + 0.09 * i, 0.18)
    k = 0
    for s in range(len(fr.STRATS)):                       # les perdants s'ecroulent : petites notes graves qui retombent
        if s == r.winner:
            continue
        for j, p in enumerate(r.plans[s]):
            if p and k % 4 == 0:
                add(kalimba(mf(43 + (j * 5) % 12)), r.tf(r.fall_t0(s, j)), 0.05, s)
            k += 1
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
lot.configure(c, 5, SEED)
colors = [s_[1] for s_ in fr.STRATS]
POOL = {p[0]: p for p in fb.POOL}
fr.STRATS[:] = [(nm, colors[i], POOL[nm][1], POOL[nm][2]) for i, nm in
                enumerate(("SMART FIT", "BOTTOM-RIGHT", "BIGGEST FIRST", "BOTTOM-LEFT"))]
fr.STRATS[0] = ("SMART FIT", colors[0], "smart_lazy", False)
fr.LAZY_FRAC = 0.17
fr.PSIZE = (4, 20, 0.5)
if "--scan" in sys.argv:
    for sd in range(1100, 1400):
        pieces, plans = fr.simulate(sd)
        fin = np.array([sum(q[2] * q[3] for q in pl if q) / fr.CAP * 100 for pl in plans])
        o = np.sort(fin)[::-1]
        w = fr.STRATS[int(np.argmax(fin))][0]
        placed = sum(sum(1 for q in pl if q) for pl in plans)
        if w in ("BOTTOM-LEFT", "BOTTOM-RIGHT", "BIGGEST FIRST") and o[0] - o[1] < 0.8 and placed <= 170:
            print(sd, w, placed, np.round(fin, 1).tolist(), round(o[0] - o[1], 2))
    sys.exit()
fr.SUB = "PACKING WARS #05 - Classic Duel"
fr.MELODY_MODULE = "fr_midi"
fr.MELODY = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.MUSICAL = fr.BRICKS = False
fr.ENDING = "fall"
fr.HOLD_FRAMES = 110
fr.LIVE_RANK = True
fr.SLOW_LAST = 4
fr.SPEEDS = [0.8, 1.1, 1.5, 1.25]
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
