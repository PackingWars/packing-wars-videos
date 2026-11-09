"""Episode #03 : course asynchrone, chaque algo pose a sa propre vitesse. Melodie MIDI (rythme regulier) jouee par l'algo le
plus lent ; les autres poses font des 'pops' doux accordes dans la gamme. Badge d'arrivee 1st..4th DONE."""
import sys
import numpy as np
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import fr_midi
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "ep03_speed_race.mp4"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1173
fr_midi.PATH = "/home/claude/bluebird.mid"
fr_midi.START = 0
fr_midi.STEADY_FRAMES = float(sys.argv[3]) if len(sys.argv) > 3 else 8.8
if fr_midi.STEADY_FRAMES == 0:                          # rythme du MIDI d'origine (avec ses pauses)
    fr_midi.SPEED = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
ROOT = 92.5                                              # fa# mineur
ALGOS = (("TETRIS", "tetris", 1.6), ("COLUMNS", "columns", 1.25), ("SPIRAL", "spiral", 1.0), ("SMART FIT", "smart_lazy", 0.8))
if fr_midi.STEADY_FRAMES == 0:                           # Smart Fit joue la melodie au tempo du MIDI, les autres vont plus vite
    ALGOS = tuple((a, b, v / 0.8) for a, b, v in ALGOS)
MEL = 3                                                  # l'algo le plus lent porte la melodie


def synth_race(r, sr=44100):
    au = pw.Audio(r.total_out / fr.FPS + 3)
    midi = r.mel["midi"]
    for s in range(len(fr.STRATS)):
        k = 0
        for j, p in enumerate(r.plans[s]):
            if not p:
                continue
            t = r.tf(r.land[s, j])
            if s == MEL:
                au.mallet(t, midi[j] + 12, 0.17)
                au.soft(t, min(1.0, p[2] * p[3] / 150), 0.05)
            else:
                au.pluck(t, 9 + (k * 2 + s) % 7, 0.035, root=ROOT, dur=0.8)
                au.soft(t, min(1.0, p[2] * p[3] / 150), 0.045)
            k += 1
    for s in range(len(fr.STRATS)):                     # petit accord quand un algo a fini
        au.chord(r.tf(r.finish[s]), (5, 7, 9), 0.05, root=ROOT)
    te = r.tf(r.END)
    au.swell(te, 2.0, 0.05, root=ROOT * 2 ** (3 / 12))
    au.chord(te, (0, 5, 7, 9, 12), 0.1, root=ROOT)
    return au.render(r.total_out / fr.FPS, rev=0.32, drive=1.4, lowpass=5000), sr


SPLIT = "--split" in sys.argv                            # option 2 : chaque note du MIDI fait tomber UN bloc, dans une des 4 boites
SPLIT_SEC = 40.0
ACCEL = "--accel" in sys.argv                            # debut au tempo d'origine (SLOW_SEC s) puis acceleration
SLOW_SEC = 10.0
BIG = "--big" in sys.argv
CADENCE = "--cadence" in sys.argv                        # la derniere pose tombe sur une fin de phrase musicale                                # blocs plus gros : environ 2 fois moins de notes


def split_schedule(r):
    S, n = len(fr.STRATS), r.n
    ev = []
    for s in range(S):
        ks = [j for j in range(n) if r.plans[s][j]]
        for k, j in enumerate(ks):
            ev.append(((k + 1) / fr.SPEEDS[s], s, j))
    ev.sort()
    rh, lh = fr_midi._load()
    span0 = rh[-1][0] - rh[0][0] + 1.0
    times, notes = [], []
    E = len(ev)
    idx = list(range(E))
    if CADENCE:                                          # finir sur une fin de phrase (note longue) : on retire quelques
        T = [x[0] for x in rh]                           # notes repetees rapides pour tomber pile dessus
        P = [x[2] for x in rh]
        ioi = list(np.diff(T)) + [2.0]
        ends = [i for i in range(len(rh)) if ioi[i] >= 1.2 and i + 1 >= E]
        if ends:
            e = ends[0]
            drop_n = e + 1 - E
            rep = [i for i in range(1, e) if P[i] == P[i - 1] and T[i] - T[i - 1] <= 0.3]
            if len(rep) >= drop_n:
                pick = [rep[int(k)] for k in np.linspace(len(rep) // 3, len(rep) - 1, drop_n)] if drop_n else []
                idx = [i for i in range(e + 1) if i not in set(pick)]
                r.cadence = (e, pick)
    for k, i in enumerate(idx):
        t, d, p = rh[i % len(rh)]
        times.append(t - rh[0][0] + (i // len(rh)) * span0)
        notes.append(p)
    times = np.array(times)
    if ACCEL:                                            # tempo d'origine au debut, puis acceleration progressive
        O = times[-1]
        p1 = min(0.9, SLOW_SEC / O)

        def warp(rmax):
            pp = times / O
            rate = np.where(pp < p1, 1.0, 1 + (rmax - 1) * (pp - p1) / (1 - p1))
            dt = np.diff(times, prepend=0.0) / rate
            return np.cumsum(dt)
        lo, hi = 1.0, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if warp(mid)[-1] > SPLIT_SEC else (lo, mid)
        out_t = warp(hi)
        r.rmax = hi
        land = 14 + out_t * fr.FPS
        factor = O / out_t[-1]
    else:
        factor = times[-1] / SPLIT_SEC
        land = 14 + times / factor * fr.FPS
    st = np.full((S, n), np.nan)
    note_of = {}
    for (v, s, j), lf, p in zip(ev, land, notes):
        st[s, j] = lf - fr.MELODY_DROP
        note_of[(s, j)] = p
    for s in range(S):                                   # pieces refusees : jamais dessinees, on recopie le voisin
        row = st[s]
        first = row[~np.isnan(row)][0]
        last = first
        for j in range(n):
            if np.isnan(row[j]):
                row[j] = last
            else:
                last = row[j]
    r.split_factor = factor
    tt = [t - rh[0][0] for t, p in lh]
    r.split_lh = [(float(np.interp(t, times, land)), p) for t, (_, p) in zip(tt, lh) if 0 <= t <= times[-1]]
    return st, note_of


def synth_split(r, sr=44100):
    au = pw.Audio(r.total_out / fr.FPS + 3)
    for (s, j), m in r.note_of.items():
        t = r.tf(r.land[s, j])
        p = r.plans[s][j]
        au.mallet(t, m + 12, 0.13)
        au.soft(t, min(1.0, p[2] * p[3] / 150), 0.04)
    for f, m in r.split_lh:
        au.mallet(r.tf(f), m + 12, 0.05)
    for s in range(len(fr.STRATS)):
        au.chord(r.tf(r.finish[s]), (5, 7, 9), 0.05, root=ROOT)
    te = r.tf(r.END)
    au.swell(te, 2.0, 0.05, root=ROOT * 2 ** (3 / 12))
    au.chord(te, (0, 5, 7, 9, 12), 0.1, root=ROOT)
    return au.render(r.total_out / fr.FPS, rev=0.32, drive=1.4, lowpass=5000), sr


c = dict(lot.CONCEPTS[0])
lot.configure(c, 1, SEED)
colors = [s_[1] for s_ in fr.STRATS]
fr.STRATS[:] = [(n, colors[i], m, False) for i, (n, m, _) in enumerate(ALGOS)]
fr.SPEEDS = [v for _, _, v in ALGOS]
fr.TITLE = "SPEED OR BRAINS?"
fr.SUB = sys.argv[4] if len(sys.argv) > 4 else "PACKING WARS #03 - Speed Race"
fr.PSIZE = (4, 20, 0.5) if BIG else (3, 14, 0.5)
fr.TETRIS_K = 10
fr.LAZY_FRAC = 0.17
fr.MELODY_MODULE = "fr_midi"
fr.MUSICAL = fr.BRICKS = False
fr.MELODY = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.LIVE_RANK = True
fr.synth = synth_race
if SPLIT:
    fr.SPEEDS = [v for _, _, v in (("TETRIS", "tetris", 1.6), ("COLUMNS", "columns", 1.25), ("SPIRAL", "spiral", 1.0), ("SMART FIT", "smart_lazy", 0.8))]
    fr.CUSTOM_ST = split_schedule
    fr.SLOW_LAST = 4
    fr.synth = synth_split
r = fr.Renderer(SEED)
print("cadence", getattr(r, "cadence", None), "tempo fin x%.2f" % getattr(r, "rmax", 0), "graine", SEED, "pieces", r.n, "gagnant", fr.STRATS[r.winner][0], [round(x, 1) for x in r.final],
      "arrivees", {fr.STRATS[k][0]: v for k, v in r.rank.items()}, "duree", round(r.total_out / fr.FPS, 1), flush=True)
if "--dry" not in sys.argv:
    fb.render(r, out)
    print("ok", out)
