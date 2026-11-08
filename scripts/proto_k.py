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
ROOT = 92.5                                              # fa# mineur
ALGOS = (("TETRIS", "tetris", 1.6), ("COLUMNS", "columns", 1.25), ("SPIRAL", "spiral", 1.0), ("SMART FIT", "smart_lazy", 0.8))
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


c = dict(lot.CONCEPTS[0])
lot.configure(c, 1, SEED)
colors = [s_[1] for s_ in fr.STRATS]
fr.STRATS[:] = [(n, colors[i], m, False) for i, (n, m, _) in enumerate(ALGOS)]
fr.SPEEDS = [v for _, _, v in ALGOS]
fr.TITLE = "SPEED OR BRAINS?"
fr.SUB = sys.argv[4] if len(sys.argv) > 4 else "PACKING WARS #03 - Speed Race"
fr.PSIZE = (3, 14, 0.5)
fr.TETRIS_K = 10
fr.LAZY_FRAC = 0.17
fr.MELODY_MODULE = "fr_midi"
fr.MUSICAL = fr.BRICKS = False
fr.MELODY = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.synth = synth_race
r = fr.Renderer(SEED)
print("graine", SEED, "pieces", r.n, "gagnant", fr.STRATS[r.winner][0], [round(x, 1) for x in r.final],
      "arrivees", {fr.STRATS[k][0]: v for k, v in r.rank.items()}, "duree", round(r.total_out / fr.FPS, 1), flush=True)
if "--dry" not in sys.argv:
    fb.render(r, out)
    print("ok", out)
