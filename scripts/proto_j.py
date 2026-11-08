"""Prototype J : SPIRAL / EXPLOSION / COLUMNS / TETRIS, effets visuels, chaque piece = note suivante du MIDI a rythme regulier, sons zen."""
import sys
import numpy as np
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import fr_midi
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "protoJ_shapes_tetris.mp4"
fr_midi.PATH = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/cyberpunk.mid"
fr_midi.START = int(sys.argv[3]) if len(sys.argv) > 3 else 82
fr_midi.STEADY_FRAMES = float(sys.argv[4]) if len(sys.argv) > 4 else 11
ROOT = float(sys.argv[5]) if len(sys.argv) > 5 else 77.78
CHORD = tuple(int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "1,6,8,9,11").split(","))


def synth_zen(r, sr=44100):
    au = pw.Audio(r.total_out / fr.FPS + 3)
    for j, m in enumerate(r.mel["midi"]):
        t = r.tf(r.starts[j] + fr.MELODY_DROP)
        areas = [p[2] * p[3] for p in (pl[j] for pl in r.plans) if p]
        au.mallet(t, m + 12, 0.16)
        au.soft(t, min(1.0, (np.mean(areas) if areas else 20) / 150), 0.06)
    for f, m in r.mel["lh"]:
        au.mallet(r.tf(f), m + 12, 0.06)
    te = r.tf(r.END)
    au.swell(te, 2.0, 0.05, root=ROOT * 2 ** (3 / 12))
    au.chord(te, CHORD, 0.1, root=ROOT)
    for i in range(10):
        au.soft(te + 0.15 + i * 0.12, 0.3 + 0.05 * i, 0.05)
    return au.render(r.total_out / fr.FPS, rev=0.32, drive=1.4, lowpass=5000), sr


SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 1173
c = dict(lot.CONCEPTS[0])
lot.configure(c, 1, SEED)
seed = SEED
colors = [s_[1] for s_ in fr.STRATS]
fr.STRATS[:] = [(n, colors[i], m, False) for i, (n, m) in enumerate(
    (("SPIRAL", "spiral"), ("EXPLOSION", "explosion"), ("COLUMNS", "columns"), ("TETRIS", "tetris")))]
fr.SUB = sys.argv[8] if len(sys.argv) > 8 else "PACKING WARS #02 - Shape Wars"
fr.PSIZE = (3, 14, 0.5)
fr.TETRIS_K = 10
fr.MELODY_MODULE = "fr_midi"
fr.MUSICAL = fr.BRICKS = False
fr.MELODY = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.synth = synth_zen
r = fr.Renderer(seed)
print("graine", seed, "pieces", r.n, "gagnant", fr.STRATS[r.winner][0], round(r.final[r.winner], 1),
      "ecart", round(r.gap, 2), "duree", round(r.total_out / fr.FPS, 1), flush=True)
fb.render(r, out)
print("ok", out)
