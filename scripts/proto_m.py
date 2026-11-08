"""Episode #05 : 4 algos classiques, refrain d'Unravel (MIDI, rythme regulier, sons zen), fin ou les perdants s'ecroulent."""
import sys
import numpy as np
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import fr_midi
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "ep05_unravel.mp4"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1260
fr_midi.PATH = "/home/claude/unravel.mid"
fr_midi.START = 187                                      # entree du refrain
fr_midi.STEADY_FRAMES = float(sys.argv[3]) if len(sys.argv) > 3 else 9
ROOT = 98.0                                              # sol mineur


def synth(r, sr=44100):
    au = pw.Audio(r.total_out / fr.FPS + 3)
    S = len(fr.STRATS)
    for j, m in enumerate(r.mel["midi"]):
        ls = [r.land[s, j] for s in range(S) if r.plans[s][j]]
        if not ls:
            continue
        t = r.tf(min(ls))
        areas = [r.plans[s][j][2] * r.plans[s][j][3] for s in range(S) if r.plans[s][j]]
        au.mallet(t, m + 12 if m < 72 else m, 0.16)
        au.soft(t, min(1.0, np.mean(areas) / 150), 0.06)
    for f, m in r.mel["lh"]:
        au.mallet(r.tf(f), m + 12, 0.06)
    te = r.tf(r.END)
    au.swell(te, 2.0, 0.05, root=ROOT * 2 ** (3 / 12))
    au.chord(te, (0, 5, 7, 9, 12), 0.1, root=ROOT)
    k = 0
    for s in range(S):                                   # cascade douce quand les perdants s'ecroulent
        if s == r.winner:
            continue
        for j, p in enumerate(r.plans[s]):
            if p and k % 3 == 0:
                au.soft(r.tf(r.fall_t0(s, j)), min(1.0, p[2] * p[3] / 150), 0.035)
            k += 1
    return au.render(r.total_out / fr.FPS, rev=0.32, drive=1.4, lowpass=5000), sr


c = dict(lot.CONCEPTS[0])
lot.configure(c, 5, seed)
colors = [s_[1] for s_ in fr.STRATS]
POOL = {p[0]: p for p in fb.POOL}
fr.STRATS[:] = [(nm, colors[i], POOL[nm][1], POOL[nm][2]) for i, nm in
                enumerate(("SMART FIT", "BOTTOM-RIGHT", "BIGGEST FIRST", "BOTTOM-LEFT"))]
fr.STRATS[0] = ("SMART FIT", colors[0], "smart_lazy", False)          # Smart Fit bride : il ne doit pas gagner celle-ci
fr.LAZY_FRAC = 0.17
if "--scan" in sys.argv:
    for sd in range(1260, 1400):
        pieces, plans = fr.simulate(sd)
        fin = np.array([sum(q[2] * q[3] for q in pl if q) / fr.CAP * 100 for pl in plans])
        o = np.sort(fin)[::-1]
        w = fr.STRATS[int(np.argmax(fin))][0]
        if w != "SMART FIT" and o[0] - o[1] < 0.8 and fin[0] >= o[1] - 0.1:
            print(sd, w, len(pieces), np.round(fin, 1).tolist(), round(o[0] - o[1], 2))
    sys.exit()
fr.SUB = "PACKING WARS #05 - Classic Duel"
fr.MELODY_MODULE = "fr_midi"
fr.ENDING = "fall"
fr.HOLD_FRAMES = 110
fr.MUSICAL = fr.BRICKS = False
fr.MELODY = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.synth = synth
r = fr.Renderer(seed)
print("graine", seed, "pieces", r.n, "gagnant", fr.STRATS[r.winner][0], [round(x, 1) for x in r.final],
      "ecart", round(r.gap, 2), "duree", round(r.total_out / fr.FPS, 1), flush=True)
fb.render(r, out)
print("ok", out)
