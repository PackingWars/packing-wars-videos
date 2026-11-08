"""Episode #04 : retour a la base (4 algos classiques du #01), effets visuels, sons doux et zen."""
import sys
import numpy as np
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "ep04_classic_zen.mp4"
seed_in = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
ROOT = 110.0


def synth_zen(r, sr=44100):
    au = pw.Audio(r.total_out / fr.FPS + 3)
    n = r.n
    for j in range(n):
        areas = [p[2] * p[3] for p in (pl[j] for pl in r.plans) if p]
        if not areas:
            continue
        t = min(r.tf(r.land[s, j]) for s in range(len(fr.STRATS)) if r.plans[s][j])
        frac = j / n
        au.soft(t, min(1.0, np.mean(areas) / 150), 0.10)
        if j % 2 == 0:
            au.pluck(t, 7 + int(frac * 9) + (j // 2) % 3, 0.045, root=ROOT, dur=1.4)
    te = r.tf(r.END)
    au.swell(te, 2.2, 0.05, root=ROOT * 2 ** (3 / 12))
    au.chord(te, (0, 5, 7, 9, 12), 0.1, root=ROOT)
    for i in range(10):
        au.soft(te + 0.15 + i * 0.12, 0.3 + 0.05 * i, 0.05)
    return au.render(r.total_out / fr.FPS, rev=0.32, drive=1.4, lowpass=5000), sr


c = dict(lot.CONCEPTS[0])
seed = seed_in
lot.configure(c, 4, seed)
colors = [s_[1] for s_ in fr.STRATS]
POOL = {p[0]: p for p in fb.POOL}
fr.STRATS[:] = [(nm, colors[i], POOL[nm][1], POOL[nm][2]) for i, nm in
                enumerate(("SMART FIT", "BOTTOM-RIGHT", "BIGGEST FIRST", "BOTTOM-LEFT"))]
if "--scan" in sys.argv:
    for sd in range(1200, 1260):
        pieces, plans = fr.simulate(sd)
        fin = np.array([sum(q[2] * q[3] for q in pl if q) / fr.CAP * 100 for pl in plans])
        o = np.sort(fin)[::-1]
        print(sd, [fr.STRATS[int(np.argmax(fin))][0]], np.round(fin, 1).tolist(), round(o[0] - o[1], 2))
    sys.exit()
fr.SUB = "PACKING WARS #04 - Classic Duel"
fr.MELODY = fr.MUSICAL = fr.BRICKS = False
fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
fr.synth = synth_zen
r = fr.Renderer(seed)
print("graine", seed, "algos", [s[0] for s in fr.STRATS], "gagnant", fr.STRATS[r.winner][0],
      [round(x, 1) for x in r.final], "ecart", round(r.gap, 2), "duree", round(r.total_out / fr.FPS, 1), flush=True)
fb.render(r, out)
print("ok", out)
