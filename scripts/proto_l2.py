"""Episode #04 : retour a la base (4 algos classiques du #01), effets visuels, sons doux et zen."""
import sys
import numpy as np
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "ep04_classic_zen_v2.mp4"
seed_in = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
ROOT = 110.0


def synth_zen(r, sr=44100):
    """Un 'pop' doux a chaque bloc pose (dans chaque boite, a son propre rythme) + notes zen eparses."""
    au = pw.Audio(r.total_out / fr.FPS + 3)
    S, n = len(fr.STRATS), r.n
    for s in range(S):
        k = 0
        for j, p in enumerate(r.plans[s]):
            if not p:
                continue
            t = r.tf(r.land[s, j])
            au.soft(t, min(1.0, p[2] * p[3] / 150), 0.08)
            if k % 3 == s % 3:
                au.pluck(t, 7 + int(9 * j / n) + s % 3, 0.035, root=ROOT, dur=1.2)
            k += 1
    for s in range(S):
        au.chord(r.tf(r.finish[s]), (5, 7, 9), 0.05, root=ROOT)
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
fr.LIVE_RANK = True
fr.SLOW_LAST = 4
fr.SPEEDS = [0.8, 1.1, 1.5, 1.25]                        # SMART FIT lent, BOTTOM-RIGHT, BIGGEST FIRST rapide, BOTTOM-LEFT
fr.synth = synth_zen
r = fr.Renderer(seed)
print("graine", seed, "algos", [s[0] for s in fr.STRATS], "gagnant", fr.STRATS[r.winner][0],
      [round(x, 1) for x in r.final], "ecart", round(r.gap, 2), "duree", round(r.total_out / fr.FPS, 1), flush=True)
fb.render(r, out)
print("ok", out)
