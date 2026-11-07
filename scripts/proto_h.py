"""Prototype H : ep01 refait, effets visuels max, SANS musique : uniquement des bruitages de briques."""
import sys
import fill_race_v2 as fr
import fill_race_batch as fb
import fill_race_lot1 as lot

out = sys.argv[1] if len(sys.argv) > 1 else "ep01_classic_v3_bricks.mp4"
seed_in = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
c = lot.CONCEPTS[0]
seed, r, rng, descs = lot.pick(c, 1, seed_in, None)
fr.MELODY = fr.MUSICAL = False
fr.BRICKS = fr.BLOOM = fr.MAXFX = fr.COLOR_FLOW = True
r = fr.Renderer(seed)
print("graine", seed, "gagnant", fr.STRATS[r.winner][0], round(r.final[r.winner], 1), "ecart", round(r.gap, 2),
      "duree", round(r.total_out / fr.FPS, 1), flush=True)
fb.render(r, out)
print("ok", out)
