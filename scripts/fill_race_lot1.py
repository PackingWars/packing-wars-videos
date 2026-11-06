#!/usr/bin/env python3
"""Lot 1 : 10 concepts dans la boite rectangulaire (meme principe, petites variantes).

Usage :
  python3 fill_race_lot1.py --outdir ../videos/lot1                 (tout rendre)
  python3 fill_race_lot1.py --only 2 5 --outdir ../videos/lot1      (certains concepts)
  python3 fill_race_lot1.py --captions-only --outdir /tmp/x         (sans rendre les videos)
Sortie : outdir/epNN_slug.mp4, captions.json, captions.csv
"""
import argparse
import csv
import json
import os

import numpy as np

import fill_race_v2 as fr
import fill_race_batch as fb

POOL = {p[0]: p for p in fb.POOL}

CONCEPTS = [
    dict(slug="classic", name="Classic Duel", theme="NEON", title="WHO PACKS BEST?", strats=None,
         hook="Which strategy fills the box best?"),
    dict(slug="size-wars", name="Size Wars", theme="OCEAN", title="BIG, SMALL OR LONG?",
         strats=["BIGGEST FIRST", "SMALL FIRST", "LONG FIRST", "RANDOM"],
         hook="Big, small or long pieces first?"),
    dict(slug="corner-wars", name="Corner Wars", theme="SUNSET", title="DOES THE CORNER MATTER?",
         strats=["BOTTOM-LEFT", "BOTTOM-RIGHT", "RANDOM", "SMART FIT"],
         hook="Does the starting corner matter?"),
    dict(slug="luck-vs-strategy", name="Luck vs Strategy", theme="CANDY", title="CAN LUCK BEAT STRATEGY?",
         strats=["RANDOM", "SMART FIT", "BIGGEST FIRST", "BEST COMBO"],
         hook="Can pure luck beat a real strategy?"),
    dict(slug="mystery", name="Mystery Strategies", theme="OCEAN", title="GUESS THE STRATEGY", strats=None,
         mystery=True, hook="Which of A, B, C, D is the smartest strategy?"),
    dict(slug="giant-blocks", name="Giant Blocks", theme="SUNSET", title="GIANT BLOCKS", strats=None,
         psize=(6, 23, 0.3), min_win=65, hook="Only a few giant pieces. Who survives?"),
    dict(slug="tiny-pieces", name="Tiny Pieces", theme="CANDY", title="TINY PIECES RACE", strats=None,
         psize=(2, 9, 0.5), hook="Hundreds of tiny pieces. Who wins?"),
    dict(slug="photo-finish", name="Photo Finish", theme="NEON", title="WON BY A HAIR", strats=None,
         gap_max=0.3, hook="This one is decided by a few cells. Who wins?"),
    dict(slug="rainbow", name="Rainbow Pieces", theme="CANDY", title="COLOR = PIECE SIZE", strats=None,
         by_size=True, hook="Colors show piece size. Which strategy packs best?"),
    dict(slug="speedrun", name="Speedrun", theme="OCEAN", title="SPEEDRUN RACE", strats=None,
         place=330, hold=60, hook="Quick race. Who wins?"),
]


def reset_engine():
    fr.PSIZE = (3, 17, 0.5)
    fr.MYSTERY = False
    fr.COLOR_BY_SIZE = False
    fr.PLACE_FRAMES, fr.HOLD_FRAMES = 540, 90
    fr.MASK = np.ones((fr.H_BIN, fr.W_BIN), bool)
    fr.CAP = int(fr.MASK.sum())


def configure(c, ep, seed):
    reset_engine()
    rng = np.random.default_rng(seed + 7919)
    bg, colors = fb.THEMES[c["theme"]]
    if c["strats"]:
        names = c["strats"]
    else:
        names = [fb.POOL[p][0] for p in rng.choice(len(fb.POOL), 4, replace=False, p=fb.WEIGHTS)]
    picks = [POOL[n] for n in names]
    fr.STRATS[:] = [(p[0], colors[i], p[1], p[2]) for i, p in enumerate(picks)]
    fr.BG = bg
    fr.EMPTY = tuple(min(255, v + d) for v, d in zip(bg, (16, 19, 24)))
    fr.TITLE = c["title"]
    fr.SUB = f"PACKING WARS #{ep:02d} - {c['name']}"
    fr.PSIZE = c.get("psize", fr.PSIZE)
    fr.MYSTERY = c.get("mystery", False)
    fr.COLOR_BY_SIZE = c.get("by_size", False)
    fr.PLACE_FRAMES = c.get("place", fr.PLACE_FRAMES)
    fr.HOLD_FRAMES = c.get("hold", fr.HOLD_FRAMES)
    return rng, {p[0]: p[3] for p in picks}


def pick(c, ep, base_seed, last_winner):
    best = None
    for k in range(120):
        seed = base_seed + k
        rng, descs = configure(c, ep, seed)
        r = fr.Renderer(seed)
        win = max(r.final)
        gmax = c.get("gap_max", 99)
        gmin = 0.02 if "gap_max" in c else 0.15
        ok = win >= c.get("min_win", 80) and gmin <= r.gap < gmax
        if ok and (fr.STRATS[r.winner][0] != last_winner or k >= 40):
            return seed, r, rng, descs
        score = -abs(r.gap - 0.15) if "gap_max" in c else win
        if best is None or score > best[0]:
            best = (score, seed)
    seed = best[1]                                         # repli : meilleur essai
    rng, descs = configure(c, ep, seed)
    return seed, fr.Renderer(seed), rng, descs


def captions(c, r, rng, descs):
    names = [s[0] for s in fr.STRATS]
    tags = " ".join(rng.choice(fb.TT_TAGS, 5, replace=False))
    ig = " ".join(rng.choice(fb.IG_TAGS, 8, replace=False))
    win = fr.STRATS[r.winner][0]
    order = np.argsort(r.final)[::-1]
    second = fr.STRATS[order[1]][0]
    if c.get("mystery"):
        tiktok = f"{c['hook']} Names revealed at the end. Pick one before the end {tags}"
        mapping = ", ".join(f"{'ABCD'[i]} = {n}" for i, n in enumerate(names))
        reveal = (f"Answer: {'ABCD'[r.winner]} ({win}) won with {r.final[r.winner]:.1f}% "
                  f"({second} was {r.gap:.1f}% behind). {mapping}")
        insta = f"{c['hook']}\n\nSame pieces, same box. Names revealed at the end.\n\n{ig}"
    else:
        tiktok = f"{c['hook']} {', '.join(names)}. Pick one before the end {tags}"
        lines = "\n".join(f"{n} {descs[n]}." for n in names)
        insta = f"{c['hook']}\n\n{lines}\n\nSame pieces, same box. Only the strategy changes.\n\n{ig}"
        reveal = (f"Answer: {win} won with {r.final[r.winner]:.1f}% "
                  f"({second} was {r.gap:.1f}% behind)")
    return tiktok, insta, reveal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="lot1")
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--captions-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows, last_winner = [], None
    for i, c in enumerate(CONCEPTS):
        ep = i + 1
        if args.only and ep not in args.only:
            continue
        seed, r, rng, descs = pick(c, ep, 1000 + 100 * ep, last_winner)
        last_winner = fr.STRATS[r.winner][0]
        tiktok, insta, reveal = captions(c, r, rng, descs)
        name = f"ep{ep:02d}_{c['slug']}.mp4"
        print(f"[{ep}/10] {name} | graine {seed} | pieces {r.n} | gagnant {last_winner} "
              f"{r.final[r.winner]:.1f}% (ecart {r.gap:.2f}%) | {r.total_out / fr.FPS:.1f}s", flush=True)
        if not args.captions_only:
            fb.render(r, os.path.join(args.outdir, name))
        rows.append({"file": name, "episode": ep, "concept": c["name"], "theme": c["theme"], "seed": seed,
                     "strategies": [s[0] for s in fr.STRATS], "winner": last_winner,
                     "winner_fill": round(r.final[r.winner], 1), "gap": round(r.gap, 2),
                     "photo_finish": bool(r.photo), "duration_s": round(r.total_out / fr.FPS, 1),
                     "pieces": r.n, "tiktok_caption": tiktok, "instagram_caption": insta,
                     "reveal_comment": reveal})
    with open(os.path.join(args.outdir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "captions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["file", "episode", "concept", "theme", "winner", "winner_fill", "duration_s", "tiktok_caption",
                "instagram_caption", "reveal_comment"]
        w.writerow(cols)
        for row in rows:
            w.writerow([row[k] for k in cols])
    print("ok ->", args.outdir)


if __name__ == "__main__":
    main()
