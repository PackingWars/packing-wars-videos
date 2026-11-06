#!/usr/bin/env python3
"""Mode lot : genere N videos variees (forme, theme, strategies) + legendes TikTok / Instagram.

Usage :
  python3 fill_race_batch.py --count 14 --outdir lot1
  python3 fill_race_batch.py --count 14 --start-episode 15 --start-seed 500 --outdir lot2
  python3 fill_race_batch.py --count 3 --captions-only          (sans rendre les videos)

Sortie : outdir/epNN_forme.mp4, outdir/captions.json, outdir/captions.csv
"""
import argparse
import csv
import json
import os
import subprocess

import numpy as np

import fill_race_v2 as fr

H, W = fr.H_BIN, fr.W_BIN
YY, XX = np.mgrid[0:H, 0:W]
XC, YC = XX + 0.5 - W / 2, YY + 0.5 - H / 2             # centre de chaque case, en cases
XN, YN = XC / (W / 2), -YC / (H / 2)                     # coordonnees normalisees, y vers le haut

SHAPES = ["RECTANGLE", "CIRCLE", "HEART", "DIAMOND", "TRIANGLE", "PLUS", "DONUT", "OBSTACLES"]

THEMES = {  # fond, 4 couleurs
    "NEON": ((14, 15, 22), [(74, 222, 128), (167, 139, 250), (251, 191, 36), (248, 113, 113)]),
    "OCEAN": ((8, 16, 28), [(56, 189, 248), (45, 212, 191), (129, 140, 248), (244, 114, 182)]),
    "SUNSET": ((24, 12, 20), [(251, 146, 60), (244, 63, 94), (250, 204, 21), (232, 121, 249)]),
    "CANDY": ((20, 14, 28), [(244, 114, 182), (110, 231, 183), (125, 211, 252), (253, 224, 71)]),
}
THEME_ORDER = list(THEMES)

# nom, mode de placement, ordre des pieces, description pour la legende
POOL = [
    ("RANDOM", "random", False, "drops each piece anywhere it fits"),
    ("BOTTOM-LEFT", "bl", False, "stacks from the bottom left"),
    ("BOTTOM-RIGHT", "br", False, "stacks from the bottom right"),
    ("BIGGEST FIRST", "bl", True, "places the biggest pieces first"),
    ("SMALL FIRST", "bl", "small", "places the smallest pieces first"),
    ("LONG FIRST", "bl", "long", "places the longest pieces first"),
    ("SMART FIT", "smart", False, "picks the spot that touches the most edges"),
    ("BEST COMBO", "smart", True, "biggest pieces first, best-contact spot"),
]
WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 1, 0.35])      # la combo est la plus forte : on la tire moins souvent
WEIGHTS = WEIGHTS / WEIGHTS.sum()
TITLES = ["WHO PACKS BEST?", "PICK THE WINNER", "WHICH STRATEGY WINS?", "WHO FILLS IT BEST?"]
TT_TAGS = ["#satisfying", "#asmr", "#algorithm", "#oddlysatisfying", "#fyp", "#simulation", "#coding", "#puzzle"]
IG_TAGS = ["#algorithms", "#simulation", "#satisfying", "#asmr", "#coding", "#reels", "#oddlysatisfying",
           "#computerscience", "#puzzle", "#visualization"]


def shape_mask(name, rng):
    if name == "RECTANGLE":
        return np.ones((H, W), bool)
    if name == "CIRCLE":
        return XC ** 2 + YC ** 2 <= (min(W, H) / 2) ** 2
    if name == "HEART":
        X, Y = XN * 1.3, YN * 1.25 + 0.15
        return (X ** 2 + Y ** 2 - 1) ** 3 - X ** 2 * Y ** 3 <= 0
    if name == "DIAMOND":
        return np.abs(XN) + np.abs(YN) <= 1
    if name == "TRIANGLE":
        return np.abs(XN) <= (1 - YN) / 2
    if name == "PLUS":
        return (np.abs(XC) <= 11) | (np.abs(YC) <= 11)
    if name == "DONUT":
        r2 = XC ** 2 + YC ** 2
        return (r2 <= (min(W, H) / 2) ** 2) & (r2 >= 11 ** 2)
    if name == "OBSTACLES":
        m = np.ones((H, W), bool)
        for _ in range(4):
            bw, bh = int(rng.integers(8, 15)), int(rng.integers(8, 15))
            x, y = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
            m[y:y + bh, x:x + bw] = False
        return m
    raise ValueError(name)


def configure(shape, theme, seed, ep, series):
    """Regle le moteur pour cette video. Retourne la description de la video."""
    rng = np.random.default_rng(seed + 7919)
    mask = shape_mask(shape, rng)
    bg, colors = THEMES[theme]
    picks = rng.choice(len(POOL), 4, replace=False, p=WEIGHTS)
    strats = [(POOL[p][0], colors[i], POOL[p][1], POOL[p][2]) for i, p in enumerate(picks)]
    fr.MASK, fr.CAP = mask, int(mask.sum())
    fr.BG = bg
    fr.EMPTY = tuple(min(255, c + d) for c, d in zip(bg, (16, 19, 24)))
    fr.STRATS[:] = strats
    fr.TITLE = TITLES[int(rng.integers(len(TITLES)))]
    fr.SUB = f"{series} #{ep:02d} - {shape.title()}"
    return {"descs": {POOL[p][0]: POOL[p][3] for p in picks}, "rng": rng}


def pick_video(ep, base_seed, last_winner, series):
    """Cherche une graine qui donne une bonne video : gagnant different du precedent, pas d'egalite."""
    shape = SHAPES[ep % len(SHAPES)]
    theme = THEME_ORDER[(ep + ep // len(SHAPES)) % len(THEME_ORDER)]
    r = info = seed = None
    for k in range(40):
        seed = base_seed + k
        info = configure(shape, theme, seed, ep, series)
        r = fr.Renderer(seed)
        win = max(r.final)
        if fr.STRATS[r.winner][0] != last_winner and win >= 80 and r.gap >= 0.15:
            break
    return shape, theme, seed, r, info


def captions(ep, shape, r, info):
    names = [s[0] for s in fr.STRATS]
    rng = info["rng"]
    tt = " ".join(rng.choice(TT_TAGS, 5, replace=False))
    ig = " ".join(rng.choice(IG_TAGS, 8, replace=False))
    win = fr.STRATS[r.winner][0]
    order = np.argsort(r.final)[::-1]
    second = fr.STRATS[order[1]][0]
    shape_t = shape.lower()
    tiktok = (f"Which strategy fills the {shape_t} best? {', '.join(names)}. "
              f"Pick one before the end {tt}")
    lines = "\n".join(f"{n} {info['descs'][n]}." for n in names)
    insta = (f"Which strategy would you pick?\n\n{lines}\n\n"
             f"Same pieces, same box. Only the strategy changes.\n\n{ig}")
    reveal = (f"Answer: {win} won with {r.final[r.winner]:.1f}% "
              f"({second} was {r.gap:.1f}% behind)")
    return tiktok, insta, reveal


def render(r, out):
    base = os.path.splitext(out)[0]
    tmp_v, tmp_a, tmp_o = base + ".v.tmp.mp4", base + ".tmp.wav", base + ".tmp.mp4"
    data, sr = fr.synth(r)
    fr.write_wav(tmp_a, data, sr)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{fr.VW}x{fr.VH}", "-r", str(fr.FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", tmp_v]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for F in range(r.total_out):
        p.stdin.write(np.ascontiguousarray(r.compose(r.tau_of(F), F)).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg video a echoue")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_v, "-i", tmp_a, "-map", "0:v",
                    "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", tmp_o],
                   check=True)
    os.replace(tmp_o, out)
    for t in (tmp_v, tmp_a):
        os.remove(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=7)
    ap.add_argument("--start-episode", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=100)
    ap.add_argument("--series", default="PACKING WARS")
    ap.add_argument("--outdir", default="lot")
    ap.add_argument("--captions-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows, last_winner = [], None
    for i in range(args.count):
        ep = args.start_episode + i
        shape, theme, seed, r, info = pick_video(ep, args.start_seed + 50 * i, last_winner, args.series)
        last_winner = fr.STRATS[r.winner][0]
        tiktok, insta, reveal = captions(ep, shape, r, info)
        name = f"ep{ep:02d}_{shape.lower()}.mp4"
        print(f"[{i + 1}/{args.count}] {name} | theme {theme} | graine {seed} | "
              f"gagnant {last_winner} {r.final[r.winner]:.1f}% (ecart {r.gap:.1f}%) | {r.total_out / fr.FPS:.0f}s",
              flush=True)
        if not args.captions_only:
            render(r, os.path.join(args.outdir, name))
        rows.append({"file": name, "episode": ep, "shape": shape, "theme": theme, "seed": seed,
                     "strategies": [s[0] for s in fr.STRATS], "winner": last_winner,
                     "winner_fill": round(r.final[r.winner], 1), "gap": round(r.gap, 2),
                     "photo_finish": bool(r.photo), "duration_s": round(r.total_out / fr.FPS, 1),
                     "tiktok_caption": tiktok, "instagram_caption": insta, "reveal_comment": reveal})
    with open(os.path.join(args.outdir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "captions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["file", "episode", "shape", "theme", "winner", "winner_fill", "tiktok_caption",
                "instagram_caption", "reveal_comment"]
        w.writerow(cols)
        for row in rows:
            w.writerow([row[c] for c in cols])
    print("ok ->", args.outdir)


if __name__ == "__main__":
    main()
