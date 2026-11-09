#!/usr/bin/env python3
"""Course de remplissage v2 : 4 strategies rangent les memes pieces dans une boite.

Nouveautes : pieces qui tombent et rebondissent, son riche (clac + notes + ding),
photo-finish (ralenti + zoom), reflets neon + etincelles, boucle parfaite.

Sortie : MP4 vertical 1080x1920, 30 fps, avec son.
Usage  : python3 fill_race_v2.py --seed 4 --out video.mp4
         python3 fill_race_v2.py --seed 4 --dry
         python3 fill_race_v2.py --seed 4 --preview 120 330 --prev-dir dossier
"""
import argparse
import os
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- reglages ----------
W_BIN, H_BIN, CELL = 60, 70, 7
PW, PH = W_BIN * CELL, H_BIN * CELL      # taille d'un panneau en px
VW, VH, FPS = 1080, 1920, 30
PLACE_FRAMES = 540                       # duree de la phase de pose (anim)
HOLD_FRAMES = 90                         # pause finale avec le gagnant
RESET_FRAMES = 36                        # remise a zero pour la boucle
FILL_TARGET = 1.35
SQ, PL, NP = 6, 14, 7                    # ecrasement, vie des etincelles, nb d'etincelles
SLOW_K, SLOW_RATE, PHOTO_GAP = 6, 0.35, 2.0

BG = (14, 15, 22)
EMPTY = (30, 34, 46)
GREY = (150, 150, 165)
SUB = "Same pieces. 4 strategies. Pick one."
STRATS = [
    ("RANDOM", (74, 222, 128), "random", False),
    ("BOTTOM-LEFT", (167, 139, 250), "bl", False),
    ("BIGGEST FIRST", (251, 191, 36), "bl", True),
    ("SMART FIT", (248, 113, 113), "smart", False),
]
PX = [90, 570]
PY = [505, 1075]
BAR_Y0, BAR_STEP, BAR_H = 255, 40, 28
BAR_X0, BAR_X1 = 330, 880
TITLE = "WHO PACKS BEST?"
MASK = np.ones((H_BIN, W_BIN), bool)     # cases utilisables de la boite (sa forme)
CAP = int(MASK.sum())                    # nombre de cases utilisables
PSIZE = (3, 17, 0.5)                     # tailles de pieces : min, max (exclu), exposant du poids 1/taille**exp
MYSTERY = False                          # True : noms caches (A,B,C,D) puis reveles a la fin
COLOR_BY_SIZE = False                    # True : pieces colorees selon leur aire, pas selon la strategie
RAINBOW = [(96, 165, 250), (45, 212, 191), (74, 222, 128), (250, 204, 21), (251, 146, 60), (244, 63, 94)]
AREA_EDGES = (24, 48, 80, 120, 170)      # seuils d'aire -> 6 couleurs (petit -> grand)
BRICKS = False                           # True : pas de musique, seulement des bruitages de briques (clac, tock, thump)
MUSICAL = False                          # True : chaque piece joue une note (gamme pentatonique) + nappe + reverb
BLOOM = False                            # True : halo lumineux sur les zones claires (look neon)
MELODY = False                           # True : les pieces se posent sur les notes d'une melodie (fr_melody)
ENDING = None                             # "fall" : les boites perdantes se vident, leurs pieces tombent
LIVE_RANK = False                         # classement en direct : les barres se reordonnent, couronne sur le 1er
SLOW_LAST = None                          # si fixe : le ralenti final ne couvre que les N dernieres poses
CUSTOM_ST = None                          # fonction(r) -> (st, note_of) : calendrier des poses impose de l'exterieur
SPEEDS = None                             # vitesse propre a chaque algo (ex [1.6, 1.2, 1.0, 0.8]) : course asynchrone
MELODY_MODULE = "fr_melody"               # module qui fournit build(n) -> frames/midi/lh
MELODY_DROP = 11                         # duree de chute constante en mode melodie (images)
MAXFX = False                            # True : tous les effets a fond (onde de choc, etincelles, zoom, flash, confettis...)
COLOR_FLOW = False                       # True : degrade de teintes autour de la couleur de la strategie, selon l'ordre d'arrivee


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def mix(c, a, t=255):
    return tuple(int(v * (1 - a) + t * a) for v in c)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------- simulation ----------
def make_pieces(rng):
    sizes = np.arange(PSIZE[0], PSIZE[1])
    p = 1.0 / sizes ** PSIZE[2]
    p /= p.sum()
    pieces, area = [], 0
    while area < CAP * FILL_TARGET:
        w, h = int(rng.choice(sizes, p=p)), int(rng.choice(sizes, p=p))
        pieces.append((w, h))
        area += w * h
    return pieces


class Bin:
    def __init__(self):
        self.O = (~MASK).astype(np.int32)

    def valid(self, w, h):
        if w > W_BIN or h > H_BIN:
            return None
        I = np.zeros((H_BIN + 1, W_BIN + 1), np.int32)
        I[1:, 1:] = self.O.cumsum(0).cumsum(1)
        S = (I[h:, w:] - I[:H_BIN - h + 1, w:]
             - I[h:, :W_BIN - w + 1] + I[:H_BIN - h + 1, :W_BIN - w + 1])
        return np.nonzero(S == 0)

    def contact(self, ys, xs, w, h):
        P = np.ones((H_BIN + 2, W_BIN + 2), np.int32)
        P[1:-1, 1:-1] = self.O
        IP = np.zeros((H_BIN + 3, W_BIN + 3), np.int32)
        IP[1:, 1:] = P.cumsum(0).cumsum(1)

        def rs(r0, r1, c0, c1):
            return IP[r1, c1] - IP[r0, c1] - IP[r1, c0] + IP[r0, c0]

        return (rs(ys, ys + 1, xs + 1, xs + 1 + w) + rs(ys + 1 + h, ys + 2 + h, xs + 1, xs + 1 + w)
                + rs(ys + 1, ys + 1 + h, xs, xs + 1) + rs(ys + 1, ys + 1 + h, xs + 1 + w, xs + 2 + w))

    def put(self, x, y, w, h):
        self.O[y:y + h, x:x + w] = 1


_RANKS = {}
TETRIS_K = 4                              # tetris : nombre de colonnes essayees au hasard
LAZY_FRAC = 0.12                         # smart_lazy : part des emplacements examines


def rank_grid(mode):
    """Rang de chaque case : l'algo cherche a couvrir la case libre de plus petit rang.
    spiral = de l'exterieur vers le centre (en tournant) ; explosion = du centre vers l'exterieur ;
    columns = tour apres tour, de gauche a droite, de bas en haut."""
    key = (mode, H_BIN, W_BIN)
    if key in _RANKS:
        return _RANKS[key]
    H, W = H_BIN, W_BIN
    R = np.zeros((H, W), np.int64)
    if mode == "spiral":
        order, y0, x0, y1, x1 = [], 0, 0, H - 1, W - 1
        while y0 <= y1 and x0 <= x1:
            order += [(y0, x) for x in range(x0, x1 + 1)]
            order += [(y, x1) for y in range(y0 + 1, y1 + 1)]
            if y1 > y0:
                order += [(y1, x) for x in range(x1 - 1, x0 - 1, -1)]
            if x1 > x0:
                order += [(y, x0) for y in range(y1 - 1, y0, -1)]
            y0, x0, y1, x1 = y0 + 1, x0 + 1, y1 - 1, x1 - 1
        for i, (y, x) in enumerate(order):
            R[y, x] = i
    elif mode == "explosion":
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = (H - 1) / 2, (W - 1) / 2
        d = np.hypot((yy - cy), (xx - cx)) + 0.001 * np.arctan2(yy - cy, xx - cx)
        R[:] = np.argsort(np.argsort(d.ravel())).reshape(H, W)
    else:                                                 # columns
        yy, xx = np.mgrid[0:H, 0:W]
        R[:] = xx * H + (H - 1 - yy)
    _RANKS[key] = R
    return R


def place(b, w, h, mode, rng):
    orient = [(w, h)] if w == h else [(w, h), (h, w)]
    cands = []
    for ww, hh in orient:
        v = b.valid(ww, hh)
        if v is not None and len(v[0]):
            cands.append((v[0], v[1], ww, hh))
    if not cands:
        return None
    if mode == "tetris":                                   # tombe tout droit dans une colonne au hasard
        H, W = b.O.shape
        occ = b.O > 0
        topm = np.where(occ.any(0), occ.argmax(0), H)
        best = None
        for _ in range(60):                                # TETRIS_K essais au hasard, garde celui qui tombe le plus bas
            ww, hh = orient[int(rng.integers(0, len(orient)))]
            if ww > W:
                continue
            x = int(rng.integers(0, W - ww + 1))
            y = int(topm[x:x + ww].min()) - hh
            if y >= 0 and not occ[y:y + hh, x:x + ww].any():
                if best is None or y + hh > best[1] + best[3]:
                    best = (x, y, ww, hh)
                tries = getattr(place, "_k", 0) + 1
                place._k = tries
                if tries >= TETRIS_K:
                    break
        place._k = 0
        if best is None:
            return None
        b.put(*best)
        return best
    if mode in ("smart_lazy", "smart_norot"):             # smart fit handicape : vue partielle / sans rotation
        if mode == "smart_norot":
            cands = cands[:1]
        best, res = None, None
        for ys, xs, ww, hh in cands:
            if mode == "smart_lazy" and len(ys) > 8:
                keep = rng.random(len(ys)) < LAZY_FRAC
                if keep.sum() >= 1:
                    ys, xs = ys[keep], xs[keep]
            score = (ys + hh).astype(np.int64) * 100 + (W_BIN - xs) + b.contact(ys, xs, ww, hh).astype(np.int64) * 10000
            k = int(np.argmax(score))
            if best is None or score[k] > best:
                best, res = score[k], (int(xs[k]), int(ys[k]), ww, hh)
        b.put(*res)
        return res
    if mode in ("spiral", "explosion", "columns"):
        R = np.where(b.O > 0, 10 ** 9, rank_grid(mode))
        best, res = None, None
        for ys, xs, ww, hh in cands:
            mr = np.lib.stride_tricks.sliding_window_view(R, (hh, ww)).min(axis=(2, 3))[ys, xs]
            score = -mr * 1000 + (b.contact(ys, xs, ww, hh) if mode != "columns" else 0)
            k = int(np.argmax(score))
            if best is None or score[k] > best:
                best, res = score[k], (int(xs[k]), int(ys[k]), ww, hh)
        b.put(*res)
        return res
    if mode == "random":
        ys, xs, ww, hh = cands[int(rng.integers(len(cands)))]
        k = int(rng.integers(len(ys)))
        res = (int(xs[k]), int(ys[k]), ww, hh)
    else:
        best, res = None, None
        for ys, xs, ww, hh in cands:
            score = (ys + hh).astype(np.int64) * 100 + (xs if mode == "br" else W_BIN - xs)
            if mode == "smart":
                score = score + b.contact(ys, xs, ww, hh).astype(np.int64) * 10000
            k = int(np.argmax(score))
            if best is None or score[k] > best:
                best, res = score[k], (int(xs[k]), int(ys[k]), ww, hh)
    b.put(*res)
    return res


def order_for(pieces, kind):
    """Ordre de pose : False = ordre d'arrivee, True/'big' = plus grosses d'abord,
    'small' = plus petites d'abord, 'long' = plus longs cotes d'abord."""
    idx = list(range(len(pieces)))
    keys = {True: lambda i: -pieces[i][0] * pieces[i][1],
            "big": lambda i: -pieces[i][0] * pieces[i][1],
            "small": lambda i: pieces[i][0] * pieces[i][1],
            "long": lambda i: -max(pieces[i])}
    return idx if not kind else sorted(idx, key=keys[kind])


def simulate(seed):
    pieces = make_pieces(np.random.default_rng(seed))
    n = len(pieces)
    plans = []
    for si, (_, _, mode, sort_big) in enumerate(STRATS):
        rng = np.random.default_rng(seed * 10 + si)
        b = Bin()
        order = order_for(pieces, sort_big)
        plans.append([place(b, *pieces[i], mode, rng) for i in order])
    return pieces, plans


# ---------- dessin ----------
def draw_rect(img, x0, y0, x1, y1, col, white=0.0):
    x0, y0 = int(round(x0)) + 1, int(round(y0)) + 1
    x1, y1 = int(round(x1)) - 2, int(round(y1)) - 2
    if x1 <= x0 or y1 <= y0:
        return
    c = mix(col, white) if white > 0 else col
    cv2.rectangle(img, (x0, y0), (x1, y1), c, -1)
    if x1 - x0 > 8 and y1 - y0 > 8:                       # reflet neon : haut/gauche clairs, bas sombre
        hi, lo = mix(c, 0.35), mix(c, 0.35, 0)
        cv2.rectangle(img, (x0, y0), (x1, y0 + 2), hi, -1)
        cv2.rectangle(img, (x0, y0), (x0 + 1, y1), hi, -1)
        cv2.rectangle(img, (x0, y1 - 1), (x1, y1), lo, -1)


class Renderer:
    def __init__(self, seed):
        self.pieces, self.plans = simulate(seed)
        S, n = len(STRATS), len(self.plans[0])
        self.n = n
        self.maskpx = cv2.resize(MASK.astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST).astype(bool)
        self.full = bool(MASK.all())
        self.contours = cv2.findContours(self.maskpx.astype(np.uint8) * 255, cv2.RETR_LIST,
                                         cv2.CHAIN_APPROX_NONE)[-2]
        self.starts = np.array([PLACE_FRAMES * (i / n) ** 0.8 for i in range(n)])
        if MELODY:                                        # atterrissage exactement sur les notes
            import importlib
            self.mel = importlib.import_module(MELODY_MODULE).build(n)
            self.starts = self.mel["frames"] - MELODY_DROP
        self.st = np.stack([self.starts[0] + (self.starts - self.starts[0]) / (SPEEDS[k] if SPEEDS else 1.0)
                            for k in range(S)])
        self.note_of = {}
        if CUSTOM_ST:
            self.st, self.note_of = CUSTOM_ST(self)
        rng = np.random.default_rng(99)
        self.colors = [[tuple(int(c * rng.uniform(0.68, 1.0)) for c in st[1]) for _ in range(n)]
                       for st in STRATS]
        if COLOR_BY_SIZE:                                 # couleur = taille de la piece (meme palette partout)
            for s, plan in enumerate(self.plans):
                for j, r in enumerate(plan):
                    if r:
                        k = int(np.searchsorted(AREA_EDGES, r[2] * r[3], side="right"))
                        self.colors[s][j] = tuple(int(c * rng.uniform(0.82, 1.0)) for c in RAINBOW[k])
        if COLOR_FLOW:
            for s, plan in enumerate(self.plans):
                h0 = int(cv2.cvtColor(np.uint8([[STRATS[s][1]]]), cv2.COLOR_RGB2HSV)[0, 0, 0])
                for j in range(n):
                    h = (h0 + (j / n - 0.5) * 46) % 180
                    c = cv2.cvtColor(np.uint8([[[int(h), int(0.78 * 255), int(rng.uniform(0.9, 1.0) * 255)]]]),
                                     cv2.COLOR_HSV2RGB)[0, 0]
                    self.colors[s][j] = tuple(int(v) for v in c)
        self.pdata = np.zeros((S, n, NP, 4))
        self.pdata[..., 0] = rng.uniform(0.05, 0.95, (S, n, NP))
        self.pdata[..., 1] = rng.uniform(-2.4, 2.4, (S, n, NP))
        self.pdata[..., 2] = rng.uniform(-4.5, -1.2, (S, n, NP))
        self.pdata[..., 3] = rng.uniform(1.6, 3.4, (S, n, NP))
        self.D = np.zeros((S, n))
        self.drop = np.zeros((S, n))
        self.land = np.zeros((S, n))
        self.done = np.zeros((S, n))
        self.ev, self.lands_sorted, self.cum_sorted = [], [], []
        for s, plan in enumerate(self.plans):
            items = []
            for j, r in enumerate(plan):
                if r is None:
                    self.land[s, j] = self.done[s, j] = self.st[s, j]
                    continue
                x, y, w, h = r
                D = (y + h) * CELL
                self.D[s, j] = D
                self.drop[s, j] = MELODY_DROP if MELODY else 5 + 9 * D / ((H_BIN + 12) * CELL)
                self.land[s, j] = self.st[s, j] + self.drop[s, j]
                self.done[s, j] = self.land[s, j] + PL
                items.append((self.land[s, j], w * h, D))
            items.sort()
            lands = np.array([i[0] for i in items])
            cum = np.concatenate([[0], np.cumsum([i[1] for i in items])])
            self.lands_sorted.append(lands)
            self.cum_sorted.append(cum)
            self.ev.append((lands, cum[1:] / CAP, np.array([i[2] for i in items])))
        self.final = [c[-1] / CAP * 100 for c in self.cum_sorted]
        order = list(np.argsort(self.final)[::-1])
        self.winner = int(order[0])
        self.top2 = order[:2]
        self.gap = self.final[order[0]] - self.final[order[1]]
        self.photo = self.gap < PHOTO_GAP

        # chronologie (en images d'animation)
        placed = np.array([[r is not None for r in self.plans[k]] for k in range(S)])
        if placed.any():                                        # la fin arrive des la derniere piece VRAIMENT posee
            self.END = float(self.done[placed].max()) + 4
        else:
            self.END = float(self.done.max()) + 4
        self.hold_end = self.END + HOLD_FRAMES
        self.T = self.hold_end + RESET_FRAMES + 2
        ps = np.sort(self.st[placed]) if placed.any() else np.sort(self.st.ravel())
        self.f_slow = float(ps[max(0, len(ps) - (SLOW_LAST or SLOW_K * S))])
        self.finish = np.array([self.done[k][placed[k]].max() if placed[k].any() else self.done[k].max() for k in range(S)])
        self.rank = {int(k): i + 1 for i, k in enumerate(np.argsort(self.finish))}
        self.f_slow_end = float(self.land[placed].max() if placed.any() else self.land.max()) + 2
        # ralenti : relation temps d'animation <-> image de sortie
        self.tau_grid = np.linspace(0, self.T, int(self.T / 0.02) + 1)
        if self.photo:
            w = smooth((self.tau_grid - self.f_slow) / 6) * (1 - smooth((self.tau_grid - self.f_slow_end) / 6))
        else:
            w = np.zeros_like(self.tau_grid)
        rate = 1 - (1 - SLOW_RATE) * w
        self.Fcum = np.concatenate([[0], np.cumsum(np.diff(self.tau_grid) / rate[:-1])])
        self.total_out = int(np.floor(self.Fcum[-1]))

        if MAXFX:                                         # donnees des effets
            amax = max(r[2] * r[3] for p in self.plans for r in p if r)
            self.wt = np.array([[(r[2] * r[3] / amax) ** 0.5 if r else 0.0 for r in p] for p in self.plans])
            self.xang = rng.uniform(-np.pi * 0.95, -np.pi * 0.05, (S, n, 12))
            self.xspd = rng.uniform(3, 9, (S, n, 12))
            self.cang = rng.uniform(0, 2 * np.pi, 90)
            self.cspd = rng.uniform(4, 13, 90)
            self.cdel = rng.integers(0, 24, 90)
            self.ccol = [tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(rng.uniform(0, 180)), 120, 255]]]),
                                                          cv2.COLOR_HSV2RGB)[0, 0]) for _ in range(90)]
            self._shake = np.zeros(2)
        self.static = [self.panel_base() for _ in STRATS]
        self.baked = [0] * S
        self.f_title, self.f_sub = font(70), font(32)
        self.f_head, self.f_pct, self.f_bar = font(32), font(32), font(26)
        self.bg = self.make_bg()

    def panel_base(self):
        """Panneau vide : cases utilisables en EMPTY, le reste (hors forme, obstacles) en fond."""
        base = np.full((PH, PW, 3), BG, np.uint8)
        base[self.maskpx] = EMPTY
        return base

    def outline(self, img, col, thick, off=(0, 0)):
        if self.full:
            cv2.rectangle(img, (off[0], off[1]), (off[0] + PW - 1, off[1] + PH - 1), col, thick)
        else:
            cv2.drawContours(img, self.contours, -1, col, thick, offset=off)

    # temps
    def tau_of(self, F):
        return float(np.interp(F, self.Fcum, self.tau_grid))

    def tf(self, tau):
        return float(np.interp(tau, self.tau_grid, self.Fcum)) / FPS

    def make_bg(self):
        img = Image.new("RGB", (VW, VH), BG)
        d = ImageDraw.Draw(img)
        size = 70
        ft = self.f_title
        while size > 40 and d.textlength(TITLE, font=ft) > 940:     # titre long : on reduit la police
            size -= 4
            ft = font(size)
        d.text((VW // 2, 110), TITLE, font=ft, fill=(240, 240, 245), anchor="mm")
        for s, (name, col, _, _) in enumerate(STRATS):
            if MYSTERY:                                   # etiquettes dessinees a part (compose)
                continue
            d.text((PX[s % 2], PY[s // 2] - 30), name, font=self.f_head, fill=col, anchor="lm")
            if not LIVE_RANK:
                d.text((90, BAR_Y0 + s * BAR_STEP + BAR_H // 2), name, font=self.f_bar, fill=col, anchor="lm")
        arr = np.array(img)
        glow = np.zeros_like(arr)
        for s, (_, col, _, _) in enumerate(STRATS):
            x0, y0 = PX[s % 2], PY[s // 2]
            if self.full:
                cv2.rectangle(glow, (x0, y0), (x0 + PW, y0 + PH), col, 10)
            else:
                cv2.drawContours(glow, self.contours, -1, col, 10, offset=(x0, y0))
        glow = cv2.GaussianBlur(glow, (0, 0), 22)
        return cv2.add(arr, (glow * 0.45).astype(np.uint8))

    def fill_at(self, s, tau):
        k = int(np.searchsorted(self.lands_sorted[s], tau, side="right"))
        return self.cum_sorted[s][k] / CAP * 100

    def rank_rows(self, tau, fills, rp=0.0):
        """Rang en direct (0 = 1er) avec glissement doux ; retient le leader et l'instant ou il a change."""
        S = len(STRATS)
        if not hasattr(self, "_rowy") or tau < getattr(self, "_last_tau", -1) or tau == 0:
            self._rowy = np.arange(S, dtype=float)
            self.leader, self.lead_t = 0, -99.0
        order = sorted(range(S), key=lambda k: (-round(fills[k], 1), k))
        target = np.empty(S)
        for rk, k in enumerate(order):
            target[k] = rk
        if rp > 0:                                        # remise a zero : retour a l'ordre du debut (boucle parfaite)
            target = np.arange(S, dtype=float)
            order = list(range(S))
        self._rowy += (target - self._rowy) * 0.22
        if rp == 0 and max(fills) > 0.5 and order[0] != self.leader:
            self.leader, self.lead_t = order[0], tau
        self._last_tau = tau
        return self._rowy

    @staticmethod
    def crown(d, cx, cy, r):
        """Petite couronne doree centree en (cx, cy), demi-largeur r."""
        gold, dark = (255, 205, 60), (190, 130, 20)
        pts = [(cx - r, cy + r * 0.55), (cx - r, cy - r * 0.35), (cx - r * 0.5, cy + r * 0.05), (cx, cy - r * 0.75),
               (cx + r * 0.5, cy + r * 0.05), (cx + r, cy - r * 0.35), (cx + r, cy + r * 0.55)]
        d.polygon(pts, fill=gold, outline=dark)
        for x, y in ((cx - r, cy - r * 0.35), (cx, cy - r * 0.75), (cx + r, cy - r * 0.35)):
            d.ellipse((x - r * 0.18, y - r * 0.18, x + r * 0.18, y + r * 0.18), fill=(255, 240, 160))

    def fall_t0(self, s, j):
        """Instant (image) ou la piece j d'une boite perdante commence a tomber : du bas vers le haut, en cascade."""
        x, y, w, h = self.plans[s][j]
        k = 1 - (y + h) / H_BIN
        return self.END + 12 + 34 * k + 6 * ((j * 37 + s * 11) % 10) / 10

    def zoom_at(self, tau):
        if not self.photo:
            return 1.0
        return 1 + 0.06 * float(smooth((tau - self.f_slow) / 10)) * (1 - float(smooth((tau - self.END) / 15)))

    # un panneau
    def panel_img(self, s, tau, rp):
        plan, n = self.plans[s], self.n
        if ENDING == "fall" and s != self.winner and tau > self.END + 10:
            panel = self.panel_base()
            if rp > 0:
                return panel
            for j, r in enumerate(plan):
                if not r:
                    continue
                x, y, w, h = r
                t0 = self.fall_t0(s, j)
                a = max(0.0, tau - t0)
                top = y * CELL + 0.9 * a * a
                if top > PH:
                    continue
                draw_rect(panel, x * CELL, top, (x + w) * CELL, top + h * CELL,
                          self.colors[s][j] if a == 0 else mix(self.colors[s][j], min(0.6, a / 40), 0))
            return panel
        if rp > 0:                                        # remise a zero : les pieces remontent
            panel = self.panel_base()
            for j, r in enumerate(plan):
                if not r:
                    continue
                x, y, w, h = r
                t0 = self.hold_end + (n - 1 - j) / n * RESET_FRAMES * 0.5
                v = float(np.clip((tau - t0) / (RESET_FRAMES * 0.5), 0, 1))
                if v >= 1:
                    continue
                top = y * CELL - self.D[s, j] * v * v
                draw_rect(panel, x * CELL, top, (x + w) * CELL, top + h * CELL, self.colors[s][j])
            return panel
        p = self.baked[s]
        while p < n and self.done[s, p] <= tau:
            r = plan[p]
            if r:
                x, y, w, h = r
                draw_rect(self.static[s], x * CELL, y * CELL, (x + w) * CELL, (y + h) * CELL, self.colors[s][p])
            p += 1
        self.baked[s] = p
        panel = self.static[s].copy()
        hi = int(np.searchsorted(self.st[s], tau, side="right"))
        col = STRATS[s][1]
        for j in range(p, hi):
            r = plan[j]
            if not r:
                continue
            x, y, w, h = r
            c = self.colors[s][j]
            tx0, ty0, tx1, ty1 = x * CELL, y * CELL, (x + w) * CELL, (y + h) * CELL
            cx, hw, ph = (tx0 + tx1) / 2, (tx1 - tx0) / 2, ty1 - ty0
            a, dr = tau - self.st[s, j], self.drop[s, j]
            if a < dr:                                    # chute
                u = a / dr
                sx, sy = 1 - 0.03 * u, 1 + 0.08 * u
                bottom = ty1 - self.D[s, j] * (1 - u * u)
                draw_rect(panel, cx - hw * sx, bottom - ph * sy, cx + hw * sx, bottom, c)
                continue
            b = a - dr                                    # temps depuis l'impact
            if b < SQ:                                    # ecrasement + flash
                t = b / SQ
                k = (1 - t) ** 2
                sx, sy = 1 + 0.06 * k, 1 - 0.14 * k
                draw_rect(panel, cx - hw * sx, ty1 - ph * sy, cx + hw * sx, ty1, c, white=0.55 * (1 - t))
            else:
                draw_rect(panel, tx0, ty0, tx1, ty1, c)
            if b < PL:                                    # etincelles
                spark = mix(col, 0.55)
                for k in range(NP):
                    fx, vx, vy, r0 = self.pdata[s, j, k]
                    rad = r0 * (1 - b / PL)
                    if rad >= 0.8:
                        cv2.circle(panel, (int(tx0 + fx * (tx1 - tx0) + vx * b),
                                           int(ty1 + vy * b + 0.175 * b * b)), int(round(rad)), spark, -1)
        return panel

    def pulse_at(self, s, tau):
        m = (self.land[s] <= tau) & (self.land[s] > tau - 20)
        return float((self.wt[s][m] * np.exp(-(tau - self.land[s][m]) / 6.0)).sum())

    def make_fx(self, tau):
        """Couche d'effets plein ecran : onde de choc + etincelles a chaque impact, confettis a la fin."""
        fx = np.zeros((VH, VW, 3), np.uint8)
        shake = np.zeros(2)
        for s in range(len(STRATS)):
            idx = np.where((self.land[s] <= tau) & (self.land[s] > tau - 20))[0]
            for j in idx:
                r = self.plans[s][j]
                if not r:
                    continue
                x, y, w, h = r
                age = tau - self.land[s, j]
                gx = PX[s % 2] + (x + w / 2) * CELL
                gy = PY[s // 2] + (y + h) * CELL
                col = self.colors[s][j]
                wt = self.wt[s, j]
                if age < 14:
                    cv2.circle(fx, (int(gx), int(gy - h * CELL / 2)), int(8 + age * (3 + 7 * wt)),
                               tuple(int(c * (1 - age / 14) * 0.85) for c in col), 2)
                for k in range(12):
                    a, sp = self.xang[s, j, k], self.xspd[s, j, k]
                    px = gx + np.cos(a) * sp * age
                    py = gy + np.sin(a) * sp * age + 0.4 * age * age
                    cv2.circle(fx, (int(px), int(py)), max(1, int(3.5 * (1 - age / 20))),
                               tuple(int(c * (1 - age / 22)) for c in mix(col, 0.45)), -1)
                if age < 5:
                    ang = (j * 2.399 + s) % (2 * np.pi)
                    shake += wt * np.exp(-age / 2.0) * 1.6 * np.array([np.cos(ang), np.sin(ang)])
        self._shake = np.clip(shake, -6, 6)
        if tau >= self.END:                               # confettis depuis le panneau gagnant
            wn = self.winner
            ox, oy = PX[wn % 2] + PW / 2, PY[wn // 2] + PH / 2
            for i in range(90):
                ta = tau - self.END - self.cdel[i]
                if ta < 0 or ta > 70:
                    continue
                px = ox + np.cos(self.cang[i]) * self.cspd[i] * ta * 0.9
                py = oy + np.sin(self.cang[i]) * self.cspd[i] * ta * 0.9 + 0.1 * ta * ta
                if 0 <= px < VW and 0 <= py < VH:
                    tw = 0.5 + 0.5 * np.sin(ta * 0.6 + i)
                    cv2.circle(fx, (int(px), int(py)), 2 + int(3 * tw),
                               tuple(int(c * (1 - ta / 72)) for c in self.ccol[i]), -1)
        return fx

    def compose(self, tau, F):
        rp = float(np.clip((tau - self.hold_end) / RESET_FRAMES, 0, 1))
        end_raw = float(np.clip((tau - self.END) / 15, 0, 1))
        end_t = end_raw * (1 - rp)
        frame = self.bg.copy()
        fills = []
        for s, (_, col, _, _) in enumerate(STRATS):
            panel = self.panel_img(s, tau, rp)
            if end_t > 0 and s != self.winner:
                panel = (panel * (1 - 0.55 * end_t)).astype(np.uint8)
            thick = 2 + (int(5 * end_t) if s == self.winner else 0)
            self.outline(panel, mix(col, min(0.7, 0.9 * self.pulse_at(s, tau))) if MAXFX else col, thick)
            x0, y0 = PX[s % 2], PY[s // 2]
            frame[y0:y0 + PH, x0:x0 + PW] = panel
            fill = self.fill_at(s, tau) * (1 - rp)
            fills.append(fill)
            by = BAR_Y0 + s * BAR_STEP
            if LIVE_RANK:
                continue
            cv2.rectangle(frame, (BAR_X0, by), (BAR_X1, by + BAR_H), EMPTY, -1)
            wbar = int((BAR_X1 - BAR_X0) * fill / 100)
            if wbar > 0:
                cv2.rectangle(frame, (BAR_X0, by), (BAR_X0 + wbar, by + BAR_H), col, -1)
        if LIVE_RANK:                                     # barres triees, positions qui glissent en douceur
            rowy = self.rank_rows(tau, fills, rp)
            for s, (_, col, _, _) in enumerate(STRATS):
                by = int(round(BAR_Y0 + rowy[s] * BAR_STEP))
                cv2.rectangle(frame, (BAR_X0, by), (BAR_X1, by + BAR_H), EMPTY, -1)
                wbar = int((BAR_X1 - BAR_X0) * fills[s] / 100)
                if wbar > 0:
                    cv2.rectangle(frame, (BAR_X0, by), (BAR_X0 + wbar, by + BAR_H), col, -1)
        if end_t > 0:                                     # halo qui pulse autour du gagnant
            pad, wn = 60, self.winner
            gx, gy = PX[wn % 2] - pad, PY[wn // 2] - pad
            layer = np.zeros((PH + 2 * pad, PW + 2 * pad, 3), np.uint8)
            if self.full:
                cv2.rectangle(layer, (pad, pad), (pad + PW, pad + PH), STRATS[wn][1], 14)
            else:
                cv2.drawContours(layer, self.contours, -1, STRATS[wn][1], 14, offset=(pad, pad))
            layer = cv2.GaussianBlur(layer, (0, 0), 18)
            layer[pad:pad + PH, pad:pad + PW][self.maskpx] = 0                 # halo seulement autour du panneau
            k = end_t * (0.55 + 0.35 * np.sin(F * 0.22))
            reg = frame[gy:gy + layer.shape[0], gx:gx + layer.shape[1]]
            frame[gy:gy + layer.shape[0], gx:gx + layer.shape[1]] = cv2.add(reg, (layer * k).astype(np.uint8))

        if MAXFX:
            frame = cv2.add(frame, self.make_fx(tau))
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        for s in range(len(STRATS)):
            pct = f"{fills[s]:.1f}%"
            d.text((PX[s % 2] + PW, PY[s // 2] - 30), pct, font=self.f_pct, fill=(235, 235, 240), anchor="rm")
            by_txt = BAR_Y0 + (self._rowy[s] if LIVE_RANK else s) * BAR_STEP + BAR_H // 2
            d.text((BAR_X1 + 15, by_txt), pct, font=self.f_bar, fill=(200, 200, 210), anchor="lm")
            if LIVE_RANK:
                d.text((90, by_txt), STRATS[s][0], font=self.f_bar, fill=STRATS[s][1], anchor="lm")
        if LIVE_RANK and rp == 0 and max(fills) > 0.5:     # couronne sur le leader (liste + titre de sa boite)
            ld = self.leader
            pop = 1 + 0.5 * max(0.0, 1 - (tau - self.lead_t) / 8)
            self.crown(d, 62, BAR_Y0 + self._rowy[ld] * BAR_STEP + BAR_H // 2, 13 * pop)
            nx = PX[ld % 2] + d.textlength(STRATS[ld][0], font=self.f_head) + 26
            self.crown(d, nx, PY[ld // 2] - 32, 17 * pop)
            if SPEEDS and rp == 0 and tau >= self.finish[s]:      # badge d'arrivee : 1st / 2nd / 3rd / 4th DONE
                rk = self.rank[s]
                lab = {1: "1ST DONE", 2: "2ND DONE", 3: "3RD DONE", 4: "4TH DONE"}[rk]
                k = min(1.0, (tau - self.finish[s]) / 6)
                c = tuple(int(b * (1 - k) + v * k) for b, v in zip(BG, (255, 214, 90) if rk == 1 else (215, 215, 225)))
                d.text((PX[s % 2] + PW // 2, PY[s // 2] + PH + 26), lab, font=self.f_bar, fill=c, anchor="mm")
        if MYSTERY:                                       # lettres A-D qui laissent place aux noms a la fin
            k = float(smooth(end_t * 1.5))
            for s, (name, col, _, _) in enumerate(STRATS):
                for txt, a in (("ABCD"[s], 1 - k), (name, k)):
                    if a <= 0.02:
                        continue
                    c = tuple(int(b * (1 - a) + v * a) for b, v in zip(BG, col))
                    d.text((PX[s % 2], PY[s // 2] - 30), txt, font=self.f_head, fill=c, anchor="lm")
                    d.text((90, BAR_Y0 + s * BAR_STEP + BAR_H // 2), txt, font=self.f_bar, fill=c, anchor="lm")
        wname, wcol = STRATS[self.winner][0], STRATS[self.winner][1]
        pos = (VW // 2, 185)
        if rp > 0:
            a_win, a_sub = 1 - min(1.0, rp * 2), max(0.0, rp * 2 - 1)
            if a_win > 0:
                d.text(pos, f"WINNER: {wname}", font=self.f_sub,
                       fill=tuple(int(b * (1 - a_win) + c * a_win) for b, c in zip(BG, wcol)), anchor="mm")
            if a_sub > 0:
                d.text(pos, SUB, font=self.f_sub,
                       fill=tuple(int(b * (1 - a_sub) + c * a_sub) for b, c in zip(BG, GREY)), anchor="mm")
        elif end_raw > 0:
            c = tuple(int(g * (1 - end_raw) + w * end_raw) for g, w in zip(GREY, wcol))
            d.text(pos, f"WINNER: {wname}", font=self.f_sub, fill=c, anchor="mm")
        elif self.photo and self.f_slow <= tau < self.END:
            t = 0.5 + 0.5 * np.sin(F * 0.3)
            d.text(pos, "PHOTO FINISH!", font=self.f_sub,
                   fill=tuple(int(g * (1 - t) + 255 * t) for g in GREY), anchor="mm")
        else:
            d.text(pos, SUB, font=self.f_sub, fill=GREY, anchor="mm")
        arr = np.array(img)
        if MAXFX and tau >= self.END:                     # flash blanc + aberration chromatique a la victoire
            ff = np.exp(-(tau - self.END) / 7.0)
            if ff > 0.02:
                arr = cv2.add(arr, np.full_like(arr, int(60 * ff)))
                sft = int(6 * ff)
                if sft:
                    arr[..., 0] = np.roll(arr[..., 0], sft, axis=1)
                    arr[..., 2] = np.roll(arr[..., 2], -sft, axis=1)
        if BLOOM:                                         # halo : on ne floute que les zones claires
            small = cv2.resize(arr, (VW // 4, VH // 4), interpolation=cv2.INTER_AREA)
            br = np.clip(small.astype(np.int16) - (60 if MAXFX else 70), 0, 255).astype(np.uint8)
            glow = cv2.resize(cv2.GaussianBlur(br, (0, 0), 6), (VW, VH), interpolation=cv2.INTER_LINEAR)
            arr = cv2.add(arr, (glow * (1.5 if MAXFX else 1.1)).clip(0, 255).astype(np.uint8))
        z = self.zoom_at(tau)
        sh = np.zeros(2)
        if MAXFX:                                         # zoom qui monte + coup de zoom a la victoire + tremblement
            push = 0.045 * float(smooth(tau / self.END)) * (1 - float(smooth((tau - self.hold_end) / RESET_FRAMES)))
            punch = 0.03 * np.exp(-(tau - self.END) / 9.0) if tau >= self.END else 0.0
            z = min(z * (1 + push + punch), 1.07)         # plafond : le titre reste entier a l'ecran
            sh = self._shake * (1 - rp)
        if z > 1.0005 or abs(sh[0]) + abs(sh[1]) > 0.2:
            if MAXFX and not self.photo:
                cx, cy = VW / 2, VH / 2
            else:                                         # zoom doux entre les deux leaders
                cs = [(PX[s % 2] + PW / 2, PY[s // 2] + PH / 2) for s in self.top2]
                cx = 0.5 * sum(c[0] for c in cs) / 2 + 0.5 * VW / 2      # centre tire vers le milieu de l'ecran
                cy = 0.5 * sum(c[1] for c in cs) / 2 + 0.5 * VH / 2
            M = np.array([[z, 0, cx * (1 - z) + sh[0]], [0, z, cy * (1 - z) + sh[1]]], np.float32)
            arr = cv2.warpAffine(arr, M, (VW, VH), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
        return arr


# ---------- son ----------
def synth_musical(r, sr=44100):
    import fr_music as fm
    rng = np.random.default_rng(5)
    n = int((r.total_out / FPS + 1.5) * sr)
    a = np.zeros(n, np.float32)

    def add(sig, t0, gain=1.0):
        st = int(t0 * sr)
        if 0 <= st < n:
            seg = (sig * gain).astype(np.float32)
            a[st:st + len(seg)] += seg[:n - st]

    tt = np.arange(int(0.12 * sr)) / sr
    tick = np.sin(2 * np.pi * (70 * tt + 90 * (1 - np.exp(-45 * tt)) / 45)) * np.exp(-tt * 40)
    dens = min(1.0, (90 / r.n) ** 0.5)
    tx = np.arange(int(0.4 * sr)) / sr
    thump_x = np.sin(2 * np.pi * (48 * tx + 60 * (1 - np.exp(-30 * tx)) / 30)) * np.exp(-tx * 12)
    cache = {}
    t_end, t_hold = r.tf(r.END), r.tf(r.hold_end)
    for s in range(len(STRATS)):
        lands, frac, Ds = r.ev[s]
        prev = 0.0
        for m in range(len(lands)):
            t0 = r.tf(lands[m])
            area = max(1.0, (frac[m] - prev) * CAP)
            prev = frac[m]
            if MELODY:                                    # la melodie est jouee a part : ici, juste le contact
                add(tick, t0, 0.03 * dens)
                if area >= 60:
                    add(thump_x, t0, 0.05 * dens)
                continue
            if MAXFX:                                     # la melodie monte avec le remplissage : le leader joue plus haut
                k = int(min(1.0, frac[m] / 0.95) * 16)
                for kx, g in ((k, 0.085), (k + 5, 0.03)):
                    if kx not in cache:
                        cache[kx] = fm.pluck(fm.ladder_freq(kx), sr, 1.2)
                    add(cache[kx], t0, g * dens)
                if area >= 60:
                    add(thump_x, t0, 0.07 * dens)
                continue
            k = fm.area_to_k(area)
            if k not in cache:
                cache[k] = fm.pluck(fm.ladder_freq(k), sr)
            add(tick, t0, 0.05 * dens)
            add(cache[k], t0, 0.10 * dens * (0.8 + 0.2 * (s % 2)))
        for thr, kk in (() if MELODY else ((0.90, 15), (0.95, 17))):
            j = int(np.searchsorted(frac, thr, side="left"))
            if j < len(lands):
                add(fm.bell(fm.ladder_freq(kk), sr), r.tf(lands[j]), 0.08)
    if MELODY:                                            # voix principale : une note par piece posee
        mel, pc = r.mel, {}
        for j in range(r.n):
            if not any(r.plans[s][j] for s in range(len(STRATS))):
                continue
            m = mel["midi"][j]
            if m not in pc:
                pc[m] = fm.pluck(fm.midi_freq(m) if hasattr(fm, "midi_freq") else 440.0, sr, 1.5, 1.2)
            add(pc[m], r.tf(r.starts[j] + MELODY_DROP), 0.30)
        for f, m in mel["lh"]:
            if m not in pc:
                pc[m] = fm.pluck(fm.midi_freq(m), sr, 1.4, 0.8)
            add(pc[m], r.tf(f), 0.17)
    if r.photo:
        th = np.arange(int(0.22 * sr)) / sr
        beat = np.sin(2 * np.pi * 55 * th) * np.exp(-th * 22)
        t = r.tf(r.f_slow)
        while t < t_end:
            add(beat, t, 0.12)
            add(beat, t + 0.2, 0.08)
            t += 0.62
    if MAXFX:                                             # montee, explosion, accord, scintillements
        rd = 2.2
        tw = np.arange(int(rd * sr)) / sr
        env = (tw / rd) ** 2.2
        ph = 2 * np.pi * np.cumsum(180 * (1 + 9 * (tw / rd) ** 2)) / sr
        nz = np.convolve(rng.standard_normal(len(tw)), np.ones(10) / 10, mode="same")
        add((np.sin(ph) * 0.8 + nz * 0.6) * env, t_end - rd, 0.2)
        tb = np.arange(int(2.2 * sr)) / sr
        add(np.sin(2 * np.pi * (46 * tb + 90 * (1 - np.exp(-6 * tb)) / 6)) * np.exp(-tb * 1.6), t_end, 0.5)
        for i, k in enumerate((5, 7, 9, 12, 14, 16)):
            add(fm.bell(fm.ladder_freq(k), sr, 3.2), t_end + 0.035 * i, 0.13)
        for _ in range(24):
            add(fm.bell(fm.ladder_freq(int(rng.integers(14, 20))), sr, 1.2), t_end + 0.2 + rng.uniform(0, 2.4), 0.04)
    else:
        for i, k in enumerate((5, 7, 9, 10, 12, 14)):           # arpege du gagnant
            add(fm.bell(fm.ladder_freq(k), sr, 2.2), t_end + 0.09 * i, 0.12)
    dur = r.total_out / FPS + 1.5
    lv = [0.05, 0.11, 0.17, 0.12, 0.0] if MAXFX else [0.03, 0.05, 0.08, 0.08, 0.0]
    a += fm.pad(sr, dur, [0, t_end * 0.5, t_end, t_hold, dur], lv) * (0.75 if MAXFX else 0.6)
    dr = RESET_FRAMES / FPS
    tw = np.arange(int(dr * sr)) / sr
    win = np.sin(np.pi * tw / dr) ** 2
    noise = np.convolve(rng.standard_normal(len(tw)), np.ones(24) / 24, mode="same")
    add(noise * win, t_hold, 0.12)
    a = fm.reverb(a, sr, 2.0, 0.36) if MAXFX else fm.reverb(a, sr)
    if MAXFX:
        a = np.tanh(a * 1.8 / max(1e-6, float(np.abs(a).max())))
    a *= 0.88 / max(1e-6, float(np.abs(a).max()))
    fade = int(0.05 * sr)
    a[-fade:] *= np.linspace(1, 0, fade)
    return (a * 32767).astype(np.int16), sr


def synth_bricks(r, sr=44100):
    """Bruitages de briques : un 'tock' plastique par piece (petite = aigu et sec, grosse = grave et rond),
    thump pour les grosses, claquement final, montee + boom avant le resultat. Aucune note de musique."""
    from scipy.signal import butter, sosfilt
    import fr_music as fm
    rng = np.random.default_rng(5)
    n = int((r.total_out / FPS + 1.5) * sr)
    a = np.zeros(n, np.float32)

    def add(sig, t0, gain=1.0):
        st = int(t0 * sr)
        if 0 <= st < n:
            seg = (sig * gain).astype(np.float32)
            a[st:st + len(seg)] += seg[:n - st]

    sos = butter(2, [1800, 7000], btype="bandpass", fs=sr, output="sos")
    cache = {}

    def brick(area, var):
        key = (int(np.clip(round(np.log2(max(area, 3)) * 2), 0, 16)), var)
        if key in cache:
            return cache[key]
        f = float(np.clip(2300 * (max(area, 3) / 6.0) ** -0.40, 230, 3000)) * (1 + 0.05 * (var - 1))
        t = np.arange(int(0.22 * sr)) / sr
        dec = float(np.clip(70 - 8 * np.log2(max(area, 3)), 28, 62))
        body = np.sin(2 * np.pi * f * t) * np.exp(-t * dec) + 0.45 * np.sin(2 * np.pi * f * 2.41 * t) * np.exp(-t * dec * 1.7)
        click = sosfilt(sos, np.random.default_rng(key[0] * 7 + var).standard_normal(len(t))) * np.exp(-t * 520)
        sig = body * 1.0 + click * 0.6
        if area >= 40:                                    # grosse brique : petit thump grave
            sig = sig + np.sin(2 * np.pi * (75 * t + 55 * (1 - np.exp(-30 * t)) / 30)) * np.exp(-t * 16) * min(1.2, area / 90)
        sig = sig * (1 - np.exp(-t * 4000))
        cache[key] = sig.astype(np.float32)
        return cache[key]

    dens = min(1.0, (90 / r.n) ** 0.5)
    t_end, t_hold = r.tf(r.END), r.tf(r.hold_end)
    for s_i in range(len(STRATS)):
        lands, frac, Ds = r.ev[s_i]
        prev = 0.0
        for m in range(len(lands)):
            t0 = r.tf(lands[m])
            area = max(1.0, (frac[m] - prev) * CAP)
            prev = frac[m]
            vel = 0.55 + 0.45 * min(1.0, Ds[m] / ((H_BIN + 12) * CELL))
            add(brick(area, int(rng.integers(0, 3))), t0, 0.24 * vel * dens)
    if r.photo:
        th = np.arange(int(0.22 * sr)) / sr
        beat = np.sin(2 * np.pi * 55 * th) * np.exp(-th * 22)
        t = r.tf(r.f_slow)
        while t < t_end:
            add(beat, t, 0.14)
            add(beat, t + 0.2, 0.09)
            t += 0.62
    rd = 2.2                                              # montee de tension (bruit + sifflement), puis boom
    tw = np.arange(int(rd * sr)) / sr
    env = (tw / rd) ** 2.2
    ph = 2 * np.pi * np.cumsum(180 * (1 + 9 * (tw / rd) ** 2)) / sr
    nz = np.convolve(rng.standard_normal(len(tw)), np.ones(10) / 10, mode="same")
    add((np.sin(ph) * 0.6 + nz * 0.7) * env, t_end - rd, 0.13)
    tb = np.arange(int(2.0 * sr)) / sr
    add(np.sin(2 * np.pi * (46 * tb + 90 * (1 - np.exp(-6 * tb)) / 6)) * np.exp(-tb * 1.8), t_end, 0.4)
    for i in range(36):                                   # pluie de briques qui retombent apres l'explosion
        tt0 = t_end + 0.04 + (i / 36) ** 1.5 * 1.6 + rng.uniform(0, 0.03)
        add(brick(float(rng.choice([4, 6, 9, 16, 30, 60])), int(rng.integers(0, 3))), tt0, 0.22 * (1 - 0.6 * i / 36))
    dr = RESET_FRAMES / FPS
    tw = np.arange(int(dr * sr)) / sr
    win = np.sin(np.pi * tw / dr) ** 2
    noise = np.convolve(rng.standard_normal(len(tw)), np.ones(24) / 24, mode="same")
    add(noise * win, t_hold, 0.14)
    a = fm.reverb(a, sr, 0.7, 0.16)
    a = np.tanh(a * 3.2 / max(1e-6, float(np.abs(a).max())))
    a *= 0.88 / max(1e-6, float(np.abs(a).max()))
    fade = int(0.05 * sr)
    a[-fade:] *= np.linspace(1, 0, fade)
    return (a * 32767).astype(np.int16), sr


def synth(r, sr=44100):
    if BRICKS:
        return synth_bricks(r, sr)
    if MUSICAL:
        return synth_musical(r, sr)
    rng = np.random.default_rng(5)
    n = int((r.total_out / FPS + 1.5) * sr)
    a = np.zeros(n, np.float32)

    def add(sig, t0, gain=1.0):
        st = int(t0 * sr)
        if 0 <= st < n:
            seg = (sig * gain).astype(np.float32)
            a[st:st + len(seg)] += seg[:n - st]

    # clac sourd a l'impact
    tt = np.arange(int(0.18 * sr)) / sr
    thunk = np.sin(2 * np.pi * (55 * tt + 130 * (1 - np.exp(-45 * tt)) / 45)) * np.exp(-tt * 28)
    thunk = thunk + rng.standard_normal(len(tt)) * np.exp(-tt * 900) * 0.5
    # note de pluck, pitch qui monte avec le remplissage
    t3 = np.arange(int(0.3 * sr)) / sr
    env = np.exp(-t3 * 34) * (1 - np.exp(-t3 * 1200))
    scale = [0, 2, 4, 7, 9]
    # cloche (ding)
    t12 = np.arange(int(1.3 * sr)) / sr

    def bell(f0):
        return sum(am * np.sin(2 * np.pi * f0 * ra * t12) * np.exp(-t12 * dc)
                   for ra, am, dc in ((1, 1.0, 3.0), (2.76, 0.5, 5.0), (5.4, 0.25, 8.0)))

    dens = min(1.0, (90 / r.n) ** 0.5)                    # beaucoup de petites pieces : on baisse le volume
    for s in range(len(STRATS)):
        lands, frac, Ds = r.ev[s]
        for m in range(len(lands)):
            t0 = r.tf(lands[m])
            vol = (0.3 + 0.5 * min(1.0, Ds[m] / ((H_BIN + 12) * CELL))) * dens
            add(thunk, t0, 0.10 * vol)
            idx = int(frac[m] * 11)
            freq = 196.0 * 2 ** ((scale[idx % 5] + 12 * (idx // 5)) / 12) * (1 + 0.003 * s)
            add((np.sin(2 * np.pi * freq * t3) + 0.3 * np.sin(4 * np.pi * freq * t3)) * env, t0, 0.04 * dens)
        for thr, mult in ((0.90, 1.0), (0.95, 1.5)):
            k = int(np.searchsorted(frac, thr, side="left"))
            if k < len(lands):
                add(bell([784, 880, 988, 1047][s] * mult), r.tf(lands[k]), 0.07)
    # battements de coeur pendant le photo-finish
    if r.photo:
        th = np.arange(int(0.22 * sr)) / sr
        beat = np.sin(2 * np.pi * 55 * th) * np.exp(-th * 22)
        t = r.tf(r.f_slow)
        while t < r.tf(r.END):
            add(beat, t, 0.14)
            add(beat, t + 0.2, 0.09)
            t += 0.62
    # accord du gagnant
    t2 = np.arange(int(1.0 * sr)) / sr
    for k, semi in enumerate([0, 4, 7, 12]):
        f0 = 392.0 * 2 ** (semi / 12)
        add(np.sin(2 * np.pi * f0 * t2) * np.exp(-t2 * 3.2) * (1 - np.exp(-t2 * 800)), r.tf(r.END) + 0.07 * k, 0.12)
    # whoosh de remise a zero
    dur = RESET_FRAMES / FPS
    tw = np.arange(int(dur * sr)) / sr
    ph = 2 * np.pi * np.cumsum(180 * (1 + (tw / dur) ** 2 * 5)) / sr
    win = np.sin(np.pi * tw / dur) ** 2
    noise = np.convolve(rng.standard_normal(len(tw)), np.ones(24) / 24, mode="same")
    add(np.sin(ph) * win, r.tf(r.hold_end), 0.08)
    add(noise * win, r.tf(r.hold_end), 0.25)
    a *= 0.85 / max(1e-6, float(np.abs(a).max()))
    fade = int(0.05 * sr)
    a[-fade:] *= np.linspace(1, 0, fade)
    return (a * 32767).astype(np.int16), sr


def write_wav(path, data, sr):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="fill_race_v2.mp4")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--preview", type=int, nargs="*")
    ap.add_argument("--prev-dir", default=".")
    args = ap.parse_args()

    r = Renderer(args.seed)
    print(f"pieces: {r.n} | images: {r.total_out} | duree: {r.total_out / FPS:.1f}s")
    for (name, *_), f in zip(STRATS, r.final):
        print(f"  {name:14s} {f:5.1f}%")
    print(f"gagnant: {STRATS[r.winner][0]} | ecart: {r.gap:.2f}% | photo-finish: {'oui' if r.photo else 'non'}")
    keyF = lambda t: int(np.interp(t, r.tau_grid, r.Fcum))
    print(f"images cles: ralenti {keyF(r.f_slow)}-{keyF(r.f_slow_end)} | gagnant {keyF(r.END)} | "
          f"reset {keyF(r.hold_end)} | fin {r.total_out - 1}")
    if args.dry:
        return
    if args.preview:
        for F in sorted(args.preview):
            path = os.path.join(args.prev_dir, f"v2_{F:04d}.png")
            Image.fromarray(r.compose(r.tau_of(F), F)).save(path)
            print("ecrit", path)
        return

    base = os.path.splitext(args.out)[0]
    tmp_v, tmp_a, tmp_o = base + ".v.tmp.mp4", base + ".tmp.wav", base + ".tmp.mp4"
    data, sr = synth(r)
    write_wav(tmp_a, data, sr)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", tmp_v]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    first = last = None
    for F in range(r.total_out):
        arr = np.ascontiguousarray(r.compose(r.tau_of(F), F))
        if F == 0:
            first = arr.copy()
        last = arr
        p.stdin.write(arr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg video a echoue")
    print(f"boucle : ecart moyen premiere/derniere image = "
          f"{float(np.abs(first.astype(int) - last.astype(int)).mean()):.3f} (0 = identiques)")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_v, "-i", tmp_a,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", tmp_o], check=True)
    os.replace(tmp_o, args.out)
    for t in (tmp_v, tmp_a):
        os.remove(t)
    print("ok ->", args.out)


if __name__ == "__main__":
    main()
