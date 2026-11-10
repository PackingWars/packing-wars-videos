"""Concept 1 : BOCAL A BILLES. 4 bocaux de meme surface (rond, carre, etoile, coeur), memes billes : lequel en contient le plus ?"""
import sys
import numpy as np
import cv2
from scipy.spatial import cKDTree
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "ep10_marbles.mp4"
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
LW, LH = 500, 600                     # zone locale d'un bocal
R = 9.0                              # rayon d'une bille
AREA = 92000                          # meme surface pour les 4 bocaux
CX, CY = 250, 330
NAMES = ["ROUND", "SQUARE", "STAR", "HEART"]
COLS = [pw.NEON[4], pw.NEON[0], pw.NEON[2], pw.NEON[5]]
PX = [30, 550]
PY = [330, 1030]


def shape_mask(kind, s):
    m = np.zeros((LH, LW), np.uint8)
    if kind == 0:
        cv2.circle(m, (CX, CY), int(s), 255, -1)
    elif kind == 1:
        cv2.rectangle(m, (int(CX - s), int(CY - s)), (int(CX + s), int(CY + s)), 255, -1)
    elif kind == 2:
        a = np.arange(10) * np.pi / 5 - np.pi / 2
        rr = np.where(np.arange(10) % 2 == 0, s, s * 0.52)
        pts = np.stack([CX + rr * np.cos(a), CY + 25 + rr * np.sin(a)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 255)
    else:
        t = np.linspace(0, 2 * np.pi, 400)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        pts = np.stack([CX + x * s / 16, CY + 20 + y * s / 16], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 255)
    return m


def build(kind):
    s = 150.0
    for _ in range(4):
        s *= np.sqrt(AREA / max(1, (shape_mask(kind, s) > 0).sum()))
    m = shape_mask(kind, s) > 0
    ys = np.where(m.any(1))[0]
    top = ys[0]
    allowed = m.copy()
    w = int(R * 2.7)
    ch = np.zeros_like(m)                                  # goulot ouvert vers le haut
    ch[0:CY, CX - w:CX + w] = True
    allowed |= ch
    allowed[:2, CX - w:CX + w] = True
    d = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
    nrm = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    cnt, _ = cv2.findContours(allowed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return dict(mask=m, d=d, gx=gx / nrm, gy=gy / nrm, contour=max(cnt, key=len), top=top)


class Jar:
    def __init__(self, kind):
        self.k = kind
        self.g = build(kind)
        self.x = np.zeros((0, 2), np.float32)
        self.v = np.zeros((0, 2), np.float32)
        self.col = []
        self.landed = []
        self.age = []
        self.blocked = 0
        self.full = False

    def spawn(self, colr):
        p = np.array([CX + rng.uniform(-6, 6), R + 3], np.float32)
        if len(self.x) and (np.hypot(*(self.x - p).T) < 2.05 * R).any():
            self.blocked += 1
            if self.blocked > 8:
                self.full = True
            return False
        self.blocked = 0
        self.x = np.vstack([self.x, p])
        self.v = np.vstack([self.v, [0, 650]]).astype(np.float32)
        self.col.append(colr)
        self.landed.append(False)
        self.age.append(0.0)
        return True

    def step(self, dt, events, t):
        if not len(self.x):
            return
        g = self.g
        prev = self.x.copy()
        self.v[:, 1] += 1400 * dt
        self.x += self.v * dt
        for _ in range(3):
            if len(self.x) > 1:
                pr = np.array(list(cKDTree(self.x).query_pairs(2 * R)), dtype=np.int64)
                if len(pr):
                    dv = self.x[pr[:, 1]] - self.x[pr[:, 0]]
                    dist = np.sqrt((dv ** 2).sum(1)) + 1e-6
                    push = ((2 * R - dist) / dist * 0.5)[:, None] * dv
                    np.add.at(self.x, pr[:, 0], -push)
                    np.add.at(self.x, pr[:, 1], push)
            xi = np.clip(self.x[:, 0].round().astype(int), 0, LW - 1)
            yi = np.clip(self.x[:, 1].round().astype(int), 0, LH - 1)
            dd = g["d"][yi, xi]
            hit = dd < R
            if hit.any():
                k = (R - dd[hit])
                self.x[hit, 0] += g["gx"][yi[hit], xi[hit]] * k
                self.x[hit, 1] += g["gy"][yi[hit], xi[hit]] * k
        nv = (self.x - prev) / dt
        self.age = [a + dt for a in self.age]
        age = np.array(self.age)
        sp = np.hypot(nv[:, 0], nv[:, 1])
        hitnow = ((self.v[:, 1] > 300) & (nv[:, 1] < 0.35 * self.v[:, 1])) | ((age > 0.25) & (sp < 120)) | (age > 1.5)
        for i in np.where((~np.array(self.landed)) & hitnow)[0]:     # chaque bille joue sa note une fois, a son arret
            self.landed[i] = True
            events.append((t, float(self.x[i, 1]), self.k))
        self.v = nv * 0.985

    def count(self):
        if not len(self.x):
            return 0
        xi = np.clip(self.x[:, 0].round().astype(int), 0, LW - 1)
        yi = np.clip(self.x[:, 1].round().astype(int), 0, LH - 1)
        return int(self.g["mask"][yi, xi].sum())


jars = [Jar(k) for k in range(4)]
auL, auR = pw.Audio(60), pw.Audio(60)
BUS = [auL, auR, auL, auR]                               # bocaux de gauche a gauche, de droite a droite
PENT = [0, 3, 5, 7, 10]
BASE = [0, 2, 4, 1]


def note(k):
    o, d = divmod(int(k), 5)
    return 57 + PENT[d] + 12 * o
vid = pw.Video(out)
SPAWN_T = 45.0
nf_fill = int(SPAWN_T * pw.FPS)
frames_end = nf_fill + 30
HOLD = 80
spawned = 0
final = None
n_notes = 0
f = 0
while f < frames_end + HOLD:
    t = f / pw.FPS
    if f < nf_fill and all(j.full for j in jars):         # tous pleins : le resultat arrive tout de suite
        nf_fill = f + 1
        frames_end = f + 18
    if f < nf_fill and f % 2 == 0:
        colr = pw.hsv((spawned * 0.0065) % 1, 0.75, 1.0)
        for j in jars:
            if not j.full:
                j.spawn(colr)
        spawned += 1
    ev = []
    if f < frames_end:
        for j in jars:
            for s in range(4):
                j.step(1 / (pw.FPS * 4), ev, t)
    for (te, y, jk) in ev:                               # une note de kalimba pour CHAQUE bille qui se pose
        fill = min(1.0, jars[jk].count() / 300)
        BUS[jk].kalimba(te + rng.uniform(0, 0.015), note(BASE[jk] + int(fill * 9)), 0.09 / np.sqrt(max(1, len(ev) / 2)))
    n_notes += len(ev)
    counts = [j.count() for j in jars]
    if f == frames_end:
        final = counts
        win = int(np.argmax(counts))
        for i, m in enumerate((57, 64, 69, 72, 76, 81)):      # accord final la mineur, cote du gagnant
            BUS[win].kalimba(t + 0.09 * i, m, 0.16)
            BUS[1 - win % 2].kalimba(t + 0.09 * i, m, 0.08)
        for q in range(2):
            pw.Audio.swell(BUS[q], t, 2.0, 0.02, root=110.0 * 2 ** (3 / 12))
    img = np.full((pw.VH, pw.VW, 3), pw.BG, np.uint8)
    for k, j in enumerate(jars):
        ox, oy = PX[k % 2], PY[k // 2]
        win_now = final is not None and k == int(np.argmax(final))
        c = COLS[k]
        inside = np.zeros((LH, LW, 3), np.uint8)
        inside[j.g["mask"] | (j.g["d"] > 0)] = (24, 27, 40)
        roi = img[oy:oy + LH, ox:ox + LW]
        roi[:] = np.maximum(roi, inside)
        for p, cc in zip(j.x, j.col):
            cx, cy = int(p[0] + ox), int(p[1] + oy)
            cv2.circle(img, (cx, cy), int(R) - 1, cc, -1, cv2.LINE_AA)
            cv2.circle(img, (cx - 3, cy - 4), 3, pw.mix(cc, 0.75), -1, cv2.LINE_AA)
        th = 6 if win_now else 3
        cv2.drawContours(img, [j.g["contour"] + np.array([ox, oy])], -1, pw.mix(c, 0.2) if win_now else c, th, cv2.LINE_AA)
    img = pw.glow(img, 60, 1.6 if final else 1.1)
    pw.header(img, "WHICH JAR HOLDS THE MOST?", "PACKING WARS #10 - Marble Jars", 140, 66)
    lead = int(np.argmax(counts)) if max(counts) > 0 else -1
    if lead >= 0 and (final is None or lead == int(np.argmax(final))):   # couronne sur le bocal en tete
        cx, cy = PX[lead % 2] + LW // 2 + 95, PY[lead // 2] + LH + 58
        pts = np.array([[cx - 22, cy + 12], [cx - 22, cy - 8], [cx - 11, cy + 1], [cx, cy - 16], [cx + 11, cy + 1],
                        [cx + 22, cy - 8], [cx + 22, cy + 12]], np.int32)
        cv2.fillPoly(img, [pts], (255, 205, 60), cv2.LINE_AA)
    for k, j in enumerate(jars):
        ox, oy = PX[k % 2], PY[k // 2]
        c = COLS[k]
        win_now = final is not None and k == int(np.argmax(final))
        pw.text(img, NAMES[k], (ox + LW // 2, oy + LH + 12), 40, c)
        pw.text(img, f"{counts[k]}", (ox + LW // 2, oy + LH + 62), 52 if not win_now else 64, (255, 255, 255) if not win_now else pw.mix(c, 0.4))
    if final is not None:
        pw.text(img, f"WINNER: {NAMES[int(np.argmax(final))]}", (pw.VW // 2, 1830), 64, pw.mix(COLS[int(np.argmax(final))], 0.3))
    else:
        pw.text(img, "Same area. Same marbles. Pick one!", (pw.VW // 2, 1830), 40, (200, 205, 220), light=True)
    vid.add(img)
    f += 1
vid.close(lambda sec: pw.render_stereo(auL, auR, sec))
print("final", dict(zip(NAMES, final)), "notes jouees", n_notes, "billes", sum(len(j.x) for j in jars), "duree", round(vid.n / pw.FPS, 1))
