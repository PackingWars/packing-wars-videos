"""Concept 4 : IMAGE CACHEE. Une mosaique de tuiles (grosses dans les zones unies, petites dans les details) revele une image."""
import sys
import numpy as np
import cv2
import pw

out = sys.argv[1] if len(sys.argv) > 1 else "proto4_mosaic.mp4"
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
X0, Y0, W, H = 40, 330, 1000, 1360


def picture():
    """Image originale dessinee par code : citrouille d'automne sous la lune, feuilles qui tombent."""
    yy = np.linspace(0, 1, H)[:, None]
    top, mid, low = np.array([22, 18, 60]), np.array([90, 40, 110]), np.array([240, 120, 70])
    sky = np.where(yy < 0.6, top + (mid - top) * (yy / 0.6), mid + (low - mid) * ((yy - 0.6) / 0.4))
    im = np.repeat(sky[:, None, :], W, 1).astype(np.uint8).copy()
    for _ in range(80):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, 600))
        cv2.circle(im, (x, y), int(rng.integers(1, 3)), (255, 250, 220), -1, cv2.LINE_AA)
    cv2.circle(im, (780, 250), 110, (255, 240, 200), -1, cv2.LINE_AA)
    cv2.circle(im, (820, 225), 100, tuple(int(v) for v in im[225, 990]), -1, cv2.LINE_AA)
    cv2.ellipse(im, (500, 1330), (700, 200), 0, 180, 360, (60, 35, 40), -1, cv2.LINE_AA)
    im[1180:] = (60, 35, 40)
    cx, cy = 500, 960                                     # citrouille : cotes orange qui se chevauchent
    ribs = [(-235, 150, (205, 85, 20)), (235, 150, (205, 85, 20)), (-150, 175, (230, 105, 25)),
            (150, 175, (230, 105, 25)), (-65, 190, (250, 130, 35)), (65, 190, (250, 130, 35)), (0, 195, (255, 150, 45))]
    for dx, rx, col in ribs:
        cv2.ellipse(im, (cx + dx, cy), (rx, 240), 0, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(im, (cx + dx, cy), (rx, 240), 0, 0, 360, tuple(int(v * 0.7) for v in col), 5, cv2.LINE_AA)
    cv2.ellipse(im, (cx - 40, cy - 120), (40, 90), -20, 0, 360, (255, 190, 110), -1, cv2.LINE_AA)
    stem = np.array([[cx - 30, cy - 230], [cx + 25, cy - 230], [cx + 45, cy - 340], [cx + 5, cy - 345]])
    cv2.fillPoly(im, [stem], (70, 110, 40), cv2.LINE_AA)
    cv2.ellipse(im, (cx + 130, cy - 290), (110, 55), -25, 0, 360, (60, 140, 50), -1, cv2.LINE_AA)
    cv2.line(im, (cx + 40, cy - 300), (cx + 210, cy - 320), (40, 100, 35), 5, cv2.LINE_AA)
    for k in range(3):
        cv2.ellipse(im, (cx - 120 - k * 40, cy - 260 + k * 20), (40 + k * 6, 30), 0, 200, 470, (80, 150, 60), 6, cv2.LINE_AA)
    for (x, y, c) in ((150, 520, (220, 90, 30)), (880, 640, (240, 170, 40)), (240, 1240, (200, 60, 30)),
                      (830, 1280, (230, 140, 40)), (690, 470, (190, 70, 40))):
        cv2.ellipse(im, (x, y), (34, 18), int(rng.integers(0, 180)), 0, 360, c, -1, cv2.LINE_AA)
    return im


IMG = picture()
leaves = []


def split(x, y, w, h):
    blk = IMG[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32)
    if (w > 170 or h > 170) or (blk.std(0).mean() > 14 and min(w, h) >= 22):
        w2, h2 = w // 2, h // 2
        for (a, b, c, d) in ((x, y, w2, h2), (x + w2, y, w - w2, h2), (x, y + h2, w2, h - h2), (x + w2, y + h2, w - w2, h - h2)):
            split(a, b, c, d)
    else:
        leaves.append((x, y, w, h, tuple(int(v) for v in blk.mean(0))))


split(0, 0, W, H)
order = rng.permutation(len(leaves))
REVEAL = int(float(sys.argv[3]) if len(sys.argv) > 3 else 33) * pw.FPS
start = np.zeros(len(leaves), int)
u = np.sort(rng.random(len(leaves))) ** 1.35
start[order] = (u * REVEAL).astype(int) + 10
END = REVEAL + 20
au = pw.Audio(40)
vid = pw.Video(out)
last_sound = -1
for f in range(END + 90):
    t = f / pw.FPS
    img = np.full((pw.VH, pw.VW, 3), pw.BG, np.uint8)
    cv2.rectangle(img, (X0 - 4, Y0 - 4), (X0 + W + 4, Y0 + H + 4), (40, 44, 64), 2)
    new = np.where(start == f)[0]
    for i in new[:2]:
        w, h = leaves[i][2], leaves[i][3]
        au.soft(t + rng.uniform(0, 0.02), min(1.0, np.sqrt(w * h) / 120), 0.11)
    if f % 18 == 0 and f < END:                            # notes zen eparses (pentatonique) qui montent doucement
        au.mallet(t, 57 + [0, 3, 5, 7, 10, 12, 15, 17][(f // 18 * 3) % 8] + (12 if f > END * 0.6 else 0), 0.07)
    fade = 0.0 if f < END else min(1.0, (f - END) / 25)
    if fade < 1:
        for i in np.where(start <= f)[0]:
            x, y, w, h, c = leaves[i]
            a = f - start[i]
            s = 1.0 if a >= 7 else pw.ease(a / 7) * 1.12 if a < 5 else 1.12 - 0.12 * (a - 5) / 2
            g = 2 if min(w, h) > 12 else 1
            cw, chh = (w - 2 * g) * s, (h - 2 * g) * s
            cx, cy = X0 + x + w / 2, Y0 + y + h / 2
            col = pw.mix(c, 0.5 * max(0, 1 - a / 8))
            cv2.rectangle(img, (int(cx - cw / 2), int(cy - chh / 2)), (int(cx + cw / 2), int(cy + chh / 2)), col, -1)
    if fade > 0:
        roi = img[Y0:Y0 + H, X0:X0 + W]
        roi[:] = cv2.addWeighted(roi, 1 - fade, IMG, fade, 0)
    if f == END:
        au.swell(t, 2.0, 0.05)
        au.chord(t, (0, 5, 7, 9, 12), 0.1)
    img = pw.glow(img, 90, 0.8 if f < END else 1.0)
    pw.header(img, "GUESS THE PICTURE", "PACKING WARS #06 - Hidden Mosaic", 140, 74)
    n = int((start <= f).sum())
    pw.text(img, f"{n} / {len(leaves)} tiles", (pw.VW // 2, 1760), 46, (255, 255, 255))
    pw.text(img, "At which tile did you guess it?" if f < END else "A PUMPKIN!", (pw.VW // 2, 1840),
            40 if f < END else 60, (200, 205, 220) if f < END else (255, 205, 90), light=f < END)
    vid.add(img)
vid.close(au, rev=0.32, drive=1.4, lowpass=5000)
print("tuiles", len(leaves), "duree", round(vid.n / pw.FPS, 1))
