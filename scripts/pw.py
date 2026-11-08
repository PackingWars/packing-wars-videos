"""Boite a outils commune aux nouveaux concepts Packing Wars : video, texte, neon, son."""
import os
import subprocess
import tempfile
import wave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

VW, VH, FPS, SR = 1080, 1920, 30, 44100
BG = (12, 13, 20)
NEON = [(74, 222, 128), (167, 139, 250), (251, 191, 36), (248, 113, 113),
        (56, 189, 248), (244, 114, 182), (45, 212, 191), (250, 204, 21)]
_FONTS = {}


def font(size, light=False):
    key = (size, light)
    if key not in _FONTS:
        p = "/usr/share/fonts/truetype/google-fonts/" + ("Poppins-Medium.ttf" if light else "Poppins-Bold.ttf")
        if not os.path.exists(p):
            p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        _FONTS[key] = ImageFont.truetype(p, size)
    return _FONTS[key]


def hsv(h, s=0.85, v=1.0):
    c = cv2.cvtColor(np.uint8([[[int(h * 179) % 180, int(s * 255), int(v * 255)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in c)


def mix(c, k, to=255):
    return tuple(int(v + (to - v) * k) for v in c)


def glow(img, thr=70, gain=1.3, k=31):
    """Bloom : les zones claires bavent un halo."""
    b = cv2.subtract(img, (thr, thr, thr, 0))
    small = cv2.resize(b, (VW // 4, VH // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), k / 4)
    halo = cv2.resize(small, (VW, VH), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1.0, halo, gain, 0)


def text(img, s, xy, size, col=(255, 255, 255), anchor="mm", light=False, shadow=True):
    """Ecrit du texte (Pillow) sur une image numpy RGB, en place."""
    f = font(size, light)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    if shadow:
        d.text((xy[0] + 3, xy[1] + 4), s, font=f, fill=(0, 0, 0), anchor=anchor)
    d.text(xy, s, font=f, fill=tuple(col), anchor=anchor)
    img[:] = np.asarray(pil)


def header(img, title, sub="PACKING WARS", y=120, size=74):
    f = font(size)
    while f.getlength(title) > 960 and size > 40:
        size -= 4
        f = font(size)
    text(img, title, (VW // 2, y), size)
    text(img, sub, (VW // 2, y + size * 0.9), 32, (170, 175, 195), light=True)


class Video:
    """Ecrit les images RGB dans ffmpeg puis ajoute le son."""

    def __init__(self, out):
        self.out = out
        self.tmpv = tempfile.mktemp(suffix=".mp4", dir=os.path.dirname(os.path.abspath(out)))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", self.tmpv]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def add(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
        self.n += 1

    def close(self, audio, **kw):
        self.p.stdin.close()
        self.p.wait()
        wav = self.tmpv + ".wav"
        a = audio.render(self.n / FPS, **kw)
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(a.tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.tmpv, "-i", wav, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", self.out], check=True)
        os.remove(self.tmpv)
        os.remove(wav)


# ---------------- son ----------------
_PENT = [0, 3, 5, 7, 10]


def ladder(k, root=110.0):
    o, d = divmod(int(k), 5)
    return root * 2 ** ((_PENT[d] + 12 * o) / 12)


class Audio:
    def __init__(self, seconds=90):
        self.a = np.zeros(int(seconds * SR), np.float32)
        self.rng = np.random.default_rng(3)
        self.cache = {}

    def add(self, sig, t, gain=1.0):
        st = int(t * SR)
        if 0 <= st < len(self.a):
            seg = sig[:len(self.a) - st] * gain
            self.a[st:st + len(seg)] += seg

    def _c(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn().astype(np.float32)
        return self.cache[key]

    def tock(self, t, size=0.5, gain=0.2, var=None):
        """Brique plastique : size 0 = petit/aigu, 1 = gros/grave."""
        var = int(self.rng.integers(0, 3)) if var is None else var
        q = int(np.clip(size, 0, 1) * 12)

        def mk():
            from scipy.signal import butter, sosfilt
            f = 2600 * 0.83 ** q * (1 + 0.05 * (var - 1))
            tt = np.arange(int(0.22 * SR)) / SR
            dec = 62 - 3 * q
            body = np.sin(2 * np.pi * f * tt) * np.exp(-tt * dec) + 0.45 * np.sin(2 * np.pi * f * 2.41 * tt) * np.exp(-tt * dec * 1.7)
            sos = butter(2, [1800, 7000], btype="bandpass", fs=SR, output="sos")
            click = sosfilt(sos, np.random.default_rng(q * 7 + var).standard_normal(len(tt))) * np.exp(-tt * 520)
            s = body + 0.6 * click
            if q >= 7:
                s += np.sin(2 * np.pi * (75 * tt + 55 * (1 - np.exp(-30 * tt)) / 30)) * np.exp(-tt * 16) * (q - 6) / 6
            return s * (1 - np.exp(-tt * 4000))
        self.add(self._c(("tock", q, var), mk), t, gain)

    def soft(self, t, size=0.5, gain=0.2, var=None):
        """Tap doux et feutre (zen) : petit 'pop' rond, sans aigus. size 0 = petit/aigu, 1 = gros/grave."""
        var = int(self.rng.integers(0, 3)) if var is None else var
        q = int(np.clip(size, 0, 1) * 10)

        def mk():
            from scipy.signal import butter, sosfilt
            f = 560 * 0.86 ** q * (1 + 0.04 * (var - 1))
            tt = np.arange(int(0.3 * SR)) / SR
            ph = 2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-tt * 60))) / SR
            body = np.sin(ph) * np.exp(-tt * (22 - q))
            sub = np.sin(2 * np.pi * f * 0.5 * tt) * np.exp(-tt * 14) * (0.2 + 0.05 * q)
            sos = butter(2, 1100, btype="lowpass", fs=SR, output="sos")
            felt = sosfilt(sos, np.random.default_rng(var + 3 * q).standard_normal(len(tt))) * np.exp(-tt * 90) * 0.5
            return (body + sub + felt) * (1 - np.exp(-tt * 600))
        self.add(self._c(("soft", q, var), mk), t, gain)

    def mallet(self, t, midi, gain=0.15):
        """Note douce facon marimba / kalimba (zen), hauteur MIDI."""
        def mk():
            f = 440.0 * 2 ** ((midi - 69) / 12)
            tt = np.arange(int(1.4 * SR)) / SR
            s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 3.2)
                 + 0.25 * np.sin(2 * np.pi * f * 4 * tt) * np.exp(-tt * 18)
                 + 0.12 * np.sin(2 * np.pi * f * 2 * tt) * np.exp(-tt * 7))
            return s * (1 - np.exp(-tt * 350))
        self.add(self._c(("mallet", int(midi)), mk), t, gain)

    def swell(self, t_end, d=2.0, gain=0.1, root=110.0):
        """Montee douce (nappe qui gonfle) a la place du riser bruite."""
        tt = np.arange(int(d * SR)) / SR
        env = (tt / d) ** 2
        s = sum(a * np.sin(2 * np.pi * root * m * tt) for m, a in ((1, 1.0), (1.5, 0.6), (2, 0.5), (3, 0.2)))
        self.add((s * env).astype(np.float32), t_end - d, gain)

    def glass(self, t, f=2400, gain=0.12):
        """Bille de verre qui tinte."""
        f = int(f / 50) * 50

        def mk():
            tt = np.arange(int(0.35 * SR)) / SR
            return (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 18) + 0.5 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt * 30)
                    + 0.3 * np.sin(2 * np.pi * f * 5.4 * tt) * np.exp(-tt * 45)) * (1 - np.exp(-tt * 3000))
        self.add(self._c(("glass", f), mk), t, gain)

    def pluck(self, t, k, gain=0.15, root=110.0, dur=1.2):
        def mk():
            f = ladder(k, root)
            tt = np.arange(int(dur * SR)) / SR
            env = np.exp(-tt * 5.0) * (1 - np.exp(-tt * 900))
            return (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(4 * np.pi * f * tt) * np.exp(-tt * 8)
                    + 0.15 * np.sin(6 * np.pi * f * tt) * np.exp(-tt * 12)) * env
        self.add(self._c(("pluck", int(k), root, dur), mk), t, gain)

    def bloop(self, t, f, gain=0.15):
        """Bulle qui eclot : petit glissando montant."""
        f = int(f / 20) * 20

        def mk():
            tt = np.arange(int(0.16 * SR)) / SR
            ph = 2 * np.pi * np.cumsum(f * (0.6 + 0.8 * (1 - np.exp(-tt * 40)))) / SR
            return np.sin(ph) * np.exp(-tt * 26) * (1 - np.exp(-tt * 2000))
        self.add(self._c(("bloop", f), mk), t, gain)

    def sand(self, t0, t1, gain=0.05):
        """Souffle de sable qui coule."""
        from scipy.signal import butter, sosfilt
        n = int((t1 - t0) * SR)
        if n <= 0:
            return
        sos = butter(2, [2500, 9000], btype="bandpass", fs=SR, output="sos")
        nz = sosfilt(sos, self.rng.standard_normal(n))
        g = np.abs(np.convolve(self.rng.standard_normal(n), np.ones(400) / 400, mode="same")) * 3 + 0.5
        env = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.3 * SR))
        self.add((nz * g * env).astype(np.float32), t0, gain)

    def riser(self, t_end, rd=2.0, gain=0.14):
        tw = np.arange(int(rd * SR)) / SR
        env = (tw / rd) ** 2.2
        ph = 2 * np.pi * np.cumsum(180 * (1 + 9 * (tw / rd) ** 2)) / SR
        nz = np.convolve(self.rng.standard_normal(len(tw)), np.ones(10) / 10, mode="same")
        self.add(((np.sin(ph) * 0.6 + nz * 0.7) * env).astype(np.float32), t_end - rd, gain)

    def boom(self, t, gain=0.45):
        tb = np.arange(int(2.0 * SR)) / SR
        self.add((np.sin(2 * np.pi * (46 * tb + 90 * (1 - np.exp(-6 * tb)) / 6)) * np.exp(-tb * 1.8)).astype(np.float32), t, gain)

    def chord(self, t, ks=(5, 7, 9, 12, 14), gain=0.1, root=110.0):
        for i, k in enumerate(ks):
            self.pluck(t + 0.05 * i, k, gain, root, 2.2)

    def whoosh(self, t, d=0.8, gain=0.12):
        tw = np.arange(int(d * SR)) / SR
        win = np.sin(np.pi * tw / d) ** 2
        nz = np.convolve(self.rng.standard_normal(len(tw)), np.ones(24) / 24, mode="same")
        self.add((nz * win).astype(np.float32), t, gain)

    def crash(self, t, gain=0.3):
        """Effondrement : pluie de briques + choc grave."""
        self.boom(t, gain * 0.8)
        for i in range(26):
            self.tock(t + (i / 26) ** 1.4 * 1.2 + self.rng.uniform(0, 0.03), self.rng.uniform(0.2, 1), gain * 0.5 * (1 - 0.6 * i / 26))

    def render(self, seconds, rev=0.18, drive=2.6, lowpass=None):
        import fr_music as fm
        a = self.a[:int((seconds + 0.01) * SR)].copy()
        if np.abs(a).max() < 1e-6:
            return np.zeros(len(a), np.int16)
        if lowpass:
            from scipy.signal import butter, sosfilt
            a = sosfilt(butter(2, lowpass, btype="lowpass", fs=SR, output="sos"), a).astype(np.float32)
        a = fm.reverb(a, SR, 1.6 if rev > 0.25 else 1.0, rev)
        a = np.tanh(a * drive / max(1e-6, float(np.abs(a).max())))
        a *= 0.88 / max(1e-6, float(np.abs(a).max()))
        f = int(0.05 * SR)
        a[-f:] *= np.linspace(1, 0, f)
        return (a * 32767).astype(np.int16)


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3
