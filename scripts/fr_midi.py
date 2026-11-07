"""Lecteur MIDI minimal (sans bibliotheque) + construction du plan de melodie pour Packing Wars."""
import struct
import numpy as np


def _vlq(d, i):
    v = 0
    while True:
        b = d[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def parse(path):
    d = open(path, "rb").read()
    assert d[:4] == b"MThd"
    hl = struct.unpack(">I", d[4:8])[0]
    fmt, ntr, div = struct.unpack(">HHH", d[8:14])
    i = 8 + hl
    tempos = [(0, 500000)]
    tracks = []
    for _ in range(ntr):
        if d[i:i + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", d[i + 4:i + 8])[0]
        p, end = i + 8, i + 8 + ln
        t, run = 0, None
        on, notes = {}, []
        name = ""
        while p < end:
            dt, p = _vlq(d, p)
            t += dt
            b = d[p]
            if b == 0xFF:
                typ = d[p + 1]
                l, q = _vlq(d, p + 2)
                data = d[q:q + l]
                if typ == 0x51 and l == 3:
                    tempos.append((t, int.from_bytes(data, "big")))
                if typ == 0x03:
                    name = data.decode("latin1")
                p = q + l
                continue
            if b in (0xF0, 0xF7):
                l, q = _vlq(d, p + 1)
                p = q + l
                continue
            if b & 0x80:
                run = b
                p += 1
            st = run
            kind = st & 0xF0
            if kind in (0xC0, 0xD0):
                p += 1
                continue
            a, c = d[p], d[p + 1]
            p += 2
            key = (st & 0x0F, a)
            if kind == 0x90 and c > 0:
                on[key] = (t, c)
            elif kind == 0x80 or (kind == 0x90 and c == 0):
                if key in on:
                    t0, vel = on.pop(key)
                    notes.append((t0, t - t0, a, vel))
        tracks.append({"name": name, "notes": sorted(notes)})
        i = end
    return div, sorted(tempos), tracks


def tick_to_sec(tempos, div):
    """Fonction ticks -> secondes (tempo variable)."""
    segs, sec, last_t, last_us = [], 0.0, 0, 500000
    for t, us in tempos:
        sec += (t - last_t) * last_us / 1e6 / div
        segs.append((t, sec, us))
        last_t, last_us = t, us
    segs_arr = segs

    def f(tick):
        base = (0, 0.0, 500000)
        for s in segs_arr:
            if s[0] <= tick:
                base = s
            else:
                break
        return base[1] + (tick - base[0]) * base[2] / 1e6 / div
    return f


def monophonic(notes, min_gap=0.0):
    """Garde la note la plus haute quand plusieurs notes partent ensemble."""
    out, i = [], 0
    notes = sorted(notes)
    while i < len(notes):
        j = i
        grp = []
        while j < len(notes) and notes[j][0] == notes[i][0]:
            grp.append(notes[j])
            j += 1
        out.append(max(grp, key=lambda n: n[2]))
        i = j
    return out


# ---------- plan de melodie pour le moteur (meme interface que fr_melody.build) ----------
PATH = "/home/claude/bluebird.mid"
START = 0            # indice de la premiere note de la main droite a jouer
FPS = 30.0           # images d'animation par seconde
TARGET_SEC = 19.0    # duree visee de la phase de pose
KEEP_LH = True
SPEED = None         # si fixe : facteur de tempo impose (1.0 = tempo du MIDI), sinon deduit de TARGET_SEC
_cache = {}


def _load():
    if "d" not in _cache:
        div, tempos, tracks = parse(PATH)
        f = tick_to_sec(tempos, div)
        rh = [(f(t), f(t + d) - f(t), p) for t, d, p, v in monophonic(tracks[1]["notes"])]
        lh = [(f(t), p) for t, d, p, v in tracks[2]["notes"]] if len(tracks) > 2 else []
        _cache["d"] = (rh, lh)
    return _cache["d"]


def build(n, offset=14):
    rh, lh = _load()
    seq = [rh[(START + i) % len(rh)] for i in range(n)]
    t0 = seq[0][0]
    span = max(0.5, seq[-1][0] - t0)
    speed = SPEED or span / TARGET_SEC              # >1 : on accelere
    frames = np.array([offset + (t - t0) / speed * FPS for t, _, _ in seq], float)
    midi = [p for _, _, p in seq]
    last = frames[-1]
    out = []
    if KEEP_LH:
        for t, p in lh:
            f = offset + (t - t0) / speed * FPS
            if offset - 2 <= f <= last + 8:
                out.append((f, p))
    _cache["speed"] = speed
    return {"frames": frames, "midi": midi, "lh": out}


def midi_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)
