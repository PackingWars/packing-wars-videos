"""Melodie du domaine public pour caler les pieces sur des notes : 'Fur Elise' (Beethoven), theme A.

Ecrite de memoire note par note (grille de doubles-croches) : a verifier a l'oreille.
Chaque piece qui se pose joue la note suivante ; la main gauche joue des arpeges en accompagnement.
"""
import numpy as np

SECTION_UNITS = 48            # longueur d'une section (2 phrases) en doubles-croches

# (position, note MIDI, duree) dans une section de 48 unites
_PHRASE_A = [(0, 76), (1, 75), (2, 76), (3, 75), (4, 76), (5, 71), (6, 74), (7, 72), (8, 69), (11, 60), (12, 64),
             (13, 69), (14, 71), (17, 64), (18, 68), (19, 71), (20, 72)]
_PHRASE_B = [(24, 76), (25, 75), (26, 76), (27, 75), (28, 76), (29, 71), (30, 74), (31, 72), (32, 69), (35, 60),
             (36, 64), (37, 69), (38, 71), (41, 64), (42, 72), (43, 71), (44, 69)]
MELODY = _PHRASE_A + _PHRASE_B                       # 34 notes par section

# main gauche : A2 E3 A3 / E2 E3 G#3 / A2 E3 A3, puis idem a +24
_LH_A = [(8, 45), (9, 52), (10, 57), (14, 40), (15, 52), (16, 56), (20, 45), (21, 52), (22, 57)]
LH = _LH_A + [(p + 24, m) for p, m in _LH_A]


def build(n, unit=4, offset=14):
    """Instants (en images) des n premieres notes de la melodie (qui boucle), + accompagnement."""
    per = len(MELODY)
    frames, midi = [], []
    for i in range(n):
        sec, k = divmod(i, per)
        pos, m = MELODY[k]
        frames.append(offset + (sec * SECTION_UNITS + pos) * unit)
        midi.append(m)
    last = frames[-1]
    lh = []
    sec = 0
    while True:
        base = sec * SECTION_UNITS
        for pos, m in LH:
            f = offset + (base + pos) * unit
            if f <= last + 3 * unit:
                lh.append((f, m))
        if offset + (base + SECTION_UNITS) * unit > last:
            break
        sec += 1
    return {"frames": np.array(frames, float), "midi": midi, "lh": lh}


def midi_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)
