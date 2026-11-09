"""Melodies du domaine public ecrites a la main (sans fichier MIDI), au format (temps en temps, note MIDI, duree).
Utilisables comme une piste MIDI : melody(name) -> liste (t_sec, pitch), bass(name) -> liste (t_sec, pitch)."""

# --- Korobeiniki (chanson populaire russe, la musique de Tetris) : la mineur, en noires=1 ---
_K = [  # (duree en temps, note)
    (1, 76), (.5, 71), (.5, 72), (1, 74), (.5, 72), (.5, 71),
    (1, 69), (.5, 69), (.5, 72), (1, 76), (.5, 74), (.5, 72),
    (1.5, 71), (.5, 72), (1, 74), (1, 76),
    (1, 72), (1, 69), (2, 69),
    (.5, None), (1, 74), (.5, 77), (1, 81), (.5, 79), (.5, 77),
    (1.5, 76), (.5, 72), (1, 76), (.5, 74), (.5, 72),
    (1, 71), (.5, 71), (.5, 72), (1, 74), (1, 76),
    (1, 72), (1, 69), (2, 69),
]
_K_BASS = [40, 45, 40, 45, 38, 36, 40, 45]                # une fondamentale par mesure : E A E A D C E A

# --- Dans l'antre du roi de la montagne (Grieg) : si mineur, en croches=0.5 ---
_M = [
    (.5, 59), (.5, 61), (.5, 62), (.5, 64), (.5, 66), (.5, 62), (1, 66),
    (.5, 65), (.5, 61), (1, 65), (.5, 64), (.5, 60), (1, 64),
    (.5, 59), (.5, 61), (.5, 62), (.5, 64), (.5, 66), (.5, 62), (.5, 66), (.5, 71),
    (.5, 69), (.5, 66), (.5, 62), (.5, 66), (2, 69),
    (.5, 66), (.5, 68), (.5, 70), (.5, 71), (.5, 73), (.5, 70), (1, 73),
    (.5, 74), (.5, 70), (1, 74), (.5, 73), (.5, 70), (1, 73),
    (.5, 66), (.5, 68), (.5, 70), (.5, 71), (.5, 73), (.5, 70), (.5, 73), (.5, 78),
    (.5, 76), (.5, 73), (.5, 70), (.5, 73), (2, 76),
]
_M_BASS = [47, 42, 47, 42, 47, 42, 47, 42, 42, 49, 42, 49, 42, 49, 42, 49]   # pulsation si / fa# en noires

SONGS = {
    "korobeiniki": dict(notes=_K, bpm=150, bass=_K_BASS, bass_every=4, root=110.0, chord=(45, 52, 57, 60, 64, 69)),
    "mountain_king": dict(notes=_M, bpm=132, bass=_M_BASS, bass_every=1, root=61.74, chord=(47, 54, 59, 62, 66, 71)),
}


def melody(name, repeats=2):
    sg = SONGS[name]
    beat = 60.0 / sg["bpm"]
    out, t = [], 0.0
    for _ in range(repeats):
        for d, p in sg["notes"]:
            if p is not None:
                out.append((t * beat, p))
            t += d
    return out


def bass(name, repeats=2):
    sg = SONGS[name]
    beat = 60.0 / sg["bpm"]
    total = sum(d for d, _ in sg["notes"]) * repeats
    out, t, i = [], 0.0, 0
    while t < total:
        out.append((t * beat, sg["bass"][i % len(sg["bass"])]))
        t += sg["bass_every"]
        i += 1
    return out
