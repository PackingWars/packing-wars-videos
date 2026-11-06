# Packing Wars - videos

Satisfying "algorithm race" videos: 4 packing strategies fill the same box with the same pieces.

- `scripts/fill_race_v2.py` : video engine (1080x1920, 30 fps, procedural sound)
- `scripts/fill_race_batch.py` : generic batch mode (shapes, themes, captions)
- `scripts/fill_race_lot1.py` : lot 1, 10 concepts in the rectangular box
- `videos/lot1/` : rendered videos + `captions.json` / `captions.csv` (TikTok caption, Instagram caption, reveal comment)

Regenerate: `cd scripts && python3 fill_race_lot1.py --outdir ../videos/lot1`
Public video URL pattern (for Metricool): `https://raw.githubusercontent.com/PackingWars/packing-wars-videos/main/videos/lot1/<file>.mp4`
