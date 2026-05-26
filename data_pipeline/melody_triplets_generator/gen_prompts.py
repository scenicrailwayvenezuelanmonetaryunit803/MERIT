#!/usr/bin/env python3
"""
Generate diverse music style prompts suitable for JASCO melody-conditioned generation.
Prompts are written to sound natural and human-like, not too specific or formulaic.

Usage:
    python gen_prompts.py                           # writes prompts_10000.txt + prompts_5000.txt
    python gen_prompts.py --out my_prompts.txt      # custom output file
    python gen_prompts.py -n 3000                   # generate N prompts
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Component pools  (simplified for natural human-like prompts)
# ---------------------------------------------------------------------------

GENRES = [
    # Rock & guitar music
    "rock", "classic rock", "indie rock", "alternative rock", "folk rock",
    "blues rock", "psychedelic rock", "soft rock", "hard rock", "country rock",
    "surf rock", "garage rock",
    # Metal & punk
    "heavy metal", "punk", "pop punk", "emo", "hard rock",
    # Jazz
    "jazz", "smooth jazz", "jazz fusion", "bebop", "swing", "big band",
    "cool jazz", "Latin jazz", "bossa nova",
    # Blues
    "blues", "delta blues", "Chicago blues", "electric blues",
    # Pop
    "pop", "indie pop", "synth-pop", "dream pop", "electro pop", "K-pop",
    # Electronic
    "electronic", "ambient", "house", "techno", "trance", "trip-hop",
    "drum and bass", "dubstep", "chillout", "lo-fi",
    # Hip hop / R&B
    "hip hop", "R&B", "soul", "funk", "neo soul", "trap", "lo-fi hip hop",
    # Country / folk
    "country", "folk", "bluegrass", "Americana", "singer-songwriter",
    "Celtic folk", "country folk",
    # Classical / orchestral
    "classical", "orchestral", "cinematic", "film score",
    # Latin / world
    "samba", "salsa", "reggae", "reggaeton", "flamenco",
    "Afrobeat", "dancehall", "Latin",
    # Other
    "gospel", "soul", "disco", "new wave", "shoegaze", "indie",
    "easy listening", "lounge",
]

ERA_MODIFIERS = [
    "", "", "", "", "",  # blank = no era (weighted more heavily)
    "60s", "late 60s",
    "70s", "late 70s",
    "80s", "early 80s", "late 80s",
    "90s", "early 90s", "late 90s",
    "2000s",
    "modern", "vintage", "classic", "retro",
]

INSTRUMENTS = [
    # Guitar / strings
    "guitar", "acoustic guitar", "electric guitar", "bass guitar",
    "violin", "cello", "banjo", "ukulele", "mandolin",
    # Keys
    "piano", "synthesizer", "organ", "electric piano",
    # Wind / brass
    "saxophone", "trumpet", "flute", "clarinet", "trombone",
    # Percussion / other
    "drums", "harmonica", "harp",
    # Voice
    "vocals", "choir",
]

MOODS = [
    "upbeat", "chill", "sad", "happy", "energetic", "relaxing",
    "groovy", "melancholic", "powerful", "dreamy", "smooth", "dark",
    "soulful", "romantic", "nostalgic", "playful", "dramatic",
    "peaceful", "intense", "warm", "bright", "emotional", "catchy",
    "mellow", "lively", "gentle",
]

TEMPO_WORDS = [
    "", "", "",  # blank = no tempo (weighted more)
    "fast", "slow", "mid-tempo", "uptempo", "upbeat", "laid-back",
]

# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def make_prompt(rng: random.Random) -> str:
    """Build a natural-sounding music description prompt."""
    genre = rng.choice(GENRES)
    era = rng.choice(ERA_MODIFIERS)
    instr = rng.choice(INSTRUMENTS)
    instr2 = rng.choice(INSTRUMENTS)
    while instr2 == instr:
        instr2 = rng.choice(INSTRUMENTS)
    mood = rng.choice(MOODS)
    tempo = rng.choice(TEMPO_WORDS)

    # Build era prefix (e.g. "80s rock" or just "rock")
    if era:
        eg = f"{era} {genre}"
    else:
        eg = genre

    template = rng.randint(1, 14)

    if template == 1:
        p = f"A {mood} {genre} song with {instr}"
    elif template == 2:
        p = f"{eg.capitalize()} with {instr}"
    elif template == 3:
        if era:
            p = f"{eg.capitalize()} music with {instr}"
        else:
            p = f"A {mood} {genre} track featuring {instr}"
    elif template == 4:
        p = f"{mood.capitalize()} {genre} with {instr} and {instr2}"
    elif template == 5:
        p = f"A {genre} song featuring {instr}, {mood} feel"
    elif template == 6:
        if era:
            p = f"{eg.capitalize()} song"
        else:
            p = f"A catchy {genre} track with {instr}"
    elif template == 7:
        p = f"{genre.capitalize()} music with {instr} in the background, {mood} vibe"
    elif template == 8:
        if tempo:
            p = f"A {tempo} {genre} song with {instr}"
        else:
            p = f"A {mood} {genre} track with {instr} and {instr2}"
    elif template == 9:
        p = f"{instr.capitalize()}-led {genre} music, {mood} and {rng.choice(MOODS)}"
    elif template == 10:
        p = f"A {genre} piece with {instr}, {mood} mood"
    elif template == 11:
        if era:
            p = f"{eg.capitalize()} inspired music"
        else:
            p = f"{mood.capitalize()} {genre} music featuring {instr}"
    elif template == 12:
        p = f"A {genre} song with {instr} and {instr2}"
    elif template == 13:
        p = f"{mood.capitalize()} and {rng.choice(MOODS)} {genre} track with {instr}"
    else:
        if era:
            p = f"{eg.capitalize()} song with {instr}, {mood} feel"
        else:
            p = f"A {mood} {genre} song"

    return p


def generate_prompts(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    prompts: list[str] = []
    # Try up to 10x to find unique ones, then fall back to repeats with shuffled seed
    attempts = 0
    max_attempts = n * 20
    while len(prompts) < n and attempts < max_attempts:
        p = make_prompt(rng)
        attempts += 1
        if p not in seen:
            seen.add(p)
            prompts.append(p)
    # If still short, add variants with slight wording changes
    if len(prompts) < n:
        extra_seed = seed + 1
        rng2 = random.Random(extra_seed)
        while len(prompts) < n:
            p = make_prompt(rng2)
            if p not in seen:
                seen.add(p)
                prompts.append(p)
    return prompts


def main() -> None:
    script_dir = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Generate diverse JASCO-compatible music prompts.")
    ap.add_argument("-n", "--num-prompts", type=int, default=10000,
                    help="Number of prompts to generate (default: 10000)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output file. Default: prompts_10000.txt (or prompts_N.txt for other N).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out is None:
        args.out = script_dir / f"prompts_{args.num_prompts}.txt"

    print(f"Generating {args.num_prompts} prompts (seed={args.seed}) ...")
    prompts = generate_prompts(args.num_prompts, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"Written {len(prompts)} prompts -> {args.out}")

    # Also write prompts_5000.txt (first 5000 for backward compat / JASCO use)
    p5k_path = script_dir / "prompts_5000.txt"
    p5k_path.write_text("\n".join(prompts[:5000]) + "\n", encoding="utf-8")
    print(f"Written first 5000 prompts -> {p5k_path}")

    # Quick stats
    from collections import Counter
    genre_counts: Counter = Counter()
    for p in prompts:
        for g in GENRES:
            if g in p.lower():
                genre_counts[g] += 1
                break
    print(f"Top 10 genres represented: {genre_counts.most_common(10)}")


if __name__ == "__main__":
    main()
