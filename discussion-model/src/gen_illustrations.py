#!/usr/bin/env python3
"""Generate conceptual illustrations for the LLM A-to-Z notebook
Chapter 10 (scaling-up arc) via OpenAI's image API.

Output: PNGs in discussion-model/figures/, sized 1024x1024.
Cost: ~$0.04-0.10 per image at "medium" quality (4 images total).

Uses gpt-image-2 by default per the project's image-model convention;
fall back to gpt-image-1 if your account doesn't have 2 access yet.

Run:
    /Users/thiel/anaconda3/bin/python discussion-model/src/gen_illustrations.py
"""

import base64
import os
import sys
import time
from openai import OpenAI

MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-2")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI()

ILLUSTRATIONS = [
    ("scaling_up.png",
     "Soft watercolor illustration: a small chibi friendly robot on the "
     "left looking at a tiny stack of 20 paper notes, and a larger but "
     "still cute chibi robot on the right surrounded by a tall library "
     "shelf of 20,000 books. Both robots smile gently. Light pastel blue, "
     "cream, and warm yellow palette. Plenty of empty white space, gentle "
     "shadows. No text, no labels, no captions."),
    ("cloud_training.png",
     "Soft pastel watercolor illustration: a friendly robot sending a "
     "neural network diagram up into a cloud floating in the sky. Inside "
     "the cloud, glowing green and gold nodes form a stylized circuit "
     "board with subtle motion lines suggesting computation. Below the "
     "cloud, a small Mac mini computer sits on a desk, looking peaceful. "
     "Light blue and warm pink palette. Wholesome, calm. No text."),
    ("oom_humor.png",
     "Soft watercolor cartoon: a friendly robot trying very hard to lift "
     "an enormously oversized brain made of glowing crystals, sweat drops "
     "around its head. The brain is too heavy and slightly squishing the "
     "robot. The mood is sympathetic and a little funny, not distressing. "
     "Pastel teal and coral palette. No text, no labels, no captions."),
    ("chat_thinking.png",
     "Soft pastel watercolor illustration: a friendly small robot reading "
     "a tiny letter (envelope visible), a thought bubble above its head "
     "filled with a swirl of cursive text fragments, books, and a glowing "
     "lightbulb. A second smaller speech bubble points back from the "
     "robot, indicating it's about to reply. Light beige and sky-blue "
     "palette. Calm and curious mood. No text inside bubbles or on the "
     "image."),
]


def main():
    print(f"using image model: {MODEL}")
    for fname, prompt in ILLUSTRATIONS:
        out = os.path.join(OUT_DIR, fname)
        if os.path.exists(out):
            print(f"  skip (exists): {out}")
            continue
        print(f"  generating {fname} ...")
        t0 = time.time()
        try:
            res = client.images.generate(
                model=MODEL,
                prompt=prompt,
                size="1024x1024",
                quality="medium",
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    if this is a model-name error, retry with "
                  f"IMAGE_MODEL=gpt-image-1 python {sys.argv[0]}")
            return
        img_b64 = res.data[0].b64_json
        with open(out, "wb") as f:
            f.write(base64.b64decode(img_b64))
        print(f"    -> {out} in {time.time()-t0:.1f}s, "
              f"{os.path.getsize(out)} bytes")
    print("done.")


if __name__ == "__main__":
    main()
