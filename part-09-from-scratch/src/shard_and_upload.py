#!/usr/bin/env python3
"""Split a uint16 token .bin into N-token shards, upload to S3, write a manifest.

The manifest is consumed by run_pretraining.wls.  Shards and checkpoint live
under <run-prefix>/ in the Batch IO bucket.

Usage:
  python3 shard_and_upload.py \
      --bin part-09-from-scratch/data/fineweb_edu_600M.bin \
      --shard-tokens 100000000 \
      --run-prefix runs/pretrain-30M-v1 \
      --train-secs 6000 \
      --manifest /tmp/pretrain_manifest.json
"""
import argparse
import json
import os
import subprocess
import tempfile

BUCKET = "EXAMPLE-BUCKET"
REGION = "eu-central-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("--train-secs", type=int, default=6000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--manifest", default="/tmp/pretrain_manifest.json")
    ap.add_argument("--fresh-start", action="store_true", default=True)
    args = ap.parse_args()

    total_bytes = os.path.getsize(args.bin)
    total_tokens = total_bytes // 2
    shard_bytes = args.shard_tokens * 2
    n_shards = (total_bytes + shard_bytes - 1) // shard_bytes
    print(f"{total_tokens:,} tokens -> {n_shards} shards of "
          f"{args.shard_tokens:,} tokens", flush=True)

    shard_keys = []
    with open(args.bin, "rb") as f:
        for i in range(n_shards):
            chunk = f.read(shard_bytes)
            if not chunk:
                break
            key = f"{args.run_prefix}/shards/s{i:02d}.bin"
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
                tf.write(chunk)
                tmp = tf.name
            uri = f"s3://{BUCKET}/{key}"
            subprocess.run(
                ["aws", "s3", "cp", tmp, uri, "--region", REGION],
                check=True, stdout=subprocess.DEVNULL)
            os.unlink(tmp)
            print(f"  uploaded {key} ({len(chunk)//2:,} tokens)", flush=True)
            shard_keys.append(key)

    manifest = {
        "runPrefix": args.run_prefix,
        "shardKeys": shard_keys,
        "trainSecsPerChunk": args.train_secs,
        "seqLen": args.seq_len,
        "vocab": 50257,
        "batchSize": args.batch_size,
        "freshStart": bool(args.fresh_start),
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest {args.manifest} ({len(shard_keys)} shards)",
          flush=True)


if __name__ == "__main__":
    main()
