"""Compare saved per-token logits from safe and unsafe executions."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("safe", type=Path)
    parser.add_argument("unsafe", type=Path)
    parser.add_argument("--token-index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    safe = torch.load(args.safe, map_location="cpu")[args.token_index]
    unsafe = torch.load(args.unsafe, map_location="cpu")[args.token_index]
    difference = (safe - unsafe).abs()
    result = {
        "token_index": args.token_index,
        "vocab_size": safe.numel(),
        "safe_argmax": safe.argmax().item(),
        "unsafe_argmax": unsafe.argmax().item(),
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "cosine_similarity": functional.cosine_similarity(
            safe.unsqueeze(0), unsafe.unsqueeze(0)
        ).item(),
        "elements_with_error": (difference != 0).sum().item(),
        "elements_over_0_125": (difference > 0.125).sum().item(),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
