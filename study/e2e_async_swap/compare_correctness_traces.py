"""Find the first token divergence and compare logits and swap history."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("safe", type=Path)
    parser.add_argument("unsafe", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def first_difference(left, right):
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def swaps_for_seq(result, seq_id):
    return [
        event for event in result["swap_trace"]
        if event["seq_id"] == seq_id
    ]


def main():
    args = parse_args()
    safe = json.loads(args.safe.read_text(encoding="utf-8"))
    unsafe = json.loads(args.unsafe.read_text(encoding="utf-8"))
    divergences = []
    for request_index, (safe_tokens, unsafe_tokens) in enumerate(zip(
        safe["request_output_token_ids"],
        unsafe["request_output_token_ids"],
    )):
        token_index = first_difference(safe_tokens, unsafe_tokens)
        if token_index is None:
            continue
        safe_seq_id = safe["request_seq_ids"][request_index]
        unsafe_seq_id = unsafe["request_seq_ids"][request_index]
        divergences.append({
            "request_index": request_index,
            "token_index": token_index,
            "safe_token": safe_tokens[token_index],
            "unsafe_token": unsafe_tokens[token_index],
            "safe_topk": safe["topk_trace"][str(safe_seq_id)][token_index],
            "unsafe_topk": unsafe["topk_trace"][str(unsafe_seq_id)][token_index],
            "safe_swap_history": swaps_for_seq(safe, safe_seq_id),
            "unsafe_swap_history": swaps_for_seq(unsafe, unsafe_seq_id),
        })
    result = {
        "matching_requests": len(safe["request_output_token_ids"]) - len(divergences),
        "total_requests": len(safe["request_output_token_ids"]),
        "divergences": divergences,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
