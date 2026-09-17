"""Measure overlap between large D2H copies and CUDA kernels in nsys SQLite."""

import argparse
import json
from pathlib import Path
import sqlite3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--min-copy-bytes", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    connection = sqlite3.connect(args.sqlite)
    query = """
        WITH copies AS (
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind = 2 AND bytes > ?
        ),
        overlaps AS (
            SELECT
                copies.start AS copy_start,
                min(copies.end, kernels.end)
                    - max(copies.start, kernels.start) AS overlap_ns,
                strings.value AS kernel
            FROM copies
            JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                ON kernels.start < copies.end
               AND kernels.end > copies.start
            JOIN StringIds AS strings
                ON strings.id = kernels.shortName
        )
        SELECT kernel, count(*), sum(overlap_ns)
        FROM overlaps
        GROUP BY kernel
        ORDER BY sum(overlap_ns) DESC
    """
    rows = connection.execute(query, (args.min_copy_bytes,)).fetchall()
    copy_count = connection.execute(
        """
        SELECT count(*)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = 2 AND bytes > ?
        """,
        (args.min_copy_bytes,),
    ).fetchone()[0]
    overlapped_copy_count = connection.execute(
        """
        SELECT count(DISTINCT copies.start)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copies
        JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernels
          ON kernels.start < copies.end AND kernels.end > copies.start
        WHERE copies.copyKind = 2 AND copies.bytes > ?
        """,
        (args.min_copy_bytes,),
    ).fetchone()[0]
    result = {
        "sqlite": args.sqlite.name,
        "min_copy_bytes": args.min_copy_bytes,
        "large_d2h_copies": copy_count,
        "overlapped_d2h_copies": overlapped_copy_count,
        "kernel_overlap_pairs": sum(row[1] for row in rows),
        "summed_overlap_ms": round(sum(row[2] for row in rows) / 1e6, 6),
        "top_overlapped_kernels": [
            {
                "kernel": name,
                "pairs": count,
                "overlap_ms": round(overlap_ns / 1e6, 6),
            }
            for name, count, overlap_ns in rows[:12]
        ],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
