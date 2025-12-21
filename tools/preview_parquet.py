"""Preview Parquet content for quick sanity checks.

Usage:
  python scripts/preview_parquet.py --parquet out/file.parquet --head 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preview Parquet schema, head rows, and basic null stats"
    )
    p.add_argument("--parquet", required=True, help="Parquet file path")
    p.add_argument("--head", type=int, default=10, help="Rows to show")
    p.add_argument(
        "--columns",
        default="",
        help="Comma-separated column subset to display (optional)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.parquet)
    if not path.exists() or path.is_dir():
        raise SystemExit(f"Parquet file not found: {path}")

    lf = pl.scan_parquet(str(path))
    try:
        schema = lf.collect_schema()
    except AttributeError:
        schema = lf.schema

    print(f"Parquet: {path}")
    print("Schema:")
    for k, v in schema.items():
        print(f"  - {k}: {v}")

    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    if cols:
        missing = [c for c in cols if c not in schema]
        if missing:
            raise SystemExit(f"Requested columns not in schema: {missing}")
        lf2 = lf.select(cols)
    else:
        lf2 = lf

    print("\nHead:")
    print(lf2.head(args.head).collect())

    # Basic null stats (small compute; does not collect whole dataset)
    print("\nNull counts (first-level, streamed aggregation):")
    schema_keys = list(schema.keys())
    exprs = [pl.len().alias("__rows")] + [pl.col(c).null_count().alias(c) for c in schema_keys]
    try:
        stats = lf.select(exprs).collect(engine="streaming")
    except TypeError:
        stats = lf.select(exprs).collect(streaming=True)
    rows = int(stats["__rows"][0])
    print(f"rows={rows:,}")
    for c in schema_keys:
        n = int(stats[c][0])
        if n:
            print(f"  - {c}: {n:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
