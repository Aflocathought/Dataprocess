"""detect_nan

检测 NaN (以及可选的 null) 连续行段在 Parquet 数据中的情况。

设计思路：
- Parquet-first: 通过 Polars 的 scan_parquet 读取。
- Console output: 打印前 N 个NaN连续段（默认 100 个）。
- Full visibility: 导出所有NaN连续段到 JSON 以便进一步检查。

Usage:
  python clean/detect_nan.py --parquet data/file.parquet
  python clean/detect_nan.py --parquet data/file.parquet --columns a,b,c
  python clean/detect_nan.py --parquet data/file.parquet --json-out out/nan_segments.json

Notes:
- "Contiguous" 基于 parquet 扫描中的行顺序（行索引）。
- 默认情况下，NaN 检测仅适用于浮点列。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class Segment:
    start_row: int
    end_row: int

    @property
    def length(self) -> int:
        return self.end_row - self.start_row + 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Detect contiguous row segments that contain NaN (optionally null) "
            "values in a Parquet file."
        )
    )
    p.add_argument("--parquet", required=True, help="Parquet file path")
    p.add_argument(
        "--columns",
        default="",
        help=(
            "Comma-separated columns to check. Default: all float columns in schema. "
            "If you pass non-float columns, they will be ignored unless --include-null is set."
        ),
    )
    p.add_argument(
        "--include-null",
        action="store_true",
        help="Treat nulls as hits as well (in addition to NaN).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max segments to print to console (default: 100)",
    )
    p.add_argument(
        "--json-out",
        default="",
        help=(
            "Write all segments + metadata to JSON. If omitted, only the first --limit "
            "segments are shown on console."
        ),
    )
    return p.parse_args()


def _contiguous_segments(sorted_rows: Iterable[int]) -> list[Segment]:
    it = iter(sorted_rows)
    try:
        first = next(it)
    except StopIteration:
        return []

    start = prev = int(first)
    segments: list[Segment] = []

    for raw in it:
        r = int(raw)
        if r == prev + 1:
            prev = r
            continue
        segments.append(Segment(start_row=start, end_row=prev))
        start = prev = r

    segments.append(Segment(start_row=start, end_row=prev))
    return segments


def _pick_columns_to_check(
    schema: dict[str, pl.DataType], requested: list[str], include_null: bool
) -> tuple[list[str], list[str]]:
    """Return (columns_to_check, ignored_columns).

    - NaN check applies only to float columns.
    - If include_null is True, we also allow any dtype for null check.
    """

    if requested:
        missing = [c for c in requested if c not in schema]
        if missing:
            raise SystemExit(f"Requested columns not in schema: {missing}")
        cols = requested
    else:
        cols = list(schema.keys())

    float_cols = [c for c in cols if schema[c] in (pl.Float32, pl.Float64)]

    if include_null:
        # Null can exist in any dtype; keep all requested columns.
        return cols, []

    ignored = [c for c in cols if c not in float_cols]
    return float_cols, ignored


def main() -> int:
    args = _parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists() or parquet_path.is_dir():
        raise SystemExit(f"Parquet file not found: {parquet_path}")

    base_lf = pl.scan_parquet(str(parquet_path))

    try:
        schema = base_lf.collect_schema()
    except AttributeError:
        schema = base_lf.schema

    lf = base_lf.with_row_index(name="__row")

    requested_cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    cols_to_check, ignored = _pick_columns_to_check(schema, requested_cols, args.include_null)

    if not cols_to_check:
        raise SystemExit(
            "No columns to check. "
            "If you didn't pass --columns, make sure the file has float columns; "
            "or pass --include-null to detect nulls on non-float columns."
        )

    exprs: list[pl.Expr] = []

    # NaN checks: only for float columns.
    float_cols = [c for c in cols_to_check if schema.get(c) in (pl.Float32, pl.Float64)]
    if float_cols:
        exprs.append(pl.any_horizontal([pl.col(c).is_nan() for c in float_cols]))

    # Null checks: any dtype.
    if args.include_null:
        exprs.append(pl.any_horizontal([pl.col(c).is_null() for c in cols_to_check]))

    if not exprs:
        raise SystemExit("No valid NaN/null checks could be constructed.")

    hit_expr = pl.any_horizontal(exprs).alias("__hit")

    hit_rows_df = (
        lf.select([pl.col("__row"), hit_expr])
        .filter(pl.col("__hit"))
        .select(pl.col("__row"))
        .collect()
    )

    hit_rows = hit_rows_df.get_column("__row").to_list()
    hit_rows.sort()

    segments = _contiguous_segments(hit_rows)

    print(f"Parquet: {parquet_path}")
    print(f"Columns checked ({len(cols_to_check)}): {cols_to_check}")
    if ignored:
        print(
            "Ignored (non-float; use --include-null to consider nulls): "
            f"{ignored}"
        )
    print(f"Hit rows: {len(hit_rows):,}")
    print(f"Contiguous segments: {len(segments):,}")

    if not segments:
        print("No NaN/null hits found.")
        return 0

    limit = int(args.limit)
    if limit < 0:
        limit = 0

    to_show = segments[:limit]

    print(f"\nFirst {min(limit, len(segments))} segments:")
    for i, seg in enumerate(to_show, start=1):
        print(
            f"{i:>3}. rows [{seg.start_row:,}..{seg.end_row:,}] "
            f"len={seg.length:,}"
        )

    if len(segments) > limit and not args.json_out:
        print(
            f"\n(Only showing first {limit} segments. "
            "Pass --json-out to export all segments.)"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "parquet": str(parquet_path),
            "columns_checked": cols_to_check,
            "include_null": bool(args.include_null),
            "hit_rows": len(hit_rows),
            "segment_count": len(segments),
            "segments": [asdict(s) | {"length": s.length} for s in segments],
        }

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
