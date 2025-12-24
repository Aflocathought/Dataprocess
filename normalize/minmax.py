"""minmax

最大最小值归一化 （Parquet 文件 & Polars）。

关键要求：
- 当对多个文件一起归一化时，必须基于所有输入文件的拼接数据（而非单个文件）计算每列的最小值和最大值。

Usage:
  python normalize/minmax.py --inputs data/a.parquet data/b.parquet --output-dir out/
  python normalize/minmax.py --inputs "data/*.parquet" --output-dir out/ --stats-json out/stats.json
  python normalize/minmax.py --inputs data/a.parquet --output-dir out/ --columns voltage,current,temp

Notes:
- 归一化是对每一列独立应用的：
    x_norm = (x - min) / (max - min)
- 如果某列的最大值等于最小值（常数列），非空值将变为 0.0。
- 空值保持为空；NaN 保持为 NaN。
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class MinMax:
    min: float
    max: float

    @property
    def denom(self) -> float:
        return float(self.max - self.min)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Min-Max normalize Parquet columns")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input parquet files or glob patterns (quoted on Windows)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to write normalized parquet files",
    )
    p.add_argument(
        "--columns",
        default="",
        help="Comma-separated columns to normalize (default: all numeric columns)",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to output filename stem (default: empty)",
    )
    p.add_argument(
        "--stats-json",
        default="",
        help="Optional path to export computed min/max statistics as JSON",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return p.parse_args()


def _expand_inputs(raw_inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_inputs:
        # Support glob patterns on Windows where shell may not expand.
        if any(ch in raw for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(raw, recursive=True)]
            files.extend(matches)
        else:
            files.append(Path(raw))

    out: list[Path] = []
    for p in files:
        if not p.exists() or p.is_dir():
            raise SystemExit(f"Input parquet not found: {p}")
        out.append(p)

    # Dedup + stable order
    unique = sorted({p.resolve() for p in out}, key=lambda x: str(x).lower())
    if not unique:
        raise SystemExit("No input files matched.")
    return unique


def _is_numeric_for_norm(dtype: pl.DataType) -> bool:
    if dtype == pl.Boolean:
        return False
    try:
        return bool(dtype.is_numeric())
    except Exception:
        return False


def _pick_columns(schema: dict[str, pl.DataType], requested: list[str]) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in schema]
        if missing:
            raise SystemExit(f"Requested columns not in schema: {missing}")
        return requested

    return [name for name, dt in schema.items() if _is_numeric_for_norm(dt)]


def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        try:
            return lf.collect(streaming=True)
        except TypeError:
            return lf.collect()


def _compute_minmax(all_inputs: list[Path], columns: list[str]) -> dict[str, MinMax]:
    lf = pl.scan_parquet([str(p) for p in all_inputs])

    exprs: list[pl.Expr] = []
    for c in columns:
        v = pl.col(c).cast(pl.Float64)
        finite = v.is_finite()
        exprs.append(v.filter(finite).min().alias(f"{c}__min"))
        exprs.append(v.filter(finite).max().alias(f"{c}__max"))

    stats_df = _collect_streaming(lf.select(exprs))

    stats: dict[str, MinMax] = {}
    for c in columns:
        mn = stats_df[f"{c}__min"][0]
        mx = stats_df[f"{c}__max"][0]
        if mn is None or mx is None:
            # Column is all-null/all-NaN across all inputs.
            continue
        stats[c] = MinMax(min=float(mn), max=float(mx))

    return stats


def _normalize_expr(col: str, mm: MinMax) -> pl.Expr:
    v = pl.col(col).cast(pl.Float64)
    denom = mm.denom
    mn_lit = pl.lit(mm.min)
    denom_lit = pl.lit(denom)

    # Nulls stay null. NaNs will propagate through arithmetic.
    return (
        pl.when(v.is_null())
        .then(None)
        .when(pl.lit(denom == 0.0))
        .then(pl.lit(0.0))
        .otherwise((v - mn_lit) / denom_lit)
        .alias(col)
    )


def main() -> int:
    args = _parse_args()

    input_paths = _expand_inputs(args.inputs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use combined schema for column selection/validation
    base_lf = pl.scan_parquet([str(p) for p in input_paths])
    try:
        schema = base_lf.collect_schema()
    except AttributeError:
        schema = base_lf.schema

    requested_cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    cols = _pick_columns(schema, requested_cols)
    if not cols:
        raise SystemExit(
            "No columns to normalize. Pass --columns or ensure the file has numeric columns."
        )

    stats = _compute_minmax(input_paths, cols)
    skipped = [c for c in cols if c not in stats]
    cols_to_norm = [c for c in cols if c in stats]

    print(f"Inputs: {len(input_paths)}")
    print(f"Normalize columns: {len(cols_to_norm)}")
    if skipped:
        print(
            "Skipped (all-null/all-NaN across inputs): "
            + ", ".join(skipped)
        )

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "minmax",
            "inputs": [str(p) for p in input_paths],
            "columns_requested": requested_cols,
            "columns_normalized": cols_to_norm,
            "stats": {
                c: {"min": stats[c].min, "max": stats[c].max} for c in cols_to_norm
            },
        }
        stats_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Stats JSON: {stats_path}")

    suffix = str(args.suffix or "")

    for p in input_paths:
        out_name = f"{p.stem}{suffix}{p.suffix}"
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            raise SystemExit(
                f"Output already exists: {out_path} (pass --overwrite to replace)"
            )

        lf = pl.scan_parquet(str(p))
        exprs = [_normalize_expr(c, stats[c]) for c in cols_to_norm]
        lf2 = lf.with_columns(exprs)
        lf2.sink_parquet(str(out_path))
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
