"""csv2parquet

鲁棒性强，内存高效的 CSV -> Parquet 转换，适用于超大文件。

设计目标：
- 防止内存溢出：绝不将整个 CSV 文件加载到内存中。
- 默认无损：绝不因转换失败而将非空字符串默默转换为 null。
- 类型安全：将数值类型提升为 64 位，仅在安全时解析日期时间。

用法：
  python src/convert/csv2parquet.py --input_file data/input.csv --output_file data/output.parquet
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


LOGGER = logging.getLogger("csv2parquet")


INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(
	r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)

FLOAT_SPECIAL_TOKENS = {
	"nan",
	"+nan",
	"-nan",
	"inf",
	"+inf",
	"-inf",
	"infinity",
	"+infinity",
	"-infinity",
}

# Common date/time formats; keep list small and practical.
DATETIME_FORMATS: list[tuple[str, str]] = [
	("date", "%Y-%m-%d"),
	("date", "%Y/%m/%d"),
	("date", "%m/%d/%Y"),
	("date", "%d/%m/%Y"),
	("datetime", "%Y-%m-%d %H:%M:%S"),
	("datetime", "%Y/%m/%d %H:%M:%S"),
	("datetime", "%Y-%m-%d %H:%M:%S%.f"),
	("datetime", "%Y/%m/%d %H:%M:%S%.f"),
	("datetime", "%Y-%m-%dT%H:%M:%S"),
	("datetime", "%Y-%m-%dT%H:%M:%S%.f"),
]


@dataclass(frozen=True)
class CastPlan:
	# kind: "int64" | "float64" | "date" | "datetime"
	kind: str
	fmt: str | None = None


def _configure_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)


def _human_bytes(num: int | float) -> str:
	unit = ["B", "KB", "MB", "GB", "TB"]
	n = float(num)
	for u in unit:
		if n < 1024 or u == unit[-1]:
			return f"{n:.2f} {u}"
		n /= 1024
	return f"{n:.2f} TB"


def _read_first_row(
	input_file: Path, *, delimiter: str, encoding: str, newline: str = ""
) -> list[str]:
	with input_file.open("r", encoding=encoding, newline=newline) as f:
		reader = csv.reader(f, delimiter=delimiter)
		for row in reader:
			return row
	return []


def _normalize_header(header: list[str]) -> list[str]:
	# Ensure unique, non-empty column names.
	seen: dict[str, int] = {}
	out: list[str] = []
	for idx, raw in enumerate(header):
		name = (raw or "").strip() or f"column_{idx+1}"
		if name in seen:
			seen[name] += 1
			name = f"{name}_{seen[name]}"
		else:
			seen[name] = 0
		out.append(name)
	return out


def _iter_sample_rows(
	input_file: Path,
	*,
	delimiter: str,
	encoding: str,
	has_header: bool,
	sample_rows: int,
) -> Iterable[list[str]]:
	if sample_rows <= 0:
		return []

	def _gen() -> Iterable[list[str]]:
		with input_file.open("r", encoding=encoding, newline="") as f:
			reader = csv.reader(f, delimiter=delimiter)
			if has_header:
				next(reader, None)
			for i, row in enumerate(reader):
				if i >= sample_rows:
					break
				yield row

	return _gen()


def _pick_datetime_format(values: list[str]) -> CastPlan | None:
	non_empty = [v for v in values if v != ""]
	if not non_empty:
		return None

	for kind, fmt in DATETIME_FORMATS:
		target = pl.Date if kind == "date" else pl.Datetime
		parsed = pl.Series(non_empty).str.strptime(target, format=fmt, strict=False)
		if parsed.null_count() == 0:
			return CastPlan(kind=kind, fmt=fmt)
	return None


def _infer_candidates_from_sample(
	header: list[str], sample: list[list[str]]
) -> dict[str, CastPlan]:
	# Lightweight heuristics on a small sample to decide what to validate.
	# We still do a full-file validation pass (streaming) before casting.
	col_values: dict[str, list[str]] = {name: [] for name in header}
	for row in sample:
		for i, name in enumerate(header):
			if i < len(row):
				col_values[name].append(row[i].strip())
			else:
				col_values[name].append("")

	candidates: dict[str, CastPlan] = {}
	for name, values in col_values.items():
		non_empty = [v for v in values if v != ""]
		if not non_empty:
			continue

		def _is_floatish(v: str) -> bool:
			vl = v.lower()
			return bool(FLOAT_RE.match(v)) or vl in FLOAT_SPECIAL_TOKENS

		if all(INT_RE.match(v) for v in non_empty):
			candidates[name] = CastPlan(kind="int64")
			continue

		if all(_is_floatish(v) for v in non_empty):
			candidates[name] = CastPlan(kind="float64")
			continue

		dt_plan = _pick_datetime_format(values)
		if dt_plan is not None:
			candidates[name] = dt_plan

	return candidates


def _scan_csv_all_strings(
	*,
	input_file: Path,
	delimiter: str,
	encoding: str,
	has_header: bool,
	columns: list[str],
	infer_schema_length: int,
) -> pl.LazyFrame:
	overrides = {c: pl.Utf8 for c in columns}
	return pl.scan_csv(
		source=str(input_file),
		separator=delimiter,
		has_header=has_header,
		new_columns=None if has_header else columns,
		encoding=encoding,
		schema_overrides=overrides,
		infer_schema_length=infer_schema_length,
		ignore_errors=False,
		try_parse_dates=False,
		low_memory=True,
	)


def _collect_streaming(df: pl.LazyFrame) -> pl.DataFrame:
	# Polars streaming collect API has changed across versions; be defensive.
	try:
		return df.collect(engine="streaming")
	except TypeError:
		try:
			return df.collect(streaming=True)
		except TypeError:
			return df.collect()


def _build_validation_expr(col: str, plan: CastPlan) -> pl.Expr:
	s = pl.col(col)
	if plan.kind == "int64":
		parsed = s.cast(pl.Int64, strict=False)
		failures = (s.ne("") & parsed.is_null()).cast(pl.UInt64).sum()
		return failures.alias(f"__fail_int64__{col}")
	if plan.kind == "float64":
		parsed = s.cast(pl.Float64, strict=False)
		failures = (s.ne("") & parsed.is_null()).cast(pl.UInt64).sum()
		return failures.alias(f"__fail_float64__{col}")
	if plan.kind in {"date", "datetime"}:
		target = pl.Date if plan.kind == "date" else pl.Datetime
		parsed = s.str.strptime(target, format=plan.fmt, strict=False)
		failures = (s.ne("") & parsed.is_null()).cast(pl.UInt64).sum()
		return failures.alias(f"__fail_{plan.kind}__{col}")
	raise ValueError(f"Unknown cast kind: {plan.kind}")


def _apply_casts(
	lf: pl.LazyFrame, casts: dict[str, CastPlan], *, strict_cast: bool
) -> pl.LazyFrame:
	exprs: list[pl.Expr] = []
	for col, plan in casts.items():
		if plan.kind == "int64":
			exprs.append(pl.col(col).cast(pl.Int64, strict=strict_cast).alias(col))
		elif plan.kind == "float64":
			exprs.append(pl.col(col).cast(pl.Float64, strict=strict_cast).alias(col))
		elif plan.kind in {"date", "datetime"}:
			target = pl.Date if plan.kind == "date" else pl.Datetime
			exprs.append(
				pl.col(col)
				.str.strptime(target, format=plan.fmt, strict=strict_cast)
				.alias(col)
			)
		else:
			raise ValueError(f"Unknown cast kind: {plan.kind}")
	if not exprs:
		return lf
	return lf.with_columns(exprs)


def _ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Stream large CSV -> Parquet (lossless, OOM-safe)",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--input_file", required=True, help="Input CSV path")
	p.add_argument("--output_file", required=True, help="Output Parquet path")

	p.add_argument("--delimiter", default=",", help="CSV delimiter")
	p.add_argument(
		"--encoding", default="utf8", help="CSV text encoding (Polars encoding)"
	)
	p.add_argument(
		"--no_header",
		action="store_true",
		help="Treat CSV as having no header row",
	)

	p.add_argument(
		"--sample_rows",
		type=int,
		default=50_000,
		help="Rows to sample for candidate type detection",
	)
	p.add_argument(
		"--infer_schema_length",
		type=int,
		default=10_000,
		help=(
			"Rows Polars may use internally for CSV schema inference. "
			"(We still override all columns to strings for lossless safety.)"
		),
	)

	p.add_argument(
		"--parse_datetimes",
		choices=["auto", "off"],
		default="auto",
		help=(
			"auto: parse standard date/time columns only if safe; "
			"off: keep all date/time columns as strings"
		),
	)
	p.add_argument(
		"--validate_full_file",
		"--validate-full-file",
		default=True,
		action=argparse.BooleanOptionalAction,
		help=(
			"Validate candidate casts on the full file (streaming) before writing. "
			"This prevents turning bad values into nulls."
		),
	)

	p.add_argument(
		"--strict",
		action="store_true",
		help=(
			"Fail the run if any candidate cast would introduce nulls; "
			"otherwise (default) keep that column as string."
		),
	)

	p.add_argument(
		"--compression",
		default="zstd",
		choices=["zstd", "snappy", "gzip", "lz4", "uncompressed"],
		help="Parquet compression codec",
	)
	p.add_argument(
		"--row_group_size",
		type=int,
		default=1_000_000,
		help="Target Parquet row group size (rows)",
	)

	p.add_argument("--verbose", action="store_true", help="Enable debug logging")
	return p.parse_args(argv)


def main(argv: list[str]) -> int:
	args = parse_args(argv)
	_configure_logging(args.verbose)

	input_file = Path(args.input_file)
	output_file = Path(args.output_file)
	if not input_file.exists():
		LOGGER.error("Input file not found: %s", input_file)
		return 2
	if input_file.is_dir():
		LOGGER.error("Input path is a directory: %s", input_file)
		return 2

	_ensure_parent_dir(output_file)
	input_size = input_file.stat().st_size
	LOGGER.info("Input: %s (%s)", input_file, _human_bytes(input_size))
	LOGGER.info("Output: %s", output_file)

	has_header = not args.no_header
	t0 = time.perf_counter()

	LOGGER.info("Reading header and sampling for candidate types...")
	first_row = _read_first_row(
		input_file, delimiter=args.delimiter, encoding=args.encoding
	)
	if not first_row:
		LOGGER.error("CSV appears to be empty: %s", input_file)
		return 2

	if has_header:
		columns = _normalize_header(first_row)
	else:
		columns = [f"column_{i+1}" for i in range(len(first_row))]

	sample_rows = []
	for row in _iter_sample_rows(
		input_file,
		delimiter=args.delimiter,
		encoding=args.encoding,
		has_header=has_header,
		sample_rows=args.sample_rows,
	):
		sample_rows.append(row)

	candidates = _infer_candidates_from_sample(columns, sample_rows)
	if args.parse_datetimes == "off":
		candidates = {k: v for k, v in candidates.items() if v.kind not in {"date", "datetime"}}

	LOGGER.info(
		"Candidate casts (from sample): %d (ints=%d, floats=%d, datetimes=%d)",
		len(candidates),
		sum(1 for v in candidates.values() if v.kind == "int64"),
		sum(1 for v in candidates.values() if v.kind == "float64"),
		sum(1 for v in candidates.values() if v.kind in {"date", "datetime"}),
	)

	LOGGER.info("Building streaming scan (all columns as strings for lossless safety)...")
	lf = _scan_csv_all_strings(
		input_file=input_file,
		delimiter=args.delimiter,
		encoding=args.encoding,
		has_header=has_header,
		columns=columns,
		infer_schema_length=args.infer_schema_length,
	)

	final_casts: dict[str, CastPlan] = {}
	if candidates and args.validate_full_file:
		LOGGER.info("Validating candidate casts over the full file (streaming pass 1/2)...")
		exprs = [pl.len().alias("__rows")] + [
			_build_validation_expr(col, plan) for col, plan in candidates.items()
		]
		stats = _collect_streaming(lf.select(exprs))
		row_count = int(stats["__rows"][0])
		LOGGER.info("Row count (streamed): %s", f"{row_count:,}")

		for col, plan in candidates.items():
			if plan.kind == "int64":
				key = f"__fail_int64__{col}"
			elif plan.kind == "float64":
				key = f"__fail_float64__{col}"
			elif plan.kind in {"date", "datetime"}:
				key = f"__fail_{plan.kind}__{col}"
			else:
				continue

			failures = int(stats[key][0])
			if failures == 0:
				final_casts[col] = plan
			else:
				msg = (
					f"Column '{col}' has {failures:,} values that would fail {plan.kind} "
					"parsing."
				)
				if args.strict:
					LOGGER.error(msg)
					LOGGER.error(
						"Strict mode is on, aborting to avoid data loss. "
						"Re-run without --strict to keep the column as string."
					)
					return 2
				LOGGER.warning("%s Keeping as string (lossless default).", msg)
	else:
		if candidates:
			LOGGER.warning(
				"Skipping full-file validation; casts may introduce nulls on bad values."
			)
		final_casts = candidates

	if final_casts:
		LOGGER.info("Applying safe casts (streaming pass 2/2)...")
		# If we validated against the full file, strict casts are safe.
		# If we didn't validate, avoid crashing by using non-strict casts
		# (may introduce nulls on unexpected dirty values).
		lf_out = _apply_casts(lf, final_casts, strict_cast=bool(args.validate_full_file))
	else:
		LOGGER.info("No safe casts detected; writing all columns as strings.")
		lf_out = lf

	if output_file.exists():
		# Avoid appending/overwriting silently in some environments.
		output_file.unlink()

	LOGGER.info(
		"Writing Parquet (compression=%s, row_group_size=%s)...",
		args.compression,
		f"{args.row_group_size:,}",
	)
	lf_out.sink_parquet(
		str(output_file),
		compression=args.compression,
		statistics=True,
		row_group_size=args.row_group_size,
	)

	elapsed = time.perf_counter() - t0
	out_size = output_file.stat().st_size if output_file.exists() else 0
	ratio = (out_size / input_size) if input_size else 0.0
	LOGGER.info("Done in %.2fs", elapsed)
	LOGGER.info("Output size: %s", _human_bytes(out_size))
	LOGGER.info("Size ratio (out/in): %.3f", ratio)
	if out_size and input_size:
		pct = abs(1.0 - ratio) * 100.0
		if ratio <= 1.0:
			LOGGER.info("Reduction: %.2f%%", pct)
		else:
			LOGGER.info("Increase (small-file overhead): %.2f%%", pct)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
