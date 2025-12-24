# Dataprocess

一些批量处理数据的脚本

## 数据清理

- NaN 连续段落检测（Parquet）：`python clean/detect_nan.py --parquet your.parquet`
	- 默认只打印前 100 个连续段落；想看全部用 `--json-out out.json` 导出

## 归一化

- Min-Max 归一化（支持多文件一起算全局 min/max）：
	- `python normalize/minmax.py --inputs file1.parquet file2.parquet --output-dir out/`
	- 需要导出本次使用的 min/max：加 `--stats-json out/stats.json`
