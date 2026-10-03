# data/

This directory holds the local historical AIS CSV consumed by `ais_loader.py`
when the dashboard is run in **Historical AIS** mode.

## sample_ais.csv

`sample_ais.csv` is a **tiny synthetic TEST FIXTURE** used only for automated
tests and to let the Historical AIS mode start without an external download.

**It is NOT real AIS data.** The vessel IDs (`TEST001`…) make this obvious. Do
not present it in any demo as genuine historical AIS.

## Using real historical AIS data

To use genuine data, download a historical AIS CSV (for example from a public
open-data maritime authority) and either:

- replace `data/sample_ais.csv` with it, or
- point the loader at your file via the configured path.

The loader normalises common AIS column names automatically. See the
"Historical AIS Data Ingestion" section of the project README for the expected
schema and recognised column aliases.

No live AIS feed, API key, or network access is involved — ingestion is from a
local CSV only.
