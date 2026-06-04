# SportsMOT basketball subset (local extract)

This folder is populated by `scripts/extract_sportsmot_basketball.py`. It is **gitignored** (frames + GT are large).

## Extract from zip

```powershell
py scripts/extract_sportsmot_basketball.py --zip data/sportsmot_publish.zip
```

Preview without extracting:

```powershell
py scripts/extract_sportsmot_basketball.py --zip data/sportsmot_publish.zip --list-only
```

## If the zip is incomplete

If download was interrupted, `tar`/`zipfile` may fail mid-archive. Either:

1. Re-download `sportsmot_publish.zip`, or  
2. Fully unzip once, then:

```powershell
py scripts/extract_sportsmot_basketball.py --source-dir path/to/sportsmot_publish
```

## Layout after extract

```text
data/datasets/sportsmot_basketball/
├── manifest.json
├── splits_txt/basketball.txt
├── train/<sequence>/img1/, gt/gt.txt, seqinfo.ini
├── val/<sequence>/...
└── test/<sequence>/...   # optional (--include-test); no public GT
```

## Use with this repo

Copy one sequence into `data/datasets/sportsmot_example/` (see [sportsmot_example/README.md](../sportsmot_example/README.md)), then run the usual SAM3 + LSTM pipeline.

Only **basketball** clips are extracted (from official `splits_txt/basketball.txt`); football and volleyball are skipped.
