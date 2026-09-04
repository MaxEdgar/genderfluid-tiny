# Data

## Sources (official government statistics)

| Source | Country | Coverage | License/terms |
|--------|---------|----------|---------------|
| U.S. Social Security Administration baby names | United States | 1880-2020, names with 5+ occurrences | Public domain (U.S. government work) |
| U.S. Census Bureau 2020 first names | United States | 2020, frequently occurring first names by sex | Public domain (U.S. government work) |
| INSEE "Base prénoms" | France | 1900-2024 | Open License 2.0 (Etalab) |
| Meiji Yasuda Life newborn name survey | Japan | 2025 edition, kanji names labeled male/female/mixed | Public website; survey facts used, no raw counts published |
| Ministry of Public Security annual name reports | China | 2018-2021, top newborn names by sex (all four published reports) | Official government statistics via state media (People's Daily, China News, Xinhua, China Daily) |

## How the dataset is built

`python fetch_multinational_data.py`

1. Downloads the INSEE parquet from data.gouv.fr (cached in `data/raw/`).
2. Tries the full official SSA `names.zip` (all years, ~1M unique spellings;
   cached in `data/raw/ssa_names.zip`). Falls back to the aggregated local
   SSA file if the download fails.
3. Parses the local Census file.
4. Downloads the Japan Meiji Yasuda name index and reads the China MPS
   name lists embedded in the fetcher.
5. Normalizes names and merges per-name counts by sex across all sources.
6. Labels by statistical association: >=85% of recorded births female ->
   `girl-associated`, >=85% male -> `boy-associated`, otherwise `uncertain`.
7. Weights rows by frequency (names with 10k+ records get full weight).
8. Splits 80/10/10 partitioned by normalized name, so a name never appears
   in two splits.

Japan and China contribute no raw birth counts (the survey and the reports
publish associations, not counts), so they use synthetic counts that
preserve the reported association: 100 for single-gender names, 50/50 for
mixed-gender names (which the threshold maps to `uncertain`).

The raw INSEE parquet and SSA zip are not committed; `data/raw/` is
gitignored.

In CI, the `prepare` job of `.github/workflows/train.yml` runs the fetcher
on a GitHub runner before training, so the full official SSA national file
is used whenever it is reachable from that network.

## Coverage limits

Official per-name statistics broken down by sex exist for only a handful of
countries. This build covers the United States, France, Japan, and China.
Most other Asian and African countries do not publish gendered name
statistics at all. Taiwan and Israel publish them but their portals block
automated download; South Korea, India, Indonesia, Malaysia, and Singapore
publish none in machine-readable form (Singapore's ICA popular-baby-names
page has been removed; Indonesia's data.go.id and the Nordic statistical
APIs reject scripted clients). The fetcher is structured so an additional
source is one function returning `{name: (girl_count, boy_count)}`.

## Adding a source

Write a `_load_<country>()` function that returns
`{normalized_name: (girl_count, boy_count)}`, register it in `sources` in
`main()`, and rerun `python fetch_multinational_data.py`. The official
portals with usable bulk files are the best candidates:
U.K. ONS, Scotland NRS, Ireland CSO, Norway SSB, Sweden SCB (all block
simple HTTP clients; download the files by hand and convert them to the
same dict format).

## Hanzi and CJK support

Names in Chinese characters flow through the same pipeline as Latin names:
normalization uses NFKC (which folds fullwidth Latin produced by CJK input
methods while preserving hanzi and kana), and feature extraction operates on
Unicode code points, so hanzi n-grams hash correctly. The Japan and China
sources are what give the model actual CJK training signal; without them a
hanzi name would still run, but the prediction would be based on no
relevant training data.

## Dataset report

`data/dataset_report.json` records per-source unique-name counts, label
distribution, split sizes, and the total number of official birth records
covered.