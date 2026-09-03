# Data

## Sources (official government statistics)

| Source | Country | Coverage | License/terms |
|--------|---------|----------|---------------|
| U.S. Social Security Administration baby names | United States | 1880-2023, names with 5+ occurrences | Public domain (U.S. government work) |
| U.S. Census Bureau 2020 first names | United States | 2020, frequently occurring first names by sex | Public domain (U.S. government work) |
| INSEE "Base prénoms" | France | 1900-2024 | Open License 2.0 (Etalab) |

## How the dataset is built

`python fetch_multinational_data.py`

1. Downloads the INSEE parquet from data.gouv.fr (cached in `data/raw/`).
2. Parses the local SSA and Census files.
3. Normalizes names and merges per-name counts by sex across all sources.
4. Labels by statistical association: >=85% of recorded births female ->
   `girl-associated`, >=85% male -> `boy-associated`, otherwise `uncertain`.
5. Weights rows by frequency (names with 10k+ records get full weight).
6. Splits 80/10/10 partitioned by normalized name, so a name never appears
   in two splits.

The raw INSEE parquet is not committed; `data/raw/` is gitignored.

## Coverage limits

Official per-name statistics broken down by sex exist for only a handful of
countries. This build covers the United States and France. Most Asian and
African countries do not publish gendered name statistics at all (Taiwan and
Israel do, but their portals block automated download; Japan, South Korea,
China and India publish none). The fetcher is structured so an additional
source is one function returning `{name: (girl_count, boy_count)}`.

## Adding a source

Write a `_load_<country>()` function that returns
`{normalized_name: (girl_count, boy_count)}`, register it in `sources` in
`main()`, and rerun `python fetch_multinational_data.py`. The official
portals with usable bulk files are the best candidates:
U.K. ONS, Scotland NRS, Ireland CSO, Norway SSB, Sweden SCB (all block
simple HTTP clients; download the files by hand and convert them to the
same dict format).

## Dataset report

`data/dataset_report.json` records per-source unique-name counts, label
distribution, split sizes, and the total number of official birth records
covered.