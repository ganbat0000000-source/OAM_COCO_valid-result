# OAM to COCO Y0 Automation

This app provides 6 pages:

1. `Input Data`
2. `Correlation Matrix`
3. `Ranked Data`
4. `COCO Y0 Estimation`
5. `Validation`
6. `Result`

## Flow

1. Upload OAM CSV from the sidebar.
2. Review parsed objects/attributes/input sheet in `Input Data`.
3. Inspect attribute relationships and direction-rule warnings in `Correlation Matrix`.
4. Review the ranked OAM table in `Ranked Data` and run COCO Y0.
5. Review COCO Y0 tables/metrics in `COCO Y0 Estimation`.
6. Run inverse validation in `Validation`.
7. Review the final validated ranking in `Result`.

## Validation

The `Validation` page adds an inverse-run consistency check before the final result page.

Workflow:
- Start from the baseline ranked OAM table.
- Run COCO Y0 and collect the baseline `Delta/Fact` values.
- Build the inverse ranked table with the formula `inverse rank = n + 1 - rank`, where `n` is the number of objects.
- Run COCO Y0 again with the inverse ranked table.
- Compute the binary validation flag for each object:
  - `Validation = 1` if `baseline(Delta/Fact) * inverse(Delta/Fact) <= 0`
  - `Validation = 0` otherwise

Interpretation:
- `1` = valid
- `0` = invalid

The final `Result` page includes this validation output together with the baseline COCO ranking.

## Correlation Matrix

The `Correlation Matrix` page computes Pearson correlation across attribute columns (`X` data).

- Uses attribute display labels with units when available (for example: `Internet Usage 2010% [%]`).
- Shows only the lower triangle (upper triangle is muted gray/blank for readability).
- Color scale:
  - Diagonal (`1.0`): green
  - Positive correlation (`0` to `1`): yellow -> green
  - Negative correlation (`0` to `-1`): light red -> red
- Number formatting follows app rules:
  - 2 significant digits overall
  - No trailing zeros (example: `0.60` -> `0.6`)

## CSV Requirements

Required row labels in column 0:
- `Direction ID`
- `Attribute ID`
- `Attribute`

Rules:
- `Direction ID = 0`: higher value is better.
- `Direction ID = 1`: lower value is better.
- `Attribute Unit` row is optional but recommended (used in table headers).

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Requests + BeautifulSoup4
- OpenPyXL
- Matplotlib (optional, PNG export)

## Project Structure

```text
COCO_OAM_Automation/
  app.py
  README.md
  requirements.txt
  src/
    coco_client.py
    coco_parse.py
    oam_io.py
    ranking.py
    ui_display.py
```

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## COCO Dependency

This app submits to:
- `https://miau.my-x.hu/myx-free/coco/beker_y0.php`

Internet access is required for COCO runs.
