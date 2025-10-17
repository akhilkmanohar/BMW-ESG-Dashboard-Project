# BMW ESG KPI Extraction Pipeline

This project assembles a repeatable Python workflow that extracts ESG KPIs from the BMW 2023–2024 sustainability PDFs and prepares a clean CSV that can be loaded into SQL databases or Power BI.

## 1. Environment Setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment:
   - **Windows (PowerShell)**  
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux (bash/zsh)**  
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Ghostscript (required by Camelot)

Camelot relies on Ghostscript to parse PDFs.

| OS      | Installation command / link                                           |
|---------|------------------------------------------------------------------------|
| Windows | Download and install from https://ghostscript.com/releases/gsdnld.html |
| macOS   | `brew install ghostscript` (via Homebrew)                              |
| Linux   | `sudo apt-get install ghostscript` (Debian/Ubuntu) or the equivalent   |

After installing Ghostscript, restart the terminal session to ensure it is on your `PATH`.

## 2. Configure Extraction Targets

The extraction manifest lives at `data/manifest.csv` and controls what pages are processed. Columns:

- `year`: Reporting year for the KPI rows.
- `pdf_path`: PDF filename in the project root.
- `pages`: Page or page ranges (e.g., `195-197`, `57;79;126`). Use `?` as a placeholder for unknown pages.
- `table_name`, `pillar`, `category`: Logical grouping metadata that flows into the outputs.
- `notes`: Free-form context or reminders (kept in downstream notes).

To add or adjust pages, edit the existing rows or append new ones—rerunning the pipeline will pick up the changes automatically. When 2023 pages are missing, the pipeline writes TODO reminders to the console so you can complete them later.

## 3. Run the Pipeline

With the virtual environment active:

```bash
python src/extract.py
```

The script will:

1. Ensure the 2023 manifest rows exist (auto-deriving them from the `BMW-Group-Sustainability-Overview-FY-2023.pdf` overview where possible).
2. Extract tables using Camelot (`lattice` first, then `stream`) and fall back to a pdfplumber text parser when Camelot finds no tables.
3. Normalise headers, clean number formats, and append metadata columns (`Year`, `Pillar`, `Category`, `Source_Page`, `Table_Name`).
4. Write raw per-table CSVs to `data/extracted/`.
5. Produce a tidy dataset at `data/clean/ESG_KPIs_2023_2024.csv`.
6. Print a QA summary (pages processed, table counts, row counts, duplicates, and sample rows) and save the same information to `data/clean/QA_summary.txt`.

Re-running the command overwrites previous outputs so the workflow stays deterministic.

## 4. Outputs & Next Steps

- `data/extracted/`: Raw tables with metadata columns per manifest entry.
- `data/clean/ESG_KPIs_2023_2024.csv`: Combined KPI table ready for SQL import or Power BI.
- `data/clean/QA_summary.txt`: JSON-formatted QA details (row counts, duplicates, samples).

### Importing to Power BI or SQL

1. Launch Power BI Desktop and choose **Get Data → Text/CSV**, then point to `data/clean/ESG_KPIs_2023_2024.csv`.
2. For SQL, copy the CSV and ingest it using your preferred loader (e.g., `COPY` for PostgreSQL or `BULK INSERT` for SQL Server). The CSV headers align with the manifest metadata, making column mapping straightforward.

If you adjust the manifest or want to incorporate new KPIs, edit `data/manifest.csv` and rerun `python src/extract.py`. Keep the PDFs in the project root so the script can resolve file paths without extra configuration.
