from __future__ import annotations

import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import camelot
import pandas as pd
from dotenv import load_dotenv

import pdfplumber
from utils import (
    collapse_multirow_header,
    expand_page_spec,
    extract_numeric_and_unit,
    pdfplumber_extract_table,
    sanitize_text,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CLEAN_DIR = DATA_DIR / "clean"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
FINAL_DATASET_PATH = CLEAN_DIR / "ESG_KPIs_2023_2024.csv"
QA_SUMMARY_PATH = CLEAN_DIR / "QA_summary.txt"

BMW_2023_REPORT = BASE_DIR / "BMW-Group-Report-2023-en.pdf"
BMW_2024_REPORT = BASE_DIR / "BMW-Group-Report-2024-en.pdf"
BMW_2023_OVERVIEW = BASE_DIR / "BMW-Group-Sustainability-Overview-FY-2023.pdf"
BMW_2024_OVERVIEW = BASE_DIR / "ESG_Overview_FY-2024.pdf"

MANIFEST_COLUMNS = [
    "year",
    "pdf_path",
    "pages",
    "table_name",
    "pillar",
    "category",
    "notes",
]

TABLE_SPECS = [
    {
        "table_name": "ghg_totals_splits",
        "pillar": "Environment",
        "category": "Emissions",
        "notes": "Scope 1/2/3 totals + splits; market vs location",
        "keywords": ["scope", "co2", "greenhouse"],
    },
    {
        "table_name": "emissions_intensity",
        "pillar": "Environment",
        "category": "Emissions",
        "notes": "t CO2e per € revenue + per vehicle",
        "keywords": ["intensity", "per", "revenue"],
    },
    {
        "table_name": "electrification",
        "pillar": "Environment",
        "category": "Transition",
        "notes": "BEV/PHEV shares; charging network",
        "keywords": ["bev", "phev", "electrified", "charging"],
    },
    {
        "table_name": "energy_mix",
        "pillar": "Environment",
        "category": "Energy",
        "notes": "Total energy; fossil vs renewable; PPAs/EACs",
        "keywords": ["energy", "renewable", "purchasing", "ppa"],
    },
    {
        "table_name": "water",
        "pillar": "Environment",
        "category": "Water",
        "notes": "Total, at-risk, recycled/reused",
        "keywords": ["water", "withdrawal", "recycled"],
    },
    {
        "table_name": "waste_circularity",
        "pillar": "Environment",
        "category": "Circularity",
        "notes": "Waste totals; recovery; recycling indicators",
        "keywords": ["waste", "recycling", "circular"],
    },
    {
        "table_name": "eu_taxonomy",
        "pillar": "Governance",
        "category": "Finance",
        "notes": "Aligned revenue/CapEx/OpEx (% and absolute)",
        "keywords": ["taxonomy", "aligned", "capex", "opex"],
    },
    {
        "table_name": "people_safety_social",
        "pillar": "Social",
        "category": "People",
        "notes": "Headcount, training, accidents, diversity",
        "keywords": ["employees", "training", "diversity", "accident"],
    },
    {
        "table_name": "supply_chain_dd",
        "pillar": "Governance",
        "category": "SupplyChain",
        "notes": "Assessments, audits, remediation, grievances",
        "keywords": ["supply", "due diligence", "supplier", "audit"],
    },
]


@dataclass
class ManifestRow:
    year: str
    pdf_path: str
    pages: str
    table_name: str
    pillar: str
    category: str
    notes: str


@dataclass
class ExtractedTable:
    dataframe: pd.DataFrame
    source_page: str
    engine: str
    page_block: str


class QACollector:
    """Track QA metrics for console summary."""

    def __init__(self) -> None:
        self.pages: set[str] = set()
        self.tables: List[Tuple[str, str, int, str]] = []

    def register_pages(self, pages: Iterable[int]) -> None:
        for page in pages:
            self.pages.add(str(page))

    def register_table(self, table_name: str, source_page: str, row_count: int, engine: str) -> None:
        self.tables.append((table_name, source_page, row_count, engine))

    @property
    def table_count(self) -> int:
        return len(self.tables)


def load_manifest(path: Path) -> pd.DataFrame:
    """Read manifest CSV enforcing required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")
    manifest_df = pd.read_csv(path, dtype=str).fillna("")
    missing_columns = [col for col in MANIFEST_COLUMNS if col not in manifest_df.columns]
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")
    return manifest_df


def ensure_manifest_has_2023(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Augment manifest with 2023 rows derived from the overview PDF."""
    if (manifest_df["year"].astype(str) == "2023").any():
        return manifest_df
    if not BMW_2023_OVERVIEW.exists():
        print("WARNING: 2023 overview PDF missing; placeholder rows will be added.")
        generated_rows = [
            {
                "year": "2023",
                "pdf_path": "BMW-Group-Report-2023-en.pdf",
                "pages": "?",
                "table_name": spec["table_name"],
                "pillar": spec["pillar"],
                "category": spec["category"],
                "notes": spec["notes"],
            }
            for spec in TABLE_SPECS
        ]
        return pd.concat([manifest_df, pd.DataFrame(generated_rows)], ignore_index=True)

    generated_rows, todos = build_2023_manifest_rows()
    if todos:
        for todo in todos:
            print(f"TODO: {todo}")
    if generated_rows:
        manifest_df = pd.concat([manifest_df, pd.DataFrame(generated_rows)], ignore_index=True)
        manifest_df.to_csv(MANIFEST_PATH, index=False)
    return manifest_df


def build_2023_manifest_rows() -> Tuple[List[Dict[str, str]], List[str]]:
    """Generate manifest rows for 2023."""
    rows: List[Dict[str, str]] = []
    todos: List[str] = []
    keyword_mapping = {spec["table_name"]: spec["keywords"] for spec in TABLE_SPECS}
    notes_mapping = {spec["table_name"]: spec["notes"] for spec in TABLE_SPECS}
    pillar_mapping = {spec["table_name"]: spec["pillar"] for spec in TABLE_SPECS}
    category_mapping = {spec["table_name"]: spec["category"] for spec in TABLE_SPECS}
    page_hits: Dict[str, set[str]] = {spec["table_name"]: set() for spec in TABLE_SPECS}

    with pdfplumber.open(BMW_2023_OVERVIEW) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text_lower = text.lower()
            referenced_pages = re.findall(r"(?:p\.|page)\s*(\d{1,4})", text_lower)
            referenced_pages = [str(int(num)) for num in referenced_pages]  # drop leading zeros
            for table_name, keywords in keyword_mapping.items():
                if any(keyword in text_lower for keyword in keywords):
                    if referenced_pages:
                        page_hits[table_name].update(referenced_pages)
                    else:
                        todos.append(
                            f"2023 {table_name}: found keywords on overview page {page_number} but no report reference."
                        )

    for table_name in page_hits:
        pages = sorted(page_hits[table_name], key=lambda x: int(x))
        pages_str = ";".join(pages) if pages else "?"
        if pages_str == "?":
            todos.append(f"2023 {table_name}: add page numbers in manifest.csv")
        rows.append(
            {
                "year": "2023",
                "pdf_path": "BMW-Group-Report-2023-en.pdf",
                "pages": pages_str if pages_str else "?",
                "table_name": table_name,
                "pillar": pillar_mapping[table_name],
                "category": category_mapping[table_name],
                "notes": notes_mapping[table_name],
            }
        )
    return rows, todos


def parse_manifest_rows(manifest_df: pd.DataFrame) -> List[ManifestRow]:
    """Convert manifest rows into ManifestRow dataclasses."""
    manifest_rows: List[ManifestRow] = []
    for _, row in manifest_df.iterrows():
        manifest_rows.append(
            ManifestRow(
                year=str(row["year"]),
                pdf_path=str(row["pdf_path"]),
                pages=str(row["pages"]),
                table_name=str(row["table_name"]),
                pillar=str(row["pillar"]),
                category=str(row["category"]),
                notes=str(row["notes"]),
            )
        )
    return manifest_rows


def camelot_read(
    pdf_path: Path, pages: str, flavor: str = "lattice", strip_text: str = "\n"
) -> Optional[List[ExtractedTable]]:
    """Try to read tables with Camelot."""
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor=flavor, strip_text=strip_text)
    except Exception as exc:
        print(f"ERROR: Camelot {flavor} failed on {pdf_path.name} pages {pages}: {exc}")
        return None

    if tables.n == 0:
        return []

    results: List[ExtractedTable] = []
    for table in tables:
        df = collapse_multirow_header(table.df)
        df = df.applymap(sanitize_text)
        if df.empty:
            continue
        source_page = getattr(table, "page", pages)
        results.append(
            ExtractedTable(
                dataframe=df,
                source_page=str(source_page),
                engine=f"camelot-{flavor}",
                page_block=pages,
            )
        )
    return results


def fallback_pdfplumber(pdf_path: Path, page_block: str) -> List[ExtractedTable]:
    """Fallback extraction using pdfplumber for each explicit page."""
    tables: List[ExtractedTable] = []
    for page_number in expand_page_spec(page_block):
        df = pdfplumber_extract_table(pdf_path, page_number)
        if df.empty:
            continue
        tables.append(
            ExtractedTable(
                dataframe=df.applymap(sanitize_text),
                source_page=str(page_number),
                engine="pdfplumber",
                page_block=str(page_number),
            )
        )
    return tables


def split_header_unit(header: str) -> Tuple[str, Optional[str]]:
    """Split out terminal unit fragments from headers."""
    text = sanitize_text(header)
    if not text:
        return "", None
    match = re.match(r"(.+?)\s*\(([^()]+)\)$", text)
    if match:
        return sanitize_text(match.group(1)), sanitize_text(match.group(2))
    return text, None


def split_kpi_unit(kpi: str) -> Tuple[str, Optional[str]]:
    """Separate KPI label from trailing unit descriptors."""
    text = sanitize_text(kpi)
    if not text:
        return "", None
    match = re.match(r"(.+?)\s*\(([^()]+)\)$", text)
    if match:
        return sanitize_text(match.group(1)), sanitize_text(match.group(2))
    return text, None


def choose_kpi_column(df: pd.DataFrame) -> str:
    """Pick the column most likely to contain KPI labels."""
    candidate = df.columns[0]
    best_score = -1
    for column in df.columns:
        values = [sanitize_text(val) for val in df[column].tolist()]
        text_values = [val for val in values if val]
        if not text_values:
            continue
        numeric_count = sum(1 for val in text_values if re.fullmatch(r"[-+]?\d[\d\s.,]*", val))
        score = len(text_values) - numeric_count
        if score > best_score:
            best_score = score
            candidate = column
    return candidate


def build_units(*units: Optional[str]) -> str:
    """Combine unit fragments into a semicolon-separated string."""
    unique_units = OrderedDict()
    for unit in units:
        cleaned = sanitize_text(unit)
        if cleaned:
            unique_units[cleaned] = None
    return "; ".join(unique_units.keys())


def determine_scope(header_base: str, cell_note: Optional[str]) -> str:
    """Derive scope/boundary from header or cell annotations."""
    lower_segments = " ".join(filter(None, [header_base, cell_note])).lower()
    if "market" in lower_segments and "based" in lower_segments:
        return "market-based"
    if "location" in lower_segments and "based" in lower_segments:
        return "location-based"
    return sanitize_text(header_base)


def tidy_table(
    df: pd.DataFrame,
    manifest_row: ManifestRow,
    source_page: str,
) -> pd.DataFrame:
    """Reshape the extracted table into the unified long format."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Pillar",
                "Category",
                "KPI",
                "Unit",
                "Value",
                "Scope_Boundary",
                "Notes",
                "Source_Page",
                "Table_Name",
            ]
        )

    kpi_column = choose_kpi_column(df)
    value_columns = [col for col in df.columns if col != kpi_column]
    tidy_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        kpi_label_raw = sanitize_text(row.get(kpi_column, ""))
        if not kpi_label_raw:
            continue
        kpi_label, kpi_unit = split_kpi_unit(kpi_label_raw)
        for value_column in value_columns if value_columns else [kpi_column]:
            cell_value = sanitize_text(row.get(value_column, ""))
            if not cell_value:
                continue
            numeric_value, value_unit, extra_note = extract_numeric_and_unit(cell_value)
            header_base, header_unit = split_header_unit(value_column)
            scope_boundary = determine_scope(header_base, extra_note)
            combined_unit = build_units(kpi_unit, header_unit, value_unit)
            manifest_note = sanitize_text(manifest_row.notes)
            notes = [note for note in [manifest_note, extra_note if extra_note not in {scope_boundary, ""} else None] if note]
            value_out: object = numeric_value if numeric_value is not None else cell_value

            tidy_rows.append(
                {
                    "Year": manifest_row.year,
                    "Pillar": manifest_row.pillar,
                    "Category": manifest_row.category,
                    "KPI": kpi_label,
                    "Unit": combined_unit,
                    "Value": value_out,
                    "Scope_Boundary": scope_boundary,
                    "Notes": "; ".join(OrderedDict.fromkeys(notes)),
                    "Source_Page": sanitize_text(source_page),
                    "Table_Name": manifest_row.table_name,
                }
            )

    return pd.DataFrame(tidy_rows)


def safe_block_name(block: str) -> str:
    """Generate filesystem-friendly fragment."""
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", block)
    return sanitized.strip("_") or "pages"


def process_manifest_row(manifest_row: ManifestRow, qa: QACollector) -> List[pd.DataFrame]:
    """Run extraction for a single manifest row and return tidy tables."""
    pdf_path = BASE_DIR / manifest_row.pdf_path
    if not pdf_path.exists():
        print(f"WARNING: PDF not found for {manifest_row.pdf_path}; skipping {manifest_row.table_name}.")
        return []

    if not manifest_row.pages or manifest_row.pages.strip() == "?":
        print(f"TODO: Provide page numbers for {manifest_row.year} {manifest_row.table_name}")
        return []

    tidy_frames: List[pd.DataFrame] = []

    page_blocks = [block.strip() for block in str(manifest_row.pages).split(";") if block.strip()]

    for block in page_blocks:
        qa.register_pages(expand_page_spec(block))
        print(f"Processing {manifest_row.table_name} ({manifest_row.year}) pages {block} via Camelot lattice...")
        camelot_tables = camelot_read(pdf_path, block, flavor="lattice")

        if camelot_tables is None:
            camelot_tables = []

        if camelot_tables == []:
            print(f"No tables found with lattice on {block}; retrying Camelot stream.")
            camelot_tables = camelot_read(pdf_path, block, flavor="stream") or []

        extracted_tables = camelot_tables or []

        if not extracted_tables:
            print(f"Camelot failed; switching to pdfplumber on pages {block}.")
            extracted_tables = fallback_pdfplumber(pdf_path, block)

        if not extracted_tables:
            print(f"WARNING: No tables extracted for {manifest_row.table_name} on {block}.")
            continue

        for idx, table_result in enumerate(extracted_tables, start=1):
            qa.register_table(manifest_row.table_name, table_result.source_page, len(table_result.dataframe), table_result.engine)
            extracted_df = table_result.dataframe.copy()
            extracted_df["Year"] = manifest_row.year
            extracted_df["Pillar"] = manifest_row.pillar
            extracted_df["Category"] = manifest_row.category
            extracted_df["Source_Page"] = table_result.source_page
            extracted_df["Table_Name"] = manifest_row.table_name

            output_name = f"{manifest_row.year}_{manifest_row.table_name}_{safe_block_name(table_result.page_block)}_{idx}.csv"
            output_path = EXTRACTED_DIR / output_name
            extracted_df.to_csv(output_path, index=False)

            tidy_df = tidy_table(table_result.dataframe, manifest_row, table_result.source_page)
            if tidy_df.empty:
                continue
            tidy_frames.append(tidy_df)

    return tidy_frames


def summarise_final_dataset(final_df: pd.DataFrame) -> Dict[str, object]:
    """Create summary metrics for console and file output."""
    summary: Dict[str, object] = {}
    if final_df.empty:
        summary["row_count_per_year_category"] = {}
        summary["duplicate_year_kpi"] = []
        summary["samples"] = []
        return summary

    grouped = final_df.groupby(["Year", "Category"]).size().to_dict()
    duplicates_mask = final_df.duplicated(subset=["Year", "KPI"], keep=False)
    duplicates = final_df.loc[duplicates_mask, ["Year", "KPI", "Scope_Boundary"]].drop_duplicates().to_dict("records")
    samples = final_df.sample(n=min(10, len(final_df)), random_state=42).to_dict("records")

    summary["row_count_per_year_category"] = grouped
    summary["duplicate_year_kpi"] = duplicates
    summary["samples"] = samples
    return summary


def print_qa_summary(qa: QACollector, summary: Dict[str, object], final_df: pd.DataFrame) -> None:
    """Print QA summary to console."""
    pages_list = sorted(qa.pages, key=lambda x: int(re.sub(r"[^0-9]", "", x)) if re.search(r"\d", x) else x)
    print("\n=== QA SUMMARY ===")
    print(f"Pages processed ({len(pages_list)}): {', '.join(pages_list)}")
    print(f"Tables extracted: {qa.table_count}")
    for table_name, source_page, row_count, engine in qa.tables:
        print(f"  - {table_name} @ page {source_page} ({row_count} rows) via {engine}")

    if final_df.empty:
        print("No data in final dataset. Skipping further QA metrics.")
        return

    print("\nRow count per Year/Category:")
    for (year, category), count in summary["row_count_per_year_category"].items():
        print(f"  {year} / {category}: {count}")

    if summary["duplicate_year_kpi"]:
        print("\nDuplicate (Year, KPI) combinations:")
        for duplicate in summary["duplicate_year_kpi"]:
            print(f"  {duplicate}")
    else:
        print("\nNo duplicate (Year, KPI) combinations detected.")

    print("\nSample rows:")
    for sample in summary["samples"][:10]:
        print(f"  {sample}")


def write_qa_summary(summary: Dict[str, object]) -> None:
    """Persist QA summary into QA_summary.txt."""
    QA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QA_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = load_manifest(MANIFEST_PATH)
    manifest_df = ensure_manifest_has_2023(manifest_df)
    manifest_rows = parse_manifest_rows(manifest_df)

    qa_collector = QACollector()

    tidy_frames: List[pd.DataFrame] = []

    for manifest_row in manifest_rows:
        tidy_frames.extend(process_manifest_row(manifest_row, qa_collector))

    if tidy_frames:
        final_df = pd.concat(tidy_frames, ignore_index=True)
    else:
        final_df = pd.DataFrame(
            columns=[
                "Year",
                "Pillar",
                "Category",
                "KPI",
                "Unit",
                "Value",
                "Scope_Boundary",
                "Notes",
                "Source_Page",
                "Table_Name",
            ]
        )

    final_df.to_csv(FINAL_DATASET_PATH, index=False)
    qa_summary = summarise_final_dataset(final_df)
    qa_summary["pages_processed"] = sorted(list(qa_collector.pages), key=lambda x: int(re.sub(r"[^0-9]", "", x)) if re.search(r"\d", x) else x)
    qa_summary["tables_extracted"] = [
        {"table": table_name, "page": source_page, "rows": rows, "engine": engine} for table_name, source_page, rows, engine in qa_collector.tables
    ]

    print_qa_summary(qa_collector, qa_summary, final_df)
    write_qa_summary(qa_summary)


if __name__ == "__main__":
    main()
