from pathlib import Path
import csv
import re
import sqlite3
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
US_ROOT = PROCESSED_ROOT / "us_by_state"
INDIA_ROOT = PROCESSED_ROOT / "india_by_state"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "integrity_report.txt"
DB_FILE = OUTPUT_DIR / "integrity_ids.sqlite"


BASE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

VALID_STATE_COLUMNS = {"state", "statee"}

US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New_Hampshire", "New_Jersey", "New_Mexico",
    "New_York", "North_Carolina", "North_Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode_Island", "South_Carolina",
    "South_Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West_Virginia", "Wisconsin", "Wyoming",
    "District_of_Columbia",
}

INDIA_STATES = {
    "Andhra_Pradesh",
    "Arunachal_Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal_Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya_Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil_Nadu",
    "Telangana",
    "Tripura",
    "Uttar_Pradesh",
    "Uttarakhand",
    "West_Bengal",
}


# ============================================================
# HELPERS
# ============================================================

def clean_state_filename(state):
    return state.replace("_", " ").strip()


def classify_file(path: Path):

    name = path.stem

    # Ignore audit/recovery artifacts
    lower = path.name.lower()

    if lower == "us_recovery_report.csv":
        return None

    if lower.endswith("_before_recovery.csv"):
        return None

    # -------------------------
    # INDIA
    # -------------------------

    if "india_by_state" in path.parts:

        patterns = [
            r"^(train_source1)_India_(.+)$",
            r"^(train_source[23])_India_english_latin_(.+)$",
            r"^(train_source[23])_India_indian_script_(.+)$",
        ]

        for pattern in patterns:
            m = re.match(pattern, name, re.IGNORECASE)

            if m:
                source = m.group(1).lower()
                suffix = m.group(2)

                if suffix in {
                    "unknown",
                    "address_present_state_unknown",
                }:
                    return source, "India", "", "special"

                if suffix in INDIA_STATES:
                    return (
                        source,
                        "India",
                        clean_state_filename(suffix),
                        "state",
                    )

        # Review files
        if "address_present_state_unknown_review" in path.parts:
            m = re.match(
                r"^(train_source[23])_India_.+_address_present_state_unknown$",
                name,
                re.IGNORECASE,
            )

            if m:
                return (
                    m.group(1).lower(),
                    "India",
                    "",
                    "review",
                )

        return None

    # -------------------------
    # US
    # -------------------------

    if "us_by_state" in path.parts:

        m = re.match(
            r"^(train_source[123])_US_(.+)$",
            name,
            re.IGNORECASE,
        )

        if m:
            source = m.group(1).lower()
            suffix = m.group(2)

            if suffix in {
                "unknown",
                "address_present_state_unknown",
            }:
                return source, "US", "", "special"

            if suffix in US_STATES:
                return (
                    source,
                    "US",
                    clean_state_filename(suffix),
                    "state",
                )

        return None

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DEEP PROCESSED DATA INTEGRITY CHECK")
    print("=" * 80)

    files = []

    if US_ROOT.exists():
        files.extend(US_ROOT.rglob("*.csv"))

    if INDIA_ROOT.exists():
        files.extend(INDIA_ROOT.rglob("*.csv"))

    files = sorted(set(files))

    print(f"\nCSV files discovered: {len(files)}")

    # --------------------------------------------------------
    # SQLite database
    # --------------------------------------------------------

    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE entity_locations (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            PRIMARY KEY (source, entity_id)
        )
    """)

    conn.commit()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_rows = 0
    source_rows = Counter()
    country_rows = Counter()
    file_rows = Counter()

    duplicate_ids = []
    empty_ids = []
    malformed_rows = []
    bad_country = []
    bad_state = []
    state_mismatch = []
    classification_errors = []

    unique_states = defaultdict(set)

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for file_index, path in enumerate(files, start=1):

        info = classify_file(path)

        if info is None:
            continue

        source, expected_country, expected_state, category = info

        relative_path = str(path.relative_to(PROJECT_ROOT))

        print(
            f"[{file_index}/{len(files)}] "
            f"{relative_path}"
        )

        try:
            with open(
                path,
                "r",
                encoding="utf-8-sig",
                newline="",
                errors="replace",
            ) as f:

                reader = csv.reader(
                    f,
                    delimiter="|",
                )

                try:
                    header = next(reader)
                except StopIteration:
                    malformed_rows.append(
                        (relative_path, 0, "empty_file")
                    )
                    continue

                header = [
                    x.strip()
                    for x in header
                ]

                if len(header) != 5:
                    classification_errors.append(
                        (
                            relative_path,
                            "wrong_column_count",
                        )
                    )
                    continue

                if header[:4] != BASE_COLUMNS:
                    classification_errors.append(
                        (
                            relative_path,
                            f"bad_base_columns: {header}",
                        )
                    )
                    continue

                state_column = header[4]

                if state_column not in VALID_STATE_COLUMNS:
                    classification_errors.append(
                        (
                            relative_path,
                            f"bad_state_column: {state_column}",
                        )
                    )
                    continue

                state_index = 4

                for row_number, row in enumerate(
                    reader,
                    start=2,
                ):

                    total_rows += 1
                    source_rows[source] += 1
                    country_rows[expected_country] += 1
                    file_rows[relative_path] += 1

                    # ----------------------------------------
                    # Row shape
                    # ----------------------------------------

                    if len(row) != 5:
                        malformed_rows.append(
                            (
                                relative_path,
                                row_number,
                                f"expected_5_found_{len(row)}",
                            )
                        )
                        continue

                    entity_id = row[0].strip()
                    country = row[3].strip()
                    row_state = row[state_index].strip()

                    # ----------------------------------------
                    # Empty entity ID
                    # ----------------------------------------

                    if not entity_id:
                        empty_ids.append(
                            (
                                relative_path,
                                row_number,
                            )
                        )
                        continue

                    # ----------------------------------------
                    # Country validation
                    # ----------------------------------------

                    if country != expected_country:
                        bad_country.append(
                            (
                                relative_path,
                                row_number,
                                entity_id,
                                expected_country,
                                country,
                            )
                        )

                    # ----------------------------------------
                    # State validation
                    # ----------------------------------------

                    if category in {"state"}:

                        if not row_state:
                            bad_state.append(
                                (
                                    relative_path,
                                    row_number,
                                    entity_id,
                                    "empty_state",
                                )
                            )

                        elif row_state != expected_state:
                            state_mismatch.append(
                                (
                                    relative_path,
                                    row_number,
                                    entity_id,
                                    expected_state,
                                    row_state,
                                )
                            )

                        unique_states[expected_country].add(
                            expected_state
                        )

                    else:

                        # Special/review files should not claim
                        # a concrete state.
                        if row_state:
                            bad_state.append(
                                (
                                    relative_path,
                                    row_number,
                                    entity_id,
                                    "special_file_should_have_blank_state",
                                    row_state,
                                )
                            )

                    # ----------------------------------------
                    # GLOBAL DUPLICATE CHECK PER SOURCE
                    # ----------------------------------------

                    try:
                        cur.execute(
                            """
                            INSERT INTO entity_locations
                            (source, entity_id, file_path)
                            VALUES (?, ?, ?)
                            """,
                            (
                                source,
                                entity_id,
                                relative_path,
                            ),
                        )

                    except sqlite3.IntegrityError:

                        old = cur.execute(
                            """
                            SELECT file_path
                            FROM entity_locations
                            WHERE source = ?
                              AND entity_id = ?
                            """,
                            (
                                source,
                                entity_id,
                            ),
                        ).fetchone()

                        duplicate_ids.append(
                            (
                                source,
                                entity_id,
                                old[0] if old else "",
                                relative_path,
                            )
                        )

                    # Commit periodically
                    if total_rows % 100000 == 0:
                        conn.commit()

        except Exception as e:

            classification_errors.append(
                (
                    relative_path,
                    f"file_read_error: {e}",
                )
            )

    conn.commit()

    # ========================================================
    # WRITE REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "DEEP PROCESSED DATA INTEGRITY REPORT\n"
        )
        report.write("=" * 80 + "\n\n")

        report.write(
            f"Total CSV files discovered: {len(files)}\n"
        )

        report.write(
            f"Total data rows scanned: {total_rows:,}\n\n"
        )

        report.write("ROWS BY SOURCE\n")
        report.write("-" * 40 + "\n")

        for source, count in sorted(source_rows.items()):
            report.write(
                f"{source}: {count:,}\n"
            )

        report.write("\nROWS BY COUNTRY\n")
        report.write("-" * 40 + "\n")

        for country, count in sorted(country_rows.items()):
            report.write(
                f"{country}: {count:,}\n"
            )

        report.write("\nDUPLICATE ENTITY IDS\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(duplicate_ids):,}\n"
        )

        for item in duplicate_ids[:100]:
            report.write(
                f"{item}\n"
            )

        report.write("\nEMPTY ENTITY IDS\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(empty_ids):,}\n"
        )

        report.write("\nMALFORMED ROWS\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(malformed_rows):,}\n"
        )

        report.write("\nBAD COUNTRY VALUES\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(bad_country):,}\n"
        )

        report.write("\nBAD STATE VALUES\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(bad_state):,}\n"
        )

        report.write("\nSTATE/FILENAME MISMATCHES\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(state_mismatch):,}\n"
        )

        report.write("\nCLASSIFICATION/READ ERRORS\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Count: {len(classification_errors):,}\n"
        )

        report.write("\nSTATES FOUND\n")
        report.write("-" * 40 + "\n")

        for country in sorted(unique_states):
            report.write(
                f"\n{country}:\n"
            )

            for state in sorted(
                unique_states[country]
            ):
                report.write(
                    f"  {state}\n"
                )

        # ----------------------------------------------------
        # VERDICT
        # ----------------------------------------------------

        problems = (
            len(duplicate_ids)
            + len(empty_ids)
            + len(malformed_rows)
            + len(bad_country)
            + len(bad_state)
            + len(state_mismatch)
            + len(classification_errors)
        )

        report.write("\n")
        report.write("=" * 80 + "\n")
        report.write("FINAL VERDICT\n")
        report.write("=" * 80 + "\n")

        if problems == 0:
            report.write(
                "PASS - No integrity violations found.\n"
            )
        else:
            report.write(
                f"FAIL - {problems:,} integrity violations found.\n"
            )

    conn.close()

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("INTEGRITY SUMMARY")
    print("=" * 80)

    print(f"\nTotal rows scanned       : {total_rows:,}")

    print("\nRows by source:")
    for source, count in sorted(source_rows.items()):
        print(
            f"  {source}: {count:,}"
        )

    print("\nRows by country:")
    for country, count in sorted(country_rows.items()):
        print(
            f"  {country}: {count:,}"
        )

    print("\nChecks:")
    print(
        f"  Duplicate IDs          : {len(duplicate_ids):,}"
    )
    print(
        f"  Empty IDs              : {len(empty_ids):,}"
    )
    print(
        f"  Malformed rows         : {len(malformed_rows):,}"
    )
    print(
        f"  Bad country values     : {len(bad_country):,}"
    )
    print(
        f"  Bad state values       : {len(bad_state):,}"
    )
    print(
        f"  State mismatches       : {len(state_mismatch):,}"
    )
    print(
        f"  Classification errors  : {len(classification_errors):,}"
    )

    problems = (
        len(duplicate_ids)
        + len(empty_ids)
        + len(malformed_rows)
        + len(bad_country)
        + len(bad_state)
        + len(state_mismatch)
        + len(classification_errors)
    )

    print("\n" + "=" * 80)

    if problems == 0:
        print(
            "✅ PASS - DATA INTEGRITY CHECK PASSED"
        )
    else:
        print(
            f"❌ FAIL - {problems:,} problems found"
        )

    print("=" * 80)

    print(
        f"\nDetailed report:"
        f"\n{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()