from pathlib import Path
import csv
import sqlite3
import sys

import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\sanja\Downloads\amazon_ml"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "features"
    / "normalized"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE = (
    OUTPUT_DIR
    / "normalization_qa_report.txt"
)

DB_FILE = (
    OUTPUT_DIR
    / "normalization_qa_ids.sqlite"
)


# ============================================================
# FILES
# ============================================================

SOURCES = {
    "train_source1": {
        "raw": RAW_DIR / "train_source1.tsv",
        "normalized": NORMALIZED_DIR / "train_source1.parquet",
    },

    "train_source2": {
        "raw": RAW_DIR / "train_source2.tsv",
        "normalized": NORMALIZED_DIR / "train_source2.parquet",
    },

    "train_source3": {
        "raw": RAW_DIR / "train_source3.tsv",
        "normalized": NORMALIZED_DIR / "train_source3.parquet",
    },
}


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",

    "name_norm",
    "name_core",
    "name_translit",

    "address_norm",
    "address_translit",

    "name_script",
    "address_script",
    "script_type",

    "postal_code",
    "house_number",
    "phone",
    "domain",

    "country_norm",
    "source",
]


# ============================================================
# RAW ROW COUNT
# ============================================================

def count_raw_rows(path):

    count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.reader(
            f,
            delimiter="\t",
        )

        # Header
        next(reader)

        for _ in reader:
            count += 1

    return count


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

def create_database():

    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(
        DB_FILE
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE ids (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (source, entity_id)
        )
    """)

    conn.commit()

    return conn


# ============================================================
# INSERT IDS IN BATCHES
# ============================================================

def insert_ids(
    conn,
    source,
    ids,
):

    cur = conn.cursor()

    rows = [
        (
            source,
            str(entity_id),
        )
        for entity_id in ids
        if str(entity_id).strip()
    ]

    if not rows:
        return 0

    before = conn.total_changes

    cur.executemany(
        """
        INSERT OR IGNORE INTO ids
        (source, entity_id)
        VALUES (?, ?)
        """,
        rows,
    )

    conn.commit()

    inserted = (
        conn.total_changes
        - before
    )

    return inserted


# ============================================================
# CHECK ONE PARQUET
# ============================================================

def check_normalized_file(
    source,
    raw_path,
    parquet_path,
    conn,
):

    print("\n" + "=" * 80)
    print(f"CHECKING {source}")
    print("=" * 80)

    result = {
        "source": source,
        "raw_rows": 0,
        "parquet_rows": 0,
        "row_count_match": False,
        "duplicate_ids": 0,
        "empty_ids": 0,
        "wrong_source": 0,
        "missing_values": {},
        "schema_ok": False,
        "indian_script_rows": 0,
        "latin_rows": 0,
        "mixed_rows": 0,
        "sample": [],
    }

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file missing:\n{raw_path}"
        )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Normalized Parquet missing:\n{parquet_path}"
        )

    # --------------------------------------------------------
    # Raw row count
    # --------------------------------------------------------

    print("Counting raw rows...")

    raw_rows = count_raw_rows(
        raw_path
    )

    result["raw_rows"] = raw_rows

    print(
        f"Raw rows: {raw_rows:,}"
    )

    # --------------------------------------------------------
    # Open Parquet
    # --------------------------------------------------------

    parquet = pq.ParquetFile(
        parquet_path
    )

    parquet_rows = parquet.metadata.num_rows

    result["parquet_rows"] = parquet_rows

    print(
        f"Parquet rows: {parquet_rows:,}"
    )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    parquet_columns = (
        parquet.schema_arrow.names
    )

    missing_columns = [
        col
        for col in EXPECTED_COLUMNS
        if col not in parquet_columns
    ]

    extra_columns = [
        col
        for col in parquet_columns
        if col not in EXPECTED_COLUMNS
    ]

    if not missing_columns:
        result["schema_ok"] = True

    print(
        "\nSchema:"
    )

    print(
        f"  Expected columns : {len(EXPECTED_COLUMNS)}"
    )

    print(
        f"  Actual columns   : {len(parquet_columns)}"
    )

    print(
        f"  Missing columns  : {missing_columns}"
    )

    print(
        f"  Extra columns    : {extra_columns}"
    )

    # --------------------------------------------------------
    # Row count comparison
    # --------------------------------------------------------

    result["row_count_match"] = (
        raw_rows == parquet_rows
    )

    print(
        "\nRow count match:",
        result["row_count_match"],
    )

    # --------------------------------------------------------
    # Stream through Parquet
    # --------------------------------------------------------

    columns_to_read = [
        "entity_id",
        "business_name",
        "business_address",
        "country",

        "name_norm",
        "name_core",
        "name_translit",

        "address_norm",
        "address_translit",

        "name_script",
        "address_script",
        "script_type",

        "postal_code",
        "house_number",
        "phone",
        "domain",

        "country_norm",
        "source",
    ]

    # Only read columns that actually exist.
    columns_to_read = [
        col
        for col in columns_to_read
        if col in parquet_columns
    ]

    total = 0

    empty_ids = 0
    wrong_source = 0

    script_counts = {
        "indian": 0,
        "latin": 0,
        "mixed": 0,
        "other": 0,
        "empty": 0,
    }

    null_counts = {
        col: 0
        for col in columns_to_read
    }

    samples = []

    # --------------------------------------------------------
    # Iterate in Arrow batches
    # --------------------------------------------------------

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=columns_to_read,
    ):

        data = batch.to_pydict()

        batch_size = batch.num_rows

        total += batch_size

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        ids = data.get(
            "entity_id",
            [],
        )

        for entity_id in ids:

            if (
                entity_id is None
                or str(entity_id).strip() == ""
            ):
                empty_ids += 1

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        sources = data.get(
            "source",
            [],
        )

        for value in sources:

            if str(value).strip() != source:
                wrong_source += 1

        # ----------------------------------------------------
        # Script
        # ----------------------------------------------------

        scripts = data.get(
            "script_type",
            [],
        )

        for value in scripts:

            value = str(value).strip()

            if value in script_counts:
                script_counts[value] += 1
            else:
                script_counts["other"] += 1

        # ----------------------------------------------------
        # Null/empty values
        # ----------------------------------------------------

        for column in columns_to_read:

            values = data.get(
                column,
                [],
            )

            for value in values:

                if (
                    value is None
                    or str(value).strip() == ""
                ):
                    null_counts[column] += 1

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        if len(samples) < 10:

            names = data.get(
                "business_name",
                [],
            )

            addresses = data.get(
                "business_address",
                [],
            )

            name_norms = data.get(
                "name_norm",
                [],
            )

            name_translits = data.get(
                "name_translit",
                [],
            )

            address_norms = data.get(
                "address_norm",
                [],
            )

            for i in range(
                min(
                    batch_size,
                    10 - len(samples),
                )
            ):

                samples.append({
                    "entity_id": ids[i],
                    "business_name": names[i],
                    "name_norm": name_norms[i],
                    "name_translit": name_translits[i],
                    "business_address": addresses[i],
                    "address_norm": address_norms[i],
                })

    # --------------------------------------------------------
    # Store counts
    # --------------------------------------------------------

    result["parquet_rows_streamed"] = total

    result["empty_ids"] = empty_ids
    result["wrong_source"] = wrong_source

    result["indian_script_rows"] = (
        script_counts["indian"]
    )

    result["latin_rows"] = (
        script_counts["latin"]
    )

    result["mixed_rows"] = (
        script_counts["mixed"]
    )

    result["missing_values"] = null_counts

    result["sample"] = samples

    # --------------------------------------------------------
    # ID uniqueness
    # --------------------------------------------------------

    print(
        "\nChecking entity_id uniqueness..."
    )

    inserted = 0

    # Read IDs separately in batches.
    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["entity_id"],
    ):

        ids = batch.column(
            0
        ).to_pylist()

        inserted += insert_ids(
            conn,
            source,
            ids,
        )

    unique_ids = conn.execute(
        """
        SELECT COUNT(*)
        FROM ids
        WHERE source = ?
        """,
        (source,),
    ).fetchone()[0]

    duplicate_ids = (
        total - unique_ids
    )

    result["duplicate_ids"] = duplicate_ids

    print(
        f"Rows              : {total:,}"
    )

    print(
        f"Unique entity IDs  : {unique_ids:,}"
    )

    print(
        f"Duplicate IDs      : {duplicate_ids:,}"
    )

    # --------------------------------------------------------
    # Script distribution
    # --------------------------------------------------------

    print("\nScript distribution:")

    for key, value in script_counts.items():

        print(
            f"  {key:8s}: {value:,}"
        )

    # --------------------------------------------------------
    # Important empty fields
    # --------------------------------------------------------

    print("\nImportant field availability:")

    important = [
        "business_name",
        "business_address",
        "name_norm",
        "name_translit",
        "address_norm",
        "address_translit",
        "postal_code",
        "house_number",
        "phone",
        "domain",
    ]

    for column in important:

        if column in null_counts:

            nonempty = (
                total
                - null_counts[column]
            )

            pct = (
                100 * nonempty / total
                if total
                else 0
            )

            print(
                f"  {column:20s}: "
                f"{nonempty:,} "
                f"({pct:.2f}%)"
            )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("NORMALIZATION QA")
    print("=" * 80)

    conn = create_database()

    results = []

    try:

        for source, paths in SOURCES.items():

            result = check_normalized_file(
                source=source,
                raw_path=paths["raw"],
                parquet_path=paths["normalized"],
                conn=conn,
            )

            results.append(result)

    finally:

        conn.close()

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    failures = []

    for result in results:

        source = result["source"]

        if not result["row_count_match"]:
            failures.append(
                f"{source}: row count mismatch"
            )

        if result["duplicate_ids"] != 0:
            failures.append(
                f"{source}: duplicate entity IDs = "
                f"{result['duplicate_ids']:,}"
            )

        if result["empty_ids"] != 0:
            failures.append(
                f"{source}: empty entity IDs = "
                f"{result['empty_ids']:,}"
            )

        if result["wrong_source"] != 0:
            failures.append(
                f"{source}: wrong source values = "
                f"{result['wrong_source']:,}"
            )

        if not result["schema_ok"]:
            failures.append(
                f"{source}: schema mismatch"
            )

    # ========================================================
    # REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "NORMALIZATION QA REPORT\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        for result in results:

            source = result["source"]

            report.write(
                f"\n{source}\n"
            )

            report.write(
                "-" * 60 + "\n"
            )

            report.write(
                f"Raw rows: "
                f"{result['raw_rows']:,}\n"
            )

            report.write(
                f"Parquet rows: "
                f"{result['parquet_rows']:,}\n"
            )

            report.write(
                f"Row count match: "
                f"{result['row_count_match']}\n"
            )

            report.write(
                f"Duplicate IDs: "
                f"{result['duplicate_ids']:,}\n"
            )

            report.write(
                f"Empty IDs: "
                f"{result['empty_ids']:,}\n"
            )

            report.write(
                f"Wrong source values: "
                f"{result['wrong_source']:,}\n"
            )

            report.write(
                f"Schema OK: "
                f"{result['schema_ok']}\n"
            )

            report.write(
                "\nScript distribution:\n"
            )

            report.write(
                f"  Indian: "
                f"{result['indian_script_rows']:,}\n"
            )

            report.write(
                f"  Latin: "
                f"{result['latin_rows']:,}\n"
            )

            report.write(
                f"  Mixed: "
                f"{result['mixed_rows']:,}\n"
            )

            report.write(
                "\nSamples:\n"
            )

            for sample in result["sample"]:

                report.write(
                    f"\nID: "
                    f"{sample['entity_id']}\n"
                )

                report.write(
                    f"Name: "
                    f"{sample['business_name']}\n"
                )

                report.write(
                    f"Name norm: "
                    f"{sample['name_norm']}\n"
                )

                report.write(
                    f"Name translit: "
                    f"{sample['name_translit']}\n"
                )

                report.write(
                    f"Address: "
                    f"{sample['business_address']}\n"
                )

                report.write(
                    f"Address norm: "
                    f"{sample['address_norm']}\n"
                )

        report.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        report.write(
            "FINAL VERDICT\n"
        )

        report.write(
            "=" * 80
            + "\n"
        )

        if failures:

            report.write(
                "FAIL\n\n"
            )

            for failure in failures:
                report.write(
                    failure + "\n"
                )

        else:

            report.write(
                "PASS\n"
            )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL NORMALIZATION QA")
    print("=" * 80)

    for result in results:

        print(
            f"\n{result['source']}"
        )

        print(
            f"  Raw rows       : "
            f"{result['raw_rows']:,}"
        )

        print(
            f"  Parquet rows   : "
            f"{result['parquet_rows']:,}"
        )

        print(
            f"  Count match    : "
            f"{result['row_count_match']}"
        )

        print(
            f"  Duplicate IDs  : "
            f"{result['duplicate_ids']:,}"
        )

        print(
            f"  Empty IDs      : "
            f"{result['empty_ids']:,}"
        )

        print(
            f"  Wrong source   : "
            f"{result['wrong_source']:,}"
        )

        print(
            f"  Schema OK      : "
            f"{result['schema_ok']}"
        )

    print("\n" + "=" * 80)

    if failures:

        print(
            "❌ FAIL - NORMALIZATION QA FOUND PROBLEMS"
        )

        for failure in failures:
            print(
                f"  - {failure}"
            )

    else:

        print(
            "✅ PASS - NORMALIZATION QA PASSED"
        )

        print(
            "\nThe normalized Parquet files are ready "
            "for the next stage."
        )

    print("=" * 80)

    print(
        f"\nDetailed report:"
        f"\n{REPORT_FILE}"
    )

    print(
        f"\nTemporary QA database:"
        f"\n{DB_FILE}"
    )


if __name__ == "__main__":
    main()