from pathlib import Path
import csv

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    "source1": RAW_DIR / "train_source1.tsv",
    "source2": RAW_DIR / "train_source2.tsv",
    "source3": RAW_DIR / "train_source3.tsv",
    "ground_truth": GT_DIR / "train_ground_truth.tsv",
}


EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "state",
]

EXPECTED_GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


# ============================================================
# BASIC FILE INFORMATION
# ============================================================

def inspect_file(path, expected_columns, delimiter="\t", sample_rows=5):

    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    if not path.exists():
        print("ERROR: File does not exist!")
        return

    print(f"Size: {path.stat().st_size / (1024 ** 3):.2f} GB")

    row_count = 0
    unique_ids = set()
    duplicate_ids = 0

    countries = {}
    states = {}

    sample = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as f:

        reader = csv.reader(f, delimiter=delimiter)

        try:
            header = next(reader)
        except StopIteration:
            print("ERROR: Empty file")
            return

        print("\nColumns:")
        for i, col in enumerate(header):
            print(f"  {i}: {repr(col)}")

        print("\nExpected columns:")
        print(expected_columns)

        if header != expected_columns:
            print("\nWARNING: Column mismatch!")
        else:
            print("\nColumn structure: OK")

        for row in reader:

            row_count += 1

            if len(row) != len(header):
                print(
                    f"WARNING: Row {row_count} has "
                    f"{len(row)} columns instead of {len(header)}"
                )

                if len(sample) < sample_rows:
                    sample.append(row)

                continue

            entity_id = row[0].strip()

            if entity_id in unique_ids:
                duplicate_ids += 1
            else:
                unique_ids.add(entity_id)

            # Source files have:
            # entity_id | business_name | business_address | country | state

            if len(row) >= 5:

                country = row[3].strip()
                state = row[4].strip()

                countries[country] = countries.get(country, 0) + 1
                states[state] = states.get(state, 0) + 1

            if len(sample) < sample_rows:
                sample.append(row)

    print("\nRow statistics:")
    print(f"  Rows              : {row_count:,}")
    print(f"  Unique entity IDs : {len(unique_ids):,}")
    print(f"  Duplicate IDs     : {duplicate_ids:,}")

    print("\nTop countries:")

    for country, count in sorted(
        countries.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]:

        print(f"  {repr(country):30} {count:,}")

    print("\nTop states:")

    for state, count in sorted(
        states.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]:

        print(f"  {repr(state):30} {count:,}")

    print("\nSample rows:")

    for row in sample:
        print(row)


# ============================================================
# GROUND TRUTH INSPECTION
# ============================================================

def inspect_ground_truth(path):

    print("\n" + "=" * 80)
    print("GROUND TRUTH")
    print("=" * 80)

    if not path.exists():
        print("ERROR: Ground-truth file does not exist!")
        return

    print(f"Size: {path.stat().st_size / (1024 ** 3):.2f} GB")

    row_count = 0
    unique_s1 = set()

    total_matches = 0
    max_matches = 0

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as f:

        reader = csv.reader(f, delimiter="\t")

        header = next(reader)

        print("\nColumns:")
        for i, col in enumerate(header):
            print(f"  {i}: {repr(col)}")

        if header != EXPECTED_GT_COLUMNS:
            print("\nWARNING: Ground-truth columns differ from expected!")

        print("\nSample ground-truth rows:")

        for row in reader:

            row_count += 1

            if len(row) < 2:
                continue

            s1_id = row[0].strip()
            matched_ids = row[1].strip()

            unique_s1.add(s1_id)

            if matched_ids:
                matches = [
                    x.strip()
                    for x in matched_ids.split(",")
                    if x.strip()
                ]

                total_matches += len(matches)
                max_matches = max(max_matches, len(matches))

            if row_count <= 10:
                print(row)

    print("\nGround-truth statistics:")
    print(f"  S1 rows             : {row_count:,}")
    print(f"  Unique S1 IDs       : {len(unique_s1):,}")
    print(f"  Total match links   : {total_matches:,}")
    print(f"  Max matches / S1    : {max_matches:,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AMAZON ML ENTITY RESOLUTION - DATA INSPECTION")
    print("=" * 80)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print("\nRaw data directory:")
    print(RAW_DIR)

    print("\nGround truth directory:")
    print(GT_DIR)

    # Inspect source datasets

    for name in ["source1", "source2", "source3"]:

        inspect_file(
            FILES[name],
            EXPECTED_SOURCE_COLUMNS,
            delimiter="\t",
        )

    # Inspect ground truth

    inspect_ground_truth(FILES["ground_truth"])

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()