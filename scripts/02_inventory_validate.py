from pathlib import Path
import csv
import re
from collections import Counter, defaultdict


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
US_ROOT = PROCESSED_ROOT / "us_by_state"
INDIA_ROOT = PROCESSED_ROOT / "india_by_state"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "processed_file_inventory.csv"


# ============================================================
# EXPECTED DATASET SCHEMA
# ============================================================

BASE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

VALID_STATE_COLUMNS = {
    "state",
    "statee",
}


# ============================================================
# CANONICAL STATES
# ============================================================

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
# SPECIAL DATASET SUFFIXES
# ============================================================

SPECIAL_SUFFIXES = {
    "unknown",
    "address_present_state_unknown",
}


# ============================================================
# ARTIFACTS THAT MUST NOT ENTER THE PIPELINE
# ============================================================

def is_artifact(path: Path):
    name = path.name.lower()

    # Previously generated recovery report
    if name == "us_recovery_report.csv":
        return True, "audit_report"

    # Files produced before a recovery operation
    if name.endswith("_before_recovery.csv"):
        return True, "before_recovery"

    # Other possible temporary/backup files
    if name.endswith("_original_backup.csv"):
        return True, "backup"

    if name.endswith("_repair_review.csv"):
        return True, "repair_review"

    if name.endswith("_temp.csv"):
        return True, "temporary"

    if name.endswith("_temp_append.csv"):
        return True, "temporary"

    return False, ""


# ============================================================
# NORMALIZE STATE NAME FROM FILENAME
# ============================================================

def filename_to_state(token):
    """
    Convert:
        New_York -> New York
        Andhra_Pradesh -> Andhra Pradesh

    Keep the original filename token separately.
    """

    return token.replace("_", " ")


# ============================================================
# CLASSIFY DATASET FILE
# ============================================================

def classify_dataset(path: Path):

    filename = path.stem

    # --------------------------------------------------------
    # INDIA
    # --------------------------------------------------------

    if "india_by_state" in path.parts:

        # Source 1:
        # train_source1_India_Karnataka.csv
        m = re.match(
            r"^(train_source1)_India_(.+)$",
            filename,
            flags=re.IGNORECASE,
        )

        if m:
            source = m.group(1).lower()
            suffix = m.group(2)

            if suffix in SPECIAL_SUFFIXES:
                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": "",
                    "category": "india_special",
                }

            if suffix in INDIA_STATES:
                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": filename_to_state(suffix),
                    "category": "india_state",
                }

        # Source 2/3 English Latin:
        # train_source2_India_english_latin_Karnataka.csv
        m = re.match(
            r"^(train_source[23])_India_english_latin_(.+)$",
            filename,
            flags=re.IGNORECASE,
        )

        if m:
            source = m.group(1).lower()
            suffix = m.group(2)

            if suffix in SPECIAL_SUFFIXES:
                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": "",
                    "category": "india_special",
                }

            if suffix in INDIA_STATES:
                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": filename_to_state(suffix),
                    "category": "india_state",
                }

        # Source 2/3 Indian script:
        # train_source2_India_indian_script_Karnataka.csv
        m = re.match(
            r"^(train_source[23])_India_indian_script_(.+)$",
            filename,
            flags=re.IGNORECASE,
        )

        if m:
            source = m.group(1).lower()
            suffix = m.group(2)

            if suffix in SPECIAL_SUFFIXES:
                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": "",
                    "category": "india_special",
                }

            if suffix in INDIA_STATES:
                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": filename_to_state(suffix),
                    "category": "india_state",
                }

        # Review directory:
        # address_present_state_unknown_review
        if "address_present_state_unknown_review" in path.parts:

            m = re.match(
                r"^(train_source[23])_India_(.+)_address_present_state_unknown$",
                filename,
                flags=re.IGNORECASE,
            )

            if m:
                source = m.group(1).lower()

                return {
                    "include": True,
                    "source": source,
                    "country": "India",
                    "state": "",
                    "category": "india_review",
                }

        return None

    # --------------------------------------------------------
    # US
    # --------------------------------------------------------

    if "us_by_state" in path.parts:

        # train_source1_US_Kansas.csv
        m = re.match(
            r"^(train_source[123])_US_(.+)$",
            filename,
            flags=re.IGNORECASE,
        )

        if m:
            source = m.group(1).lower()
            suffix = m.group(2)

            if suffix in SPECIAL_SUFFIXES:
                return {
                    "include": True,
                    "source": source,
                    "country": "US",
                    "state": "",
                    "category": "us_special",
                }

            if suffix in US_STATES:
                return {
                    "include": True,
                    "source": source,
                    "country": "US",
                    "state": filename_to_state(suffix),
                    "category": "us_state",
                }

        return None

    return None


# ============================================================
# READ HEADER ONLY
# ============================================================

def read_header(path: Path):

    try:
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline="",
            errors="replace",
        ) as f:

            first_line = f.readline()

            if not first_line:
                return None, "empty_file"

            # Processed files are pipe-delimited.
            # Be strict here because we don't want to silently
            # accept a malformed delimiter.
            header = next(
                csv.reader(
                    [first_line],
                    delimiter="|",
                )
            )

            header = [x.strip() for x in header]

            return header, ""

    except Exception as e:
        return None, f"read_error: {e}"


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(header):

    if header is None:
        return False, "missing_header"

    if len(header) != 5:
        return False, f"expected_5_columns_found_{len(header)}"

    if header[:4] != BASE_COLUMNS:
        return False, f"first_four_columns_invalid: {header}"

    if header[4] not in VALID_STATE_COLUMNS:
        return False, f"invalid_state_column: {header[4]}"

    return True, ""


# ============================================================
# MAIN INVENTORY
# ============================================================

def main():

    print("=" * 80)
    print("PROCESSED DATA INVENTORY VALIDATION")
    print("=" * 80)

    all_files = []

    if US_ROOT.exists():
        all_files.extend(US_ROOT.rglob("*.csv"))

    if INDIA_ROOT.exists():
        all_files.extend(INDIA_ROOT.rglob("*.csv"))

    all_files = sorted(set(all_files))

    print(f"\nCSV files discovered: {len(all_files)}")

    records = []

    unclassified = []
    schema_errors = []
    artifacts = []

    source_counts = Counter()
    country_counts = Counter()
    category_counts = Counter()
    state_counts = Counter()

    # --------------------------------------------------------
    # PROCESS EACH FILE
    # --------------------------------------------------------

    for path in all_files:

        relative_path = path.relative_to(PROJECT_ROOT)

        # ----------------------------------------------------
        # ARTIFACT CHECK FIRST
        # ----------------------------------------------------

        artifact, artifact_type = is_artifact(path)

        if artifact:

            records.append({
                "path": str(relative_path),
                "filename": path.name,
                "source": "",
                "country": "",
                "state": "",
                "category": artifact_type,
                "included": "NO",
                "schema_status": "EXCLUDED_ARTIFACT",
                "reason": artifact_type,
            })

            artifacts.append((relative_path, artifact_type))
            continue

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        info = classify_dataset(path)

        if info is None:

            records.append({
                "path": str(relative_path),
                "filename": path.name,
                "source": "",
                "country": "",
                "state": "",
                "category": "unclassified",
                "included": "NO",
                "schema_status": "NOT_CHECKED",
                "reason": "filename_or_location_not_recognized",
            })

            unclassified.append(relative_path)
            continue

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header, header_error = read_header(path)

        if header_error:

            records.append({
                "path": str(relative_path),
                "filename": path.name,
                "source": info["source"],
                "country": info["country"],
                "state": info["state"],
                "category": info["category"],
                "included": "NO",
                "schema_status": "ERROR",
                "reason": header_error,
            })

            schema_errors.append(
                (relative_path, header_error)
            )

            continue

        valid_schema, schema_reason = validate_schema(header)

        if not valid_schema:

            records.append({
                "path": str(relative_path),
                "filename": path.name,
                "source": info["source"],
                "country": info["country"],
                "state": info["state"],
                "category": info["category"],
                "included": "NO",
                "schema_status": "ERROR",
                "reason": schema_reason,
            })

            schema_errors.append(
                (relative_path, schema_reason)
            )

            continue

        # ----------------------------------------------------
        # VALID DATASET
        # ----------------------------------------------------

        source = info["source"]
        country = info["country"]
        state = info["state"]
        category = info["category"]

        source_counts[source] += 1
        country_counts[country] += 1
        category_counts[category] += 1

        if state:
            state_counts[(country, state)] += 1

        records.append({
            "path": str(relative_path),
            "filename": path.name,
            "source": source,
            "country": country,
            "state": state,
            "category": category,
            "included": "YES",
            "schema_status": "OK",
            "reason": "",
        })

    # ========================================================
    # WRITE INVENTORY REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "filename",
                "source",
                "country",
                "state",
                "category",
                "included",
                "schema_status",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(records)

    # ========================================================
    # SUMMARY
    # ========================================================

    included_count = sum(
        1
        for r in records
        if r["included"] == "YES"
    )

    excluded_count = sum(
        1
        for r in records
        if r["included"] == "NO"
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal CSV files discovered : {len(all_files)}")
    print(f"Included dataset files     : {included_count}")
    print(f"Excluded files             : {excluded_count}")

    print("\n--- INCLUDED BY SOURCE ---")

    for source in sorted(source_counts):
        print(
            f"{source}: {source_counts[source]}"
        )

    print("\n--- INCLUDED BY COUNTRY ---")

    for country in sorted(country_counts):
        print(
            f"{country}: {country_counts[country]}"
        )

    print("\n--- INCLUDED BY CATEGORY ---")

    for category in sorted(category_counts):
        print(
            f"{category}: {category_counts[category]}"
        )

    print("\n--- EXCLUDED ARTIFACTS ---")

    if artifacts:
        artifact_counter = Counter(
            artifact_type
            for _, artifact_type in artifacts
        )

        for artifact_type, count in sorted(
            artifact_counter.items()
        ):
            print(
                f"{artifact_type}: {count}"
            )

        for path, artifact_type in artifacts:
            print(
                f"  [{artifact_type}] {path}"
            )

    else:
        print("None")

    print("\n--- UNCLASSIFIED FILES ---")

    if unclassified:
        for path in unclassified:
            print(f"  {path}")
    else:
        print("None")

    print("\n--- SCHEMA ERRORS ---")

    if schema_errors:
        for path, reason in schema_errors:
            print(
                f"  {path}"
                f"\n      {reason}"
            )
    else:
        print("None")

    # ========================================================
    # STATE INVENTORY
    # ========================================================

    print("\n--- STATE FILE COUNTS ---")

    us_states = sorted(
        state
        for (country, state) in state_counts
        if country == "US"
    )

    india_states = sorted(
        state
        for (country, state) in state_counts
        if country == "India"
    )

    print(f"US states/districts found    : {len(us_states)}")
    print(f"India states/UTs found       : {len(india_states)}")

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if unclassified:
        print(
            "\n❌ FAIL: Unclassified files exist."
        )

        for path in unclassified:
            print(f"   {path}")

    elif schema_errors:
        print(
            "\n❌ FAIL: Schema errors exist."
        )

        print(
            "Fix the schema errors before normalization."
        )

    else:
        print(
            "\n✅ PASS: Every non-artifact CSV file is classified "
            "and has a valid 5-column processed schema."
        )

        print(
            "\nThe normalizer can safely accept both:"
        )
        print("    state")
        print("    statee")

        print(
            "\nNo raw TSV files were inspected or modified."
        )

    print(
        f"\nInventory report written to:"
        f"\n{REPORT_FILE}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()