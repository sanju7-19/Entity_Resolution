from pathlib import Path
import re
import unicodedata

import pandas as pd

try:
    from unidecode import unidecode
except ImportError:
    raise SystemExit(
        "Unidecode is not installed.\n"
        "Run: pip install Unidecode"
    )


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\sanja\Downloads\amazon_ml"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "features"
    / "normalized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RAW_FILES = {
    "train_source1": RAW_DIR / "train_source1.tsv",
    "train_source2": RAW_DIR / "train_source2.tsv",
    "train_source3": RAW_DIR / "train_source3.tsv",
}


# ============================================================
# EXPECTED RAW SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


# ============================================================
# PROCESSING SETTINGS
# ============================================================

CHUNK_SIZE = 100_000


# ============================================================
# LEGAL SUFFIXES
# ============================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "limited",
    "ltd",
    "llc",
    "llp",
    "pvt",
    "private",
}


# ============================================================
# UNICODE NORMALIZATION
# ============================================================

def unicode_normalize(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    return value.strip()


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def basic_normalize(value):

    value = unicode_normalize(value)

    if not value:
        return ""

    value = value.lower()

    # Normalize ampersand to "and"
    value = value.replace("&", " and ")

    # Keep Unicode letters/numbers.
    # Punctuation becomes spaces.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    # Collapse whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# NAME CORE
# ============================================================

def make_name_core(value):

    normalized = basic_normalize(value)

    if not normalized:
        return ""

    tokens = normalized.split()

    tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


# ============================================================
# TRANSLITERATION
# ============================================================

def make_translit(value):

    value = unicode_normalize(value)

    if not value:
        return ""

    value = unidecode(value)

    return basic_normalize(value)


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_script(value):

    value = unicode_normalize(value)

    if not value:
        return "empty"

    latin = False
    indian = False

    for char in value:

        code = ord(char)

        # Latin / Latin Extended
        if (
            0x0041 <= code <= 0x024F
        ):
            latin = True

        # Devanagari
        elif (
            0x0900 <= code <= 0x097F
        ):
            indian = True

        # Bengali / Assamese
        elif (
            0x0980 <= code <= 0x09FF
        ):
            indian = True

        # Gurmukhi
        elif (
            0x0A00 <= code <= 0x0A7F
        ):
            indian = True

        # Gujarati
        elif (
            0x0A80 <= code <= 0x0AFF
        ):
            indian = True

        # Odia
        elif (
            0x0B00 <= code <= 0x0B7F
        ):
            indian = True

        # Tamil
        elif (
            0x0B80 <= code <= 0x0BFF
        ):
            indian = True

        # Telugu
        elif (
            0x0C00 <= code <= 0x0C7F
        ):
            indian = True

        # Kannada
        elif (
            0x0C80 <= code <= 0x0CFF
        ):
            indian = True

        # Malayalam
        elif (
            0x0D00 <= code <= 0x0D7F
        ):
            indian = True

    if latin and indian:
        return "mixed"

    if indian:
        return "indian"

    if latin:
        return "latin"

    return "other"


# ============================================================
# POSTAL CODE
# ============================================================

def extract_postal_code(address, country):

    address = unicode_normalize(address)

    if not address:
        return ""

    if country == "US":

        # ZIP / ZIP+4
        matches = re.findall(
            r"\b\d{5}(?:-\d{4})?\b",
            address,
        )

        if matches:
            return matches[-1]

    elif country == "India":

        # Indian PIN
        matches = re.findall(
            r"\b[1-9]\d{5}\b",
            address,
        )

        if matches:
            return matches[-1]

    return ""


# ============================================================
# HOUSE NUMBER
# ============================================================

def extract_house_number(address):

    address = unicode_normalize(address)

    if not address:
        return ""

    # Beginning:
    # 123 Main Street
    # 123A Main Street
    match = re.search(
        r"^\s*(\d+[A-Za-z]?)\b",
        address,
    )

    if match:
        return match.group(1)

    # No. 123
    # #123
    match = re.search(
        r"\b(?:no|number)\.?\s*#?\s*(\d+[A-Za-z]?)\b",
        address,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"#\s*(\d+[A-Za-z]?)\b",
        address,
    )

    if match:
        return match.group(1)

    return ""


# ============================================================
# PHONE EXTRACTION
# ============================================================

def extract_phone(name, address):

    text = (
        unicode_normalize(name)
        + " "
        + unicode_normalize(address)
    )

    if not text:
        return ""

    # Look for plausible phone-like sequences.
    candidates = re.findall(
        r"(?:\+?\d[\d\s().-]{6,}\d)",
        text,
    )

    if not candidates:
        return ""

    # Normalize candidates and retain plausible numbers.
    normalized = []

    for candidate in candidates:

        digits = re.sub(
            r"\D",
            "",
            candidate,
        )

        if 7 <= len(digits) <= 15:
            normalized.append(digits)

    if not normalized:
        return ""

    # Do NOT arbitrarily use the last 10 digits.
    # Preserve the complete normalized candidate.
    return normalized[0]


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domain(name, address):

    text = (
        unicode_normalize(name)
        + " "
        + unicode_normalize(address)
    )

    if not text:
        return ""

    matches = re.findall(
        r"\b(?:https?://)?(?:www\.)?"
        r"([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)"
        r"(?:/[^\s]*)?",
        text,
    )

    if not matches:
        return ""

    domains = []

    for domain in matches:

        domain = domain.lower().strip(".,;:()[]{}<>")

        if domain.startswith("www."):
            domain = domain[4:]

        # Avoid treating obvious file-like things as domains.
        if "." in domain:
            domains.append(domain)

    if not domains:
        return ""

    return domains[0]


# ============================================================
# APPLY NORMALIZATION TO ONE CHUNK
# ============================================================

def normalize_chunk(chunk, source):

    # Force strings.
    for column in EXPECTED_COLUMNS:
        chunk[column] = (
            chunk[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Preserve original raw fields
    # --------------------------------------------------------

    chunk["source"] = source

    # --------------------------------------------------------
    # Name normalization
    # --------------------------------------------------------

    chunk["name_norm"] = (
        chunk["business_name"]
        .map(basic_normalize)
    )

    chunk["name_core"] = (
        chunk["business_name"]
        .map(make_name_core)
    )

    chunk["name_translit"] = (
        chunk["business_name"]
        .map(make_translit)
    )

    # --------------------------------------------------------
    # Address normalization
    # --------------------------------------------------------

    chunk["address_norm"] = (
        chunk["business_address"]
        .map(basic_normalize)
    )

    chunk["address_translit"] = (
        chunk["business_address"]
        .map(make_translit)
    )

    # --------------------------------------------------------
    # Script
    # --------------------------------------------------------

    chunk["name_script"] = (
        chunk["business_name"]
        .map(detect_script)
    )

    chunk["address_script"] = (
        chunk["business_address"]
        .map(detect_script)
    )

    # Combined script is useful later.
    chunk["script_type"] = (
        chunk["business_name"]
        + " "
        + chunk["business_address"]
    ).map(detect_script)

    # --------------------------------------------------------
    # Postal code
    # --------------------------------------------------------

    chunk["postal_code"] = [
        extract_postal_code(
            address,
            country,
        )
        for address, country
        in zip(
            chunk["business_address"],
            chunk["country"],
        )
    ]

    # --------------------------------------------------------
    # House number
    # --------------------------------------------------------

    chunk["house_number"] = (
        chunk["business_address"]
        .map(extract_house_number)
    )

    # --------------------------------------------------------
    # Phone
    # --------------------------------------------------------

    chunk["phone"] = [
        extract_phone(
            name,
            address,
        )
        for name, address
        in zip(
            chunk["business_name"],
            chunk["business_address"],
        )
    ]

    # --------------------------------------------------------
    # Domain
    # --------------------------------------------------------

    chunk["domain"] = [
        extract_domain(
            name,
            address,
        )
        for name, address
        in zip(
            chunk["business_name"],
            chunk["business_address"],
        )
    ]

    # --------------------------------------------------------
    # Country normalization
    # --------------------------------------------------------

    chunk["country_norm"] = (
        chunk["country"]
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    columns = [
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

    return chunk[columns]


# ============================================================
# PROCESS ONE RAW SOURCE
# ============================================================

def process_source(source, input_path):

    print("\n" + "=" * 80)
    print(f"PROCESSING {source}")
    print("=" * 80)

    print(f"Input : {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {input_path}"
        )

    output_path = (
        OUTPUT_DIR
        / f"{source}.parquet"
    )

    temp_dir = (
        OUTPUT_DIR
        / "_chunks"
        / source
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove previous temporary chunks.
    for old in temp_dir.glob("*.parquet"):
        old.unlink()

    # Remove previous output.
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    chunk_number = 0

    # --------------------------------------------------------
    # Read raw TSV in chunks
    # --------------------------------------------------------

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="error",
    )

    first_chunk = True

    for chunk in reader:

        chunk_number += 1

        print(
            f"  Chunk {chunk_number:,}"
            f" | rows: {len(chunk):,}"
            f" | total: {total_rows + len(chunk):,}"
        )

        # ----------------------------------------------------
        # Verify schema
        # ----------------------------------------------------

        actual_columns = list(
            chunk.columns
        )

        if actual_columns != EXPECTED_COLUMNS:

            raise ValueError(
                "\nRAW SCHEMA MISMATCH\n"
                f"File: {input_path}\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found: {actual_columns}"
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized = normalize_chunk(
            chunk,
            source,
        )

        # ----------------------------------------------------
        # Write temporary Parquet
        # ----------------------------------------------------

        chunk_path = (
            temp_dir
            / f"chunk_{chunk_number:05d}.parquet"
        )

        normalized.to_parquet(
            chunk_path,
            index=False,
            engine="pyarrow",
        )

        total_rows += len(chunk)

        del chunk
        del normalized

    # ========================================================
    # Merge chunks
    # ========================================================

    print("\n  Merging Parquet chunks...")

    chunk_files = sorted(
        temp_dir.glob("*.parquet")
    )

    if not chunk_files:
        raise RuntimeError(
            f"No normalized chunks generated for {source}"
        )

    first = True

    for chunk_path in chunk_files:

        df = pd.read_parquet(
            chunk_path
        )

        if first:

            df.to_parquet(
                output_path,
                index=False,
                engine="pyarrow",
            )

            first = False

        else:

            # Append using PyArrow ParquetWriter.
            import pyarrow as pa
            import pyarrow.parquet as pq

            # This branch is intentionally handled below.
            break

    # --------------------------------------------------------
    # Proper chunk merge using PyArrow
    # --------------------------------------------------------

    import pyarrow.parquet as pq

    if output_path.exists():
        output_path.unlink()

    writer = None

    try:

        for chunk_path in chunk_files:

            table = pq.read_table(
                chunk_path
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(
                table
            )

    finally:

        if writer is not None:
            writer.close()

    # --------------------------------------------------------
    # Remove temporary chunks
    # --------------------------------------------------------

    for chunk_path in chunk_files:
        chunk_path.unlink()

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print(
        f"\nCompleted {source}"
    )

    print(
        f"Rows written: {total_rows:,}"
    )

    print(
        f"Output: {output_path}"
    )

    return total_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RAW DATA NORMALIZATION")
    print("=" * 80)

    print(
        "\nIMPORTANT:"
        "\nThe state-wise processed CSV files are NOT used."
        "\nOnly the immutable raw TSV files are used."
    )

    total = 0

    for source, input_path in RAW_FILES.items():

        rows = process_source(
            source,
            input_path,
        )

        total += rows

    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal normalized rows: {total:,}"
    )

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "\nGenerated:"
    )

    for source in RAW_FILES:

        output = (
            OUTPUT_DIR
            / f"{source}.parquet"
        )

        if output.exists():

            print(
                f"  ✓ {output.name}"
            )

    print(
        "\nRaw files were not modified."
    )


if __name__ == "__main__":
    main()