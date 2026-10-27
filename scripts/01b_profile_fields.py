from pathlib import Path
import csv
import re
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FILES = [
    RAW_DIR / "train_source1.tsv",
    RAW_DIR / "train_source2.tsv",
    RAW_DIR / "train_source3.tsv",
]


def profile_file(path, max_rows=500_000):

    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)

    total = 0

    empty_name = 0
    empty_address = 0
    empty_country = 0

    urls = 0
    phones = 0
    postal_like = 0
    indian_script = 0
    latin_script = 0
    mixed_script = 0

    countries = Counter()

    sample_urls = []
    sample_phone = []
    sample_postal = []
    sample_indian = []

    # Rough indicators only.
    # We are measuring patterns, not changing the data.

    url_pattern = re.compile(
        r"(https?://|www\.|[a-z0-9-]+\.(com|org|net|in|co\.in|biz|info)\b)",
        re.I,
    )

    phone_pattern = re.compile(
        r"(?:\+?\d[\d\s().-]{7,}\d)"
    )

    postal_pattern = re.compile(
        r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b"
    )

    devanagari = re.compile(r"[\u0900-\u097F]")
    bengali = re.compile(r"[\u0980-\u09FF]")
    gurmukhi = re.compile(r"[\u0A00-\u0A7F]")
    gujarati = re.compile(r"[\u0A80-\u0AFF]")
    kannada = re.compile(r"[\u0C80-\u0CFF]")
    malayalam = re.compile(r"[\u0D00-\u0D7F]")
    tamil = re.compile(r"[\u0B80-\u0BFF]")
    telugu = re.compile(r"[\u0C00-\u0C7F]")

    indian_script_pattern = re.compile(
        r"[\u0900-\u097F"
        r"\u0980-\u09FF"
        r"\u0A00-\u0A7F"
        r"\u0A80-\u0AFF"
        r"\u0B00-\u0B7F"
        r"\u0B80-\u0BFF"
        r"\u0C00-\u0C7F"
        r"\u0C80-\u0CFF"
        r"\u0D00-\u0D7F]"
    )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            total += 1

            name = (row.get("business_name") or "").strip()
            address = (row.get("business_address") or "").strip()
            country = (row.get("country") or "").strip()

            if not name:
                empty_name += 1

            if not address:
                empty_address += 1

            if not country:
                empty_country += 1

            if country:
                countries[country] += 1

            combined = f"{name} {address}"

            if url_pattern.search(combined):
                urls += 1

                if len(sample_urls) < 10:
                    sample_urls.append(combined)

            if phone_pattern.search(combined):
                phones += 1

                if len(sample_phone) < 10:
                    sample_phone.append(combined)

            if postal_pattern.search(combined):
                postal_like += 1

                if len(sample_postal) < 10:
                    sample_postal.append(combined)

            if indian_script_pattern.search(combined):
                indian_script += 1

                if len(sample_indian) < 10:
                    sample_indian.append(combined)

            has_latin = bool(re.search(r"[A-Za-z]", combined))
            has_indian = bool(indian_script_pattern.search(combined))

            if has_latin and has_indian:
                mixed_script += 1
            elif has_latin:
                latin_script += 1

            if total >= max_rows:
                break

    print(f"\nRows inspected: {total:,}")

    print("\nEmpty fields:")
    print(f"  Empty business_name    : {empty_name:,}")
    print(f"  Empty business_address : {empty_address:,}")
    print(f"  Empty country          : {empty_country:,}")

    print("\nDetected patterns:")
    print(f"  URL/domain-like        : {urls:,}")
    print(f"  Phone-like             : {phones:,}")
    print(f"  Postal-code-like       : {postal_like:,}")
    print(f"  Indian-script          : {indian_script:,}")
    print(f"  Latin-only             : {latin_script:,}")
    print(f"  Mixed Latin + Indian   : {mixed_script:,}")

    print("\nCountries:")

    for country, count in countries.most_common(20):
        print(f"  {repr(country):20} {count:,}")

    print("\nURL/domain examples:")
    for x in sample_urls:
        print(" ", x)

    print("\nPhone examples:")
    for x in sample_phone:
        print(" ", x)

    print("\nPostal examples:")
    for x in sample_postal:
        print(" ", x)

    print("\nIndian-script examples:")
    for x in sample_indian:
        print(" ", x)


def main():

    for path in FILES:
        profile_file(path)


if __name__ == "__main__":
    main()