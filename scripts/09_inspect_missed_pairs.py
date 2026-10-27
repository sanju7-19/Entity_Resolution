from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

S1_FILE = ROOT / "features" / "normalized" / "train_source1.parquet"
S2_FILE = ROOT / "features" / "normalized" / "train_source2.parquet"
S3_FILE = ROOT / "features" / "normalized" / "train_source3.parquet"

MISSED_S2 = (
    ROOT
    / "outputs"
    / "blocking_diagnostics"
    / "missed_s2.parquet"
)

MISSED_S3 = (
    ROOT
    / "outputs"
    / "blocking_diagnostics"
    / "missed_s3.parquet"
)

OUT = (
    ROOT
    / "outputs"
    / "blocking_diagnostics"
)

# ============================================================
# COLUMNS WE NEED
# ============================================================

RECORD_COLUMNS = [
    "entity_id",
    "business_name",
    "name_norm",
    "name_core",
    "name_translit",
    "business_address",
    "address_norm",
    "address_translit",
    "postal_code",
    "house_number",
    "phone",
    "domain",
    "name_script",
    "address_script",
]

# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

print("=" * 70)
print("LOADING NORMALIZED RECORDS")
print("=" * 70)

print("\nLoading S1...")

s1 = pd.read_parquet(
    S1_FILE,
    columns=RECORD_COLUMNS
)

print(f"S1 rows: {len(s1):,}")

print("\nLoading S2...")

s2 = pd.read_parquet(
    S2_FILE,
    columns=RECORD_COLUMNS
)

print(f"S2 rows: {len(s2):,}")

print("\nLoading S3...")

s3 = pd.read_parquet(
    S3_FILE,
    columns=RECORD_COLUMNS
)

print(f"S3 rows: {len(s3):,}")

# ============================================================
# RENAME COLUMNS
# ============================================================

def prepare_source(df, prefix):

    df = df.copy()

    df["entity_id"] = df["entity_id"].astype(str)

    rename = {}

    for column in df.columns:

        if column != "entity_id":
            rename[column] = f"{prefix}_{column}"

    df = df.rename(columns=rename)

    return df


s1 = prepare_source(s1, "s1")
s2 = prepare_source(s2, "target")
s3 = prepare_source(s3, "target")

# ============================================================
# EXACT MATCH HELPER
# ============================================================

def valid(series):

    return (
        series.notna()
        &
        (series.astype(str).str.strip() != "")
    )


def exact_rate(df, left, right):

    v = (
        valid(df[left])
        &
        valid(df[right])
    )

    if v.sum() == 0:
        return 0.0

    return (
        (
            df.loc[v, left].astype(str).str.strip()
            ==
            df.loc[v, right].astype(str).str.strip()
        )
        .mean()
        * 100
    )


def both_exact(df, a, b, c, d):

    v = (
        valid(df[a])
        &
        valid(df[b])
        &
        valid(df[c])
        &
        valid(df[d])
    )

    if v.sum() == 0:
        return 0.0

    return (
        (
            (
                df.loc[v, a].astype(str).str.strip()
                ==
                df.loc[v, b].astype(str).str.strip()
            )
            &
            (
                df.loc[v, c].astype(str).str.strip()
                ==
                df.loc[v, d].astype(str).str.strip()
            )
        )
        .mean()
        * 100
    )


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyse(missed_file, target_df, target_source):

    print("\n" + "=" * 70)
    print(f"{target_source} MISSED PAIRS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load missed pair IDs
    # --------------------------------------------------------

    missed = pd.read_parquet(missed_file)

    missed["s1_id"] = missed["s1_id"].astype(str)
    missed["target_id"] = missed["target_id"].astype(str)

    print(
        f"\nMissed pairs: {len(missed):,}"
    )

    # --------------------------------------------------------
    # Join S1 details
    # --------------------------------------------------------

    print("\nJoining S1 records...")

    s1_join = s1.rename(
        columns={
            "entity_id": "s1_id"
        }
    )

    df = missed.merge(
        s1_join,
        on="s1_id",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------------
    # Join target details
    # --------------------------------------------------------

    print("Joining target records...")

    target_join = target_df.rename(
        columns={
            "entity_id": "target_id"
        }
    )

    df = df.merge(
        target_join,
        on="target_id",
        how="left",
        validate="many_to_one"
    )

    print(
        f"Rows after enrichment: {len(df):,}"
    )

    # ========================================================
    # CHECK JOIN SUCCESS
    # ========================================================

    missing_s1 = df["s1_business_name"].isna().sum()
    missing_target = df["target_business_name"].isna().sum()

    print("\nJoin validation:")

    print(
        f"S1 records not found     : {missing_s1:,}"
    )

    print(
        f"Target records not found : {missing_target:,}"
    )

    if missing_s1 > 0 or missing_target > 0:

        print(
            "\nWARNING: Some entity IDs could not be enriched."
        )

    # ========================================================
    # FIELD AVAILABILITY
    # ========================================================

    print("\n" + "-" * 70)
    print("FIELD AVAILABILITY")
    print("-" * 70)

    fields = [
        (
            "name_norm",
            "s1_name_norm",
            "target_name_norm"
        ),
        (
            "name_core",
            "s1_name_core",
            "target_name_core"
        ),
        (
            "name_translit",
            "s1_name_translit",
            "target_name_translit"
        ),
        (
            "address_norm",
            "s1_address_norm",
            "target_address_norm"
        ),
        (
            "postal",
            "s1_postal_code",
            "target_postal_code"
        ),
        (
            "house_number",
            "s1_house_number",
            "target_house_number"
        ),
        (
            "phone",
            "s1_phone",
            "target_phone"
        ),
        (
            "domain",
            "s1_domain",
            "target_domain"
        ),
    ]

    for label, left, right in fields:

        left_available = (
            valid(df[left]).mean() * 100
        )

        right_available = (
            valid(df[right]).mean() * 100
        )

        print(
            f"{label:18s}"
            f"S1={left_available:7.2f}%   "
            f"Target={right_available:7.2f}%"
        )

    # ========================================================
    # EXACT AGREEMENT
    # ========================================================

    print("\n" + "-" * 70)
    print("EXACT AGREEMENT AMONG MISSED TRUE PAIRS")
    print("-" * 70)

    for label, left, right in fields:

        rate = exact_rate(
            df,
            left,
            right
        )

        print(
            f"{label:18s}"
            f"{rate:7.2f}%"
        )

    # ========================================================
    # COMPOSITE AGREEMENT
    # ========================================================

    print("\n" + "-" * 70)
    print("COMPOSITE AGREEMENT")
    print("-" * 70)

    rate = both_exact(
        df,
        "s1_name_core",
        "target_name_core",
        "s1_postal_code",
        "target_postal_code"
    )

    print(
        f"name_core + postal : {rate:.2f}%"
    )

    rate = both_exact(
        df,
        "s1_name_core",
        "target_name_core",
        "s1_house_number",
        "target_house_number"
    )

    print(
        f"name_core + house  : {rate:.2f}%"
    )

    rate = both_exact(
        df,
        "s1_name_norm",
        "target_name_norm",
        "s1_postal_code",
        "target_postal_code"
    )

    print(
        f"name_norm + postal : {rate:.2f}%"
    )

    rate = both_exact(
        df,
        "s1_name_translit",
        "target_name_translit",
        "s1_postal_code",
        "target_postal_code"
    )

    print(
        f"translit + postal  : {rate:.2f}%"
    )

    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    sample_size = min(
        1000,
        len(df)
    )

    sample = df.sample(
        sample_size,
        random_state=42
    )

    sample_file = (
        OUT
        / f"{target_source.lower()}_missed_sample.csv"
    )

    sample.to_csv(
        sample_file,
        index=False
    )

    print(
        f"\nManual sample saved:"
        f"\n{sample_file}"
    )

    # ========================================================
    # DISPLAY EXAMPLES
    # ========================================================

    print("\n" + "-" * 70)
    print("EXAMPLE MISSED PAIRS")
    print("-" * 70)

    display_columns = [
        "s1_id",
        "s1_business_name",
        "target_id",
        "target_business_name",
        "s1_name_norm",
        "target_name_norm",
        "s1_name_core",
        "target_name_core",
        "s1_postal_code",
        "target_postal_code",
        "s1_house_number",
        "target_house_number",
    ]

    print(
        sample[display_columns]
        .head(20)
        .to_string(index=False)
    )

    return df


# ============================================================
# RUN S2
# ============================================================

df_s2 = analyse(
    MISSED_S2,
    s2,
    "S2"
)

# ============================================================
# RUN S3
# ============================================================

df_s3 = analyse(
    MISSED_S3,
    s3,
    "S3"
)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("MISSED-PAIR INSPECTION COMPLETE")
print("=" * 70)