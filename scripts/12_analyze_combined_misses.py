from pathlib import Path
import duckdb
import pandas as pd
import time

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

S1_SAMPLE = ROOT / "outputs" / "blocking_sample_s1.parquet"
GT_SAMPLE = ROOT / "outputs" / "blocking_sample_gt.parquet"

S2_PARQUET = ROOT / "features" / "normalized" / "train_source2.parquet"
S3_PARQUET = ROOT / "features" / "normalized" / "train_source3.parquet"

OUT_DIR = ROOT / "outputs" / "blocking_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUT_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# We analyze the best recall configuration from V1+V2.
THRESHOLD = 0.80

# Number of missed pairs to inspect in detail.
MAX_MISSED_PAIRS = 50_000

DUCKDB_MEMORY = "4GB"
DUCKDB_THREADS = 2
MAX_TEMP_SIZE = "30GB"

# ============================================================
# HELPERS
# ============================================================

def q(path):
    return path.as_posix().replace("'", "''")


def configure(con):
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
    con.execute(f"SET max_temp_directory_size='{MAX_TEMP_SIZE}'")


def name_score_sql(a="s", b="t"):
    return f"""
        GREATEST(
            COALESCE(
                jaro_winkler_similarity(
                    {a}.name_core,
                    {b}.name_core
                ),
                0.0
            ),
            COALESCE(
                jaro_winkler_similarity(
                    {a}.name_norm,
                    {b}.name_norm
                ),
                0.0
            ),
            COALESCE(
                jaro_winkler_similarity(
                    {a}.name_translit,
                    {b}.name_translit
                ),
                0.0
            )
        )
    """


# ============================================================
# START
# ============================================================

print("=" * 70)
print("COMBINED BLOCKING FAILURE ANALYSIS")
print("=" * 70)

print()
print(f"Threshold       : {THRESHOLD}")
print(f"Max missed rows : {MAX_MISSED_PAIRS}")
print()

start_time = time.time()

con = duckdb.connect()
configure(con)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading S1...")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW s1 AS
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(name_norm AS VARCHAR) AS name_norm,
        CAST(name_core AS VARCHAR) AS name_core,
        CAST(name_translit AS VARCHAR) AS name_translit,
        CAST(address_norm AS VARCHAR) AS address_norm,
        CAST(postal_code AS VARCHAR) AS postal_code,
        CAST(house_number AS VARCHAR) AS house_number,
        CAST(phone AS VARCHAR) AS phone,
        CAST(domain AS VARCHAR) AS domain
    FROM read_parquet('{q(S1_SAMPLE)}')
""")


print("Loading S2...")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW s2 AS
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(name_norm AS VARCHAR) AS name_norm,
        CAST(name_core AS VARCHAR) AS name_core,
        CAST(name_translit AS VARCHAR) AS name_translit,
        CAST(address_norm AS VARCHAR) AS address_norm,
        CAST(postal_code AS VARCHAR) AS postal_code,
        CAST(house_number AS VARCHAR) AS house_number,
        CAST(phone AS VARCHAR) AS phone,
        CAST(domain AS VARCHAR) AS domain
    FROM read_parquet('{q(S2_PARQUET)}')
""")


print("Loading S3...")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW s3 AS
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(name_norm AS VARCHAR) AS name_norm,
        CAST(name_core AS VARCHAR) AS name_core,
        CAST(name_translit AS VARCHAR) AS name_translit,
        CAST(address_norm AS VARCHAR) AS address_norm,
        CAST(postal_code AS VARCHAR) AS postal_code,
        CAST(house_number AS VARCHAR) AS house_number,
        CAST(phone AS VARCHAR) AS phone,
        CAST(domain AS VARCHAR) AS domain
    FROM read_parquet('{q(S3_PARQUET)}')
""")


print("Loading ground truth...")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW gt AS
    SELECT
        CAST(source1_entity_id AS VARCHAR) AS s1_id,
        CAST(target_entity_id AS VARCHAR) AS target_id
    FROM read_parquet('{q(GT_SAMPLE)}')
""")


# ============================================================
# CLASSIFY GT
# ============================================================

print("Classifying ground truth...")

con.execute("""
    CREATE OR REPLACE TEMP VIEW gt_s2 AS
    SELECT g.s1_id, g.target_id
    FROM gt g
    INNER JOIN s2 t
        ON g.target_id = t.entity_id
""")

con.execute("""
    CREATE OR REPLACE TEMP VIEW gt_s3 AS
    SELECT g.s1_id, g.target_id
    FROM gt g
    INNER JOIN s3 t
        ON g.target_id = t.entity_id
""")


# ============================================================
# GET S1 IDS
# ============================================================

s1_ids = con.execute("""
    SELECT entity_id
    FROM s1
    ORDER BY entity_id
""").fetchdf()["entity_id"].tolist()

BATCH_SIZE = 5_000

total_batches = (
    len(s1_ids) + BATCH_SIZE - 1
) // BATCH_SIZE


# ============================================================
# PROCESS ONE TARGET
# ============================================================

def analyze_source(source_name, target_view, gt_view):

    print()
    print("=" * 70)
    print(f"ANALYZING MISSED PAIRS: S1 -> {source_name}")
    print("=" * 70)

    missed_parts = []

    total_gt = con.execute(f"""
        SELECT COUNT(*)
        FROM {gt_view}
    """).fetchone()[0]

    print(f"Ground-truth pairs: {total_gt:,}")

    for batch_no in range(total_batches):

        start = batch_no * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(s1_ids))

        batch_ids = s1_ids[start:end]

        print(
            f"Batch {batch_no + 1}/{total_batches} "
            f"{start + 1:,}-{end:,}",
            end=" ... "
        )

        # ----------------------------------------------------
        # Batch IDs
        # ----------------------------------------------------

        con.execute("""
            DROP TABLE IF EXISTS batch_ids
        """)

        con.execute("""
            CREATE TEMP TABLE batch_ids (
                entity_id VARCHAR
            )
        """)

        df_ids = pd.DataFrame({
            "entity_id": batch_ids
        })

        con.register("batch_df", df_ids)

        con.execute("""
            INSERT INTO batch_ids
            SELECT entity_id
            FROM batch_df
        """)

        con.unregister("batch_df")

        # ----------------------------------------------------
        # Batch S1
        # ----------------------------------------------------

        con.execute("""
            DROP TABLE IF EXISTS batch_s1
        """)

        con.execute("""
            CREATE TEMP TABLE batch_s1 AS
            SELECT s.*
            FROM s1 s
            INNER JOIN batch_ids b
                ON s.entity_id = b.entity_id
        """)

        # ====================================================
        # RECREATE V1
        # ====================================================

        con.execute("""
            DROP TABLE IF EXISTS v1_candidates
        """)

        v1_sql = f"""
        CREATE TEMP TABLE v1_candidates AS

        SELECT DISTINCT s.entity_id AS s1_id,
                        t.entity_id AS target_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_norm IS NOT NULL
           AND s.name_norm <> ''
           AND s.name_norm = t.name_norm

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_core IS NOT NULL
           AND s.name_core <> ''
           AND s.name_core = t.name_core

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_translit IS NOT NULL
           AND s.name_translit <> ''
           AND s.name_translit = t.name_translit

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.phone IS NOT NULL
           AND s.phone <> ''
           AND s.phone = t.phone

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_norm IS NOT NULL
           AND s.name_norm <> ''
           AND s.postal_code IS NOT NULL
           AND s.postal_code <> ''
           AND s.name_norm = t.name_norm
           AND s.postal_code = t.postal_code

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_core IS NOT NULL
           AND s.name_core <> ''
           AND s.postal_code IS NOT NULL
           AND s.postal_code <> ''
           AND s.name_core = t.name_core
           AND s.postal_code = t.postal_code

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_core IS NOT NULL
           AND s.name_core <> ''
           AND s.house_number IS NOT NULL
           AND s.house_number <> ''
           AND s.name_core = t.name_core
           AND s.house_number = t.house_number

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.name_translit IS NOT NULL
           AND s.name_translit <> ''
           AND s.postal_code IS NOT NULL
           AND s.postal_code <> ''
           AND s.name_translit = t.name_translit
           AND s.postal_code = t.postal_code

        UNION

        SELECT DISTINCT s.entity_id,
                        t.entity_id
        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.address_norm IS NOT NULL
           AND s.address_norm <> ''
           AND s.postal_code IS NOT NULL
           AND s.postal_code <> ''
           AND s.address_norm = t.address_norm
           AND s.postal_code = t.postal_code
        """

        con.execute(v1_sql)

        # ====================================================
        # V2
        # ====================================================

        con.execute("""
            DROP TABLE IF EXISTS v2_candidates
        """)

        score = name_score_sql("s", "t")

        v2_sql = f"""
        CREATE TEMP TABLE v2_candidates AS

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.postal_code IS NOT NULL
           AND s.postal_code <> ''
           AND t.postal_code IS NOT NULL
           AND t.postal_code <> ''
           AND s.postal_code = t.postal_code
           AND ({score}) >= {THRESHOLD}

        UNION

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s
        INNER JOIN {target_view} t
            ON s.house_number IS NOT NULL
           AND s.house_number <> ''
           AND t.house_number IS NOT NULL
           AND t.house_number <> ''
           AND s.house_number = t.house_number
           AND ({score}) >= {THRESHOLD}
        """

        con.execute(v2_sql)

        # ====================================================
        # COMBINED
        # ====================================================

        con.execute("""
            DROP TABLE IF EXISTS combined_candidates
        """)

        con.execute("""
            CREATE TEMP TABLE combined_candidates AS

            SELECT s1_id, target_id
            FROM v1_candidates

            UNION

            SELECT s1_id, target_id
            FROM v2_candidates
        """)

        # ====================================================
        # MISSED TRUE PAIRS
        # ====================================================

        # Ground truth pairs NOT retrieved by V1+V2.
        #
        # We only keep IDs here first, which keeps memory low.
        # ====================================================

        missed = con.execute(f"""
            SELECT g.s1_id, g.target_id
            FROM {gt_view} g
            INNER JOIN batch_ids b
              ON g.s1_id = b.entity_id
            LEFT JOIN combined_candidates c
              ON g.s1_id = c.s1_id
             AND g.target_id = c.target_id
            WHERE c.s1_id IS NULL
        """).fetchdf()

        if len(missed) > 0:
            missed_parts.append(missed)

        print(f"missed {len(missed):,}")

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS batch_ids")
        con.execute("DROP TABLE IF EXISTS batch_s1")
        con.execute("DROP TABLE IF EXISTS v1_candidates")
        con.execute("DROP TABLE IF EXISTS v2_candidates")
        con.execute("DROP TABLE IF EXISTS combined_candidates")

    # ========================================================
    # COMBINE MISSES
    # ========================================================

    if not missed_parts:
        print("No missed pairs!")
        return

    missed_df = pd.concat(
        missed_parts,
        ignore_index=True
    )

    missed_df = missed_df.drop_duplicates(
        subset=["s1_id", "target_id"]
    )

    print()
    print(f"TOTAL MISSED TRUE PAIRS: {len(missed_df):,}")

    # Limit detailed analysis to avoid huge output.
    inspect_df = missed_df.head(MAX_MISSED_PAIRS).copy()

    # ========================================================
    # SAVE RAW MISSED IDS
    # ========================================================

    missed_file = (
        OUT_DIR /
        f"missed_combined_{source_name.lower()}.parquet"
    )

    missed_df.to_parquet(
        missed_file,
        index=False
    )

    print(f"Saved all missed IDs: {missed_file}")

    # ========================================================
    # REGISTER MISSED PAIRS
    # ========================================================

    con.register(
        f"missed_{source_name.lower()}_df",
        inspect_df
    )

    # ========================================================
    # ENRICH MISSED PAIRS
    # ========================================================

    enriched = con.execute(f"""
        SELECT

            m.s1_id,
            m.target_id,

            s.name_norm AS s1_name_norm,
            t.name_norm AS target_name_norm,

            s.name_core AS s1_name_core,
            t.name_core AS target_name_core,

            s.name_translit AS s1_name_translit,
            t.name_translit AS target_name_translit,

            s.address_norm AS s1_address,
            t.address_norm AS target_address,

            s.postal_code AS s1_postal,
            t.postal_code AS target_postal,

            s.house_number AS s1_house,
            t.house_number AS target_house,

            s.phone AS s1_phone,
            t.phone AS target_phone,

            s.domain AS s1_domain,
            t.domain AS target_domain,

            {name_score_sql("s", "t")} AS name_jw,

            CASE
                WHEN s.name_norm = t.name_norm
                THEN 1 ELSE 0
            END AS exact_name_norm,

            CASE
                WHEN s.name_core = t.name_core
                THEN 1 ELSE 0
            END AS exact_name_core,

            CASE
                WHEN s.name_translit = t.name_translit
                THEN 1 ELSE 0
            END AS exact_name_translit,

            CASE
                WHEN s.address_norm IS NOT NULL
                 AND t.address_norm IS NOT NULL
                 AND s.address_norm <> ''
                 AND t.address_norm <> ''
                 AND s.address_norm = t.address_norm
                THEN 1 ELSE 0
            END AS exact_address,

            CASE
                WHEN s.postal_code IS NOT NULL
                 AND t.postal_code IS NOT NULL
                 AND s.postal_code <> ''
                 AND t.postal_code <> ''
                 AND s.postal_code = t.postal_code
                THEN 1 ELSE 0
            END AS exact_postal,

            CASE
                WHEN s.house_number IS NOT NULL
                 AND t.house_number IS NOT NULL
                 AND s.house_number <> ''
                 AND t.house_number <> ''
                 AND s.house_number = t.house_number
                THEN 1 ELSE 0
            END AS exact_house,

            CASE
                WHEN s.phone IS NOT NULL
                 AND t.phone IS NOT NULL
                 AND s.phone <> ''
                 AND t.phone <> ''
                 AND s.phone = t.phone
                THEN 1 ELSE 0
            END AS exact_phone,

            CASE
                WHEN s.domain IS NOT NULL
                 AND t.domain IS NOT NULL
                 AND s.domain <> ''
                 AND t.domain <> ''
                 AND s.domain = t.domain
                THEN 1 ELSE 0
            END AS exact_domain

        FROM missed_{source_name.lower()}_df m

        INNER JOIN s1 s
            ON m.s1_id = s.entity_id

        INNER JOIN {target_view} t
            ON m.target_id = t.entity_id
    """).fetchdf()

    con.unregister(
        f"missed_{source_name.lower()}_df"
    )

    # ========================================================
    # SAVE ENRICHED
    # ========================================================

    enriched_file = (
        OUT_DIR /
        f"missed_combined_{source_name.lower()}_enriched.csv"
    )

    enriched.to_csv(
        enriched_file,
        index=False
    )

    print(f"Saved enriched sample: {enriched_file}")

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    print()
    print("-" * 70)
    print(f"MISSED-PAIR SUMMARY: {source_name}")
    print("-" * 70)

    n = len(enriched)

    def pct(col):
        return (
            enriched[col].mean() * 100
            if n > 0 else 0
        )

    print()
    print("Field agreement among missed TRUE pairs:")

    print(
        f"  Exact name_norm     : "
        f"{pct('exact_name_norm'):.2f}%"
    )

    print(
        f"  Exact name_core     : "
        f"{pct('exact_name_core'):.2f}%"
    )

    print(
        f"  Exact translit name : "
        f"{pct('exact_name_translit'):.2f}%"
    )

    print(
        f"  Exact address       : "
        f"{pct('exact_address'):.2f}%"
    )

    print(
        f"  Exact postal        : "
        f"{pct('exact_postal'):.2f}%"
    )

    print(
        f"  Exact house         : "
        f"{pct('exact_house'):.2f}%"
    )

    print(
        f"  Exact phone         : "
        f"{pct('exact_phone'):.2f}%"
    )

    print(
        f"  Exact domain        : "
        f"{pct('exact_domain'):.2f}%"
    )

    print()
    print("Name Jaro-Winkler distribution:")

    print(
        enriched["name_jw"].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # ========================================================
    # IMPORTANT SEGMENTS
    # ========================================================

    print()
    print("Missed-pair segments:")

    print(
        "  Same postal + different name :",
        (
            (
                (enriched["exact_postal"] == 1)
                &
                (enriched["exact_name_core"] == 0)
            ).sum()
        )
    )

    print(
        "  Same house + different name  :",
        (
            (
                (enriched["exact_house"] == 1)
                &
                (enriched["exact_name_core"] == 0)
            ).sum()
        )
    )

    print(
        "  Different postal + same house:",
        (
            (
                (enriched["exact_house"] == 1)
                &
                (enriched["exact_postal"] == 0)
            ).sum()
        )
    )

    print(
        "  No exact postal/house signal  :",
        (
            (
                (enriched["exact_postal"] == 0)
                &
                (enriched["exact_house"] == 0)
            ).sum()
        )
    )

    print(
        "  Name JW >= 0.90               :",
        (enriched["name_jw"] >= 0.90).sum()
    )

    print(
        "  Name JW >= 0.85               :",
        (enriched["name_jw"] >= 0.85).sum()
    )

    print(
        "  Name JW >= 0.80               :",
        (enriched["name_jw"] >= 0.80).sum()
    )

    # ========================================================
    # EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("EXAMPLES OF MISSED TRUE PAIRS")
    print("=" * 70)

    cols = [
        "s1_id",
        "target_id",
        "s1_name_norm",
        "target_name_norm",
        "s1_name_translit",
        "target_name_translit",
        "s1_postal",
        "target_postal",
        "s1_house",
        "target_house",
        "name_jw",
    ]

    print(
        enriched[cols]
        .head(30)
        .to_string(index=False)
    )

    return missed_df, enriched


# ============================================================
# RUN S2
# ============================================================

s2_result = analyze_source(
    "S2",
    "s2",
    "gt_s2"
)


# ============================================================
# RUN S3
# ============================================================

s3_result = analyze_source(
    "S3",
    "s3",
    "gt_s3"
)


# ============================================================
# DONE
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 70)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 70)

print()
print(f"Elapsed time: {elapsed / 60:.2f} minutes")

print()
print("Files are in:")
print(OUT_DIR)

print()
print("NEXT STEP:")
print("Use the missed-pair patterns to design Blocking V3.")
print("Do NOT create the final candidate dataset yet.")