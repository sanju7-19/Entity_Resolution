from pathlib import Path
import duckdb
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

SAMPLE_S1 = ROOT / "outputs" / "blocking_sample_s1.parquet"
SAMPLE_GT = ROOT / "outputs" / "blocking_sample_gt.parquet"
BENCHMARK = ROOT / "outputs" / "blocking_sample_benchmark.csv"

S2 = ROOT / "features" / "normalized" / "train_source2.parquet"
S3 = ROOT / "features" / "normalized" / "train_source3.parquet"

OUT_DIR = ROOT / "outputs" / "blocking_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# Smaller = lower RAM usage
BATCH_SIZE = 10_000

# ============================================================
# DUCKDB
# ============================================================

DB = duckdb.connect()

DB.execute("SET threads=2")
DB.execute("SET memory_limit='5GB'")
DB.execute("SET preserve_insertion_order=false")

# ============================================================
# HELPERS
# ============================================================

def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BATCHED BLOCKING FAILURE ANALYSIS")
print("=" * 70)

print(f"\nBatch size : {BATCH_SIZE:,}")
print("DuckDB threads : 2")
print("DuckDB memory : 5GB")

# ============================================================
# LOAD VIEWS
# ============================================================

DB.execute(f"""
CREATE OR REPLACE VIEW sample_s1 AS
SELECT *
FROM read_parquet('{sql_path(SAMPLE_S1)}')
""")

DB.execute(f"""
CREATE OR REPLACE VIEW sample_gt AS
SELECT *
FROM read_parquet('{sql_path(SAMPLE_GT)}')
""")

DB.execute(f"""
CREATE OR REPLACE VIEW s2 AS
SELECT *
FROM read_parquet('{sql_path(S2)}')
""")

DB.execute(f"""
CREATE OR REPLACE VIEW s3 AS
SELECT *
FROM read_parquet('{sql_path(S3)}')
""")

# ============================================================
# GROUND TRUTH CLASSIFICATION
# ============================================================

print("\nClassifying ground truth...")

DB.execute("""
CREATE OR REPLACE TEMP VIEW gt_s2 AS
SELECT DISTINCT
    CAST(g.source1_entity_id AS VARCHAR) AS s1_id,
    CAST(g.target_entity_id AS VARCHAR) AS target_id
FROM sample_gt g
INNER JOIN s2 t
    ON CAST(g.target_entity_id AS VARCHAR)
     = CAST(t.entity_id AS VARCHAR)
WHERE NOT EXISTS (
    SELECT 1
    FROM s3 t3
    WHERE CAST(g.target_entity_id AS VARCHAR)
        = CAST(t3.entity_id AS VARCHAR)
)
""")

DB.execute("""
CREATE OR REPLACE TEMP VIEW gt_s3 AS
SELECT DISTINCT
    CAST(g.source1_entity_id AS VARCHAR) AS s1_id,
    CAST(g.target_entity_id AS VARCHAR) AS target_id
FROM sample_gt g
INNER JOIN s3 t
    ON CAST(g.target_entity_id AS VARCHAR)
     = CAST(t.entity_id AS VARCHAR)
WHERE NOT EXISTS (
    SELECT 1
    FROM s2 t2
    WHERE CAST(g.target_entity_id AS VARCHAR)
        = CAST(t2.entity_id AS VARCHAR)
)
""")

# ============================================================
# RULES
# ============================================================

rules = {

    "exact_name_norm": """
        s.name_norm IS NOT NULL
        AND s.name_norm <> ''
        AND t.name_norm = s.name_norm
    """,

    "exact_name_core": """
        s.name_core IS NOT NULL
        AND s.name_core <> ''
        AND t.name_core = s.name_core
    """,

    "exact_name_translit": """
        s.name_translit IS NOT NULL
        AND s.name_translit <> ''
        AND t.name_translit = s.name_translit
    """,

    "exact_domain": """
        s.domain IS NOT NULL
        AND s.domain <> ''
        AND t.domain = s.domain
    """,

    "exact_phone": """
        s.phone IS NOT NULL
        AND s.phone <> ''
        AND t.phone = s.phone
    """,

    "name_norm_postal": """
        s.name_norm IS NOT NULL
        AND s.name_norm <> ''
        AND s.postal_code IS NOT NULL
        AND s.postal_code <> ''
        AND t.name_norm = s.name_norm
        AND t.postal_code = s.postal_code
    """,

    "name_core_postal": """
        s.name_core IS NOT NULL
        AND s.name_core <> ''
        AND s.postal_code IS NOT NULL
        AND s.postal_code <> ''
        AND t.name_core = s.name_core
        AND t.postal_code = s.postal_code
    """,

    "name_core_house": """
        s.name_core IS NOT NULL
        AND s.name_core <> ''
        AND s.house_number IS NOT NULL
        AND s.house_number <> ''
        AND t.name_core = s.name_core
        AND t.house_number = s.house_number
    """,

    "name_translit_postal": """
        s.name_translit IS NOT NULL
        AND s.name_translit <> ''
        AND s.postal_code IS NOT NULL
        AND s.postal_code <> ''
        AND t.name_translit = s.name_translit
        AND t.postal_code = s.postal_code
    """,

    "address_postal": """
        s.address_norm IS NOT NULL
        AND s.address_norm <> ''
        AND s.postal_code IS NOT NULL
        AND s.postal_code <> ''
        AND t.address_norm = s.address_norm
        AND t.postal_code = s.postal_code
    """
}

# ============================================================
# GET S1 IDS
# ============================================================

s1_ids = DB.execute("""
SELECT
    CAST(entity_id AS VARCHAR) AS s1_id
FROM sample_s1
ORDER BY entity_id
""").fetchdf()["s1_id"].tolist()

total_s1 = len(s1_ids)

print(f"\nSample S1 records : {total_s1:,}")

# ============================================================
# MAIN BATCH FUNCTION
# ============================================================

def analyze_source(source_name, target_view, gt_view):

    print("\n" + "=" * 70)
    print(f"S1 -> {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Total ground truth
    # --------------------------------------------------------

    total_gt = DB.execute(f"""
        SELECT COUNT(*)
        FROM {gt_view}
    """).fetchone()[0]

    print(f"True pairs : {total_gt:,}")

    # --------------------------------------------------------
    # Store only retrieved TRUE pairs
    #
    # This remains tiny compared with all candidates.
    # --------------------------------------------------------

    retrieved_pairs = set()

    # --------------------------------------------------------
    # Per-rule diagnostics
    # --------------------------------------------------------

    rule_hits = {
        rule: 0
        for rule in rules
    }

    # --------------------------------------------------------
    # Process S1 in batches
    # --------------------------------------------------------

    for start in range(0, total_s1, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total_s1)

        batch_ids = s1_ids[start:end]

        print(
            f"\nBatch "
            f"{start // BATCH_SIZE + 1}"
            f" / "
            f"{(total_s1 + BATCH_SIZE - 1) // BATCH_SIZE}"
            f"   S1 rows {start + 1:,}-{end:,}"
        )

        # ----------------------------------------------------
        # Create batch S1 table
        # ----------------------------------------------------

        DB.execute("DROP TABLE IF EXISTS batch_s1")

        DB.execute("""
        CREATE TEMP TABLE batch_s1 AS

        SELECT *
        FROM sample_s1
        WHERE CAST(entity_id AS VARCHAR) IN (
            SELECT *
            FROM UNNEST(?)
        )
        """, [batch_ids])

        # ----------------------------------------------------
        # Process every blocking rule
        # ----------------------------------------------------

        for rule_name, condition in rules.items():

            query = f"""
            SELECT DISTINCT

                CAST(s.entity_id AS VARCHAR) AS s1_id,
                CAST(t.entity_id AS VARCHAR) AS target_id

            FROM batch_s1 s

            INNER JOIN {target_view} t

                ON {condition}
            """

            # Only retain pairs that are actually ground truth.
            # This is dramatically smaller than materializing
            # all candidates.

            query = f"""
            SELECT DISTINCT

                g.s1_id,
                g.target_id

            FROM {gt_view} g

            INNER JOIN (
                {query}
            ) c

                ON g.s1_id = c.s1_id
                AND g.target_id = c.target_id
            """

            hits = DB.execute(query).fetchall()

            rule_hits[rule_name] += len(hits)

            for pair in hits:
                retrieved_pairs.add(
                    (pair[0], pair[1])
                )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        DB.execute("DROP TABLE IF EXISTS batch_s1")

        print(
            f"  Retrieved unique true pairs so far: "
            f"{len(retrieved_pairs):,}"
        )

    # ========================================================
    # FINAL RECALL
    # ========================================================

    unique_hits = len(retrieved_pairs)

    recall = (
        unique_hits / total_gt
        if total_gt > 0
        else 0
    )

    print("\n" + "-" * 70)

    print(f"Unique true pairs retrieved : {unique_hits:,}")
    print(f"True pairs                  : {total_gt:,}")

    print(
        f"UNION BLOCKING RECALL       : "
        f"{recall * 100:.4f}%"
    )

    print(
        f"MISSED TRUE PAIRS           : "
        f"{total_gt - unique_hits:,}"
    )

    print(
        f"MISS RATE                   : "
        f"{(1 - recall) * 100:.4f}%"
    )

    # ========================================================
    # RULE-LEVEL RESULTS
    # ========================================================

    print("\nRule-level true-pair retrieval:")

    for rule_name, count in rule_hits.items():

        rule_recall = (
            count / total_gt
            if total_gt > 0
            else 0
        )

        print(
            f"{rule_name:25s}"
            f" {count:10,}"
            f"  {rule_recall * 100:8.3f}%"
        )

    # ========================================================
    # SAVE MISSED PAIRS
    # ========================================================

    print("\nCreating missed-pair table...")

    # Put retrieved pairs into a small temporary table.

    DB.execute("DROP TABLE IF EXISTS retrieved_pairs")

    DB.execute("""
    CREATE TEMP TABLE retrieved_pairs (
        s1_id VARCHAR,
        target_id VARCHAR
    )
    """)

    if retrieved_pairs:

        DB.executemany(
            """
            INSERT INTO retrieved_pairs
            VALUES (?, ?)
            """,
            list(retrieved_pairs)
        )

    # --------------------------------------------------------
    # Extract missed pairs
    # --------------------------------------------------------

    missed_query = f"""

    SELECT
        g.s1_id,
        g.target_id

    FROM {gt_view} g

    LEFT JOIN retrieved_pairs r

        ON g.s1_id = r.s1_id
        AND g.target_id = r.target_id

    WHERE r.s1_id IS NULL

    """

    missed = DB.execute(missed_query).fetchdf()

    missed_path = (
        OUT_DIR /
        f"missed_{source_name.lower()}.parquet"
    )

    missed.to_parquet(
        missed_path,
        index=False,
        compression="zstd"
    )

    print(
        f"\nMissed pairs saved: {missed_path}"
    )

    print(
        f"Missed rows       : {len(missed):,}"
    )

    # ========================================================
    # RETURN MISSED PAIRS
    # ========================================================

    return missed


# ============================================================
# RUN S2
# ============================================================

missed_s2 = analyze_source(
    "S2",
    "s2",
    "gt_s2"
)

# ============================================================
# RUN S3
# ============================================================

missed_s3 = analyze_source(
    "S3",
    "s3",
    "gt_s3"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BATCHED FAILURE ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput files:")

print(
    OUT_DIR / "missed_s2.parquet"
)

print(
    OUT_DIR / "missed_s3.parquet"
)

print("\nNext step:")
print("Inspect missed pairs and design Blocking V2.")