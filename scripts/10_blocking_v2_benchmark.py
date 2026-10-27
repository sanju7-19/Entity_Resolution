from pathlib import Path
import duckdb
import pandas as pd
import math
import time


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

SAMPLE_S1 = (
    ROOT
    / "outputs"
    / "blocking_sample_s1.parquet"
)

SAMPLE_GT = (
    ROOT
    / "outputs"
    / "blocking_sample_gt.parquet"
)

S2 = (
    ROOT
    / "features"
    / "normalized"
    / "train_source2.parquet"
)

S3 = (
    ROOT
    / "features"
    / "normalized"
    / "train_source3.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "blocking_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_DIR = (
    OUTPUT_DIR
    / "duckdb_tmp"
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MEMORY / BATCH SETTINGS
# ============================================================

BATCH_SIZE = 5_000

DUCKDB_MEMORY = "4GB"

THREADS = 2

# We test several fuzzy thresholds.
# Jaro-Winkler is 0..1.
THRESHOLDS = [
    0.80,
    0.85,
    0.90,
]

# ============================================================
# HELPER
# ============================================================

def sql_path(path):

    return (
        str(path)
        .replace("\\", "/")
        .replace("'", "''")
    )


# ============================================================
# DUCKDB
# ============================================================

DB = duckdb.connect()

DB.execute(
    f"SET memory_limit='{DUCKDB_MEMORY}'"
)

DB.execute(
    f"SET threads={THREADS}"
)

DB.execute(
    "SET preserve_insertion_order=false"
)

DB.execute(
    f"SET temp_directory='{sql_path(TEMP_DIR)}'"
)

DB.execute(
    "SET max_temp_directory_size='30GB'"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BLOCKING V2 BENCHMARK")
print("=" * 70)

print()
print(f"S1 sample       : {SAMPLE_S1}")
print(f"S2              : {S2}")
print(f"S3              : {S3}")
print(f"Batch size      : {BATCH_SIZE:,}")
print(f"DuckDB memory   : {DUCKDB_MEMORY}")
print(f"DuckDB threads  : {THREADS}")
print(f"Thresholds      : {THRESHOLDS}")
print()


# ============================================================
# LOAD VIEWS
# ============================================================

print("Loading Parquet views...")

DB.execute(
    f"""
    CREATE OR REPLACE VIEW sample_s1 AS
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(name_norm AS VARCHAR) AS name_norm,
        CAST(name_core AS VARCHAR) AS name_core,
        CAST(name_translit AS VARCHAR) AS name_translit,
        CAST(postal_code AS VARCHAR) AS postal_code,
        CAST(house_number AS VARCHAR) AS house_number
    FROM read_parquet(
        '{sql_path(SAMPLE_S1)}'
    )
    """
)

DB.execute(
    f"""
    CREATE OR REPLACE VIEW s2 AS
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(name_norm AS VARCHAR) AS name_norm,
        CAST(name_core AS VARCHAR) AS name_core,
        CAST(name_translit AS VARCHAR) AS name_translit,
        CAST(postal_code AS VARCHAR) AS postal_code,
        CAST(house_number AS VARCHAR) AS house_number
    FROM read_parquet(
        '{sql_path(S2)}'
    )
    """
)

DB.execute(
    f"""
    CREATE OR REPLACE VIEW s3 AS
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(name_norm AS VARCHAR) AS name_norm,
        CAST(name_core AS VARCHAR) AS name_core,
        CAST(name_translit AS VARCHAR) AS name_translit,
        CAST(postal_code AS VARCHAR) AS postal_code,
        CAST(house_number AS VARCHAR) AS house_number
    FROM read_parquet(
        '{sql_path(S3)}'
    )
    """
)


# ============================================================
# GROUND TRUTH
# ============================================================

print("Loading ground truth...")

DB.execute(
    f"""
    CREATE OR REPLACE VIEW sample_gt AS
    SELECT
        CAST(source1_entity_id AS VARCHAR)
            AS s1_id,
        CAST(target_entity_id AS VARCHAR)
            AS target_id
    FROM read_parquet(
        '{sql_path(SAMPLE_GT)}'
    )
    """
)


# ============================================================
# CLASSIFY GROUND TRUTH INTO S2 / S3
# ============================================================

print("Classifying ground-truth targets...")


DB.execute(
    """
    CREATE OR REPLACE TEMP VIEW gt_s2 AS

    SELECT DISTINCT
        g.s1_id,
        g.target_id

    FROM sample_gt g

    INNER JOIN s2 t
        ON g.target_id = t.entity_id

    WHERE NOT EXISTS (

        SELECT 1

        FROM s3 t3

        WHERE t3.entity_id = g.target_id
    )
    """
)


DB.execute(
    """
    CREATE OR REPLACE TEMP VIEW gt_s3 AS

    SELECT DISTINCT
        g.s1_id,
        g.target_id

    FROM sample_gt g

    INNER JOIN s3 t

        ON g.target_id = t.entity_id

    WHERE NOT EXISTS (

        SELECT 1

        FROM s2 t2

        WHERE t2.entity_id = g.target_id
    )
    """
)


# ============================================================
# GROUND TRUTH COUNTS
# ============================================================

gt_s2_count = DB.execute(
    """
    SELECT COUNT(*)
    FROM gt_s2
    """
).fetchone()[0]


gt_s3_count = DB.execute(
    """
    SELECT COUNT(*)
    FROM gt_s3
    """
).fetchone()[0]


print()
print("Ground truth:")
print(
    f"  S2 true pairs : {gt_s2_count:,}"
)
print(
    f"  S3 true pairs : {gt_s3_count:,}"
)
print(
    f"  Total         : "
    f"{gt_s2_count + gt_s3_count:,}"
)


# ============================================================
# S1 IDS
# ============================================================

s1_ids = DB.execute(
    """
    SELECT entity_id
    FROM sample_s1
    ORDER BY entity_id
    """
).fetchdf()["entity_id"].tolist()


TOTAL_S1 = len(s1_ids)

TOTAL_BATCHES = math.ceil(
    TOTAL_S1 / BATCH_SIZE
)


print()
print(
    f"S1 sample size : {TOTAL_S1:,}"
)

print(
    f"Total batches  : {TOTAL_BATCHES}"
)


# ============================================================
# TEST DUCKDB FUZZY FUNCTIONS
# ============================================================

print()
print("Testing DuckDB similarity functions...")

test_result = DB.execute(
    """
    SELECT
        jaro_winkler_similarity(
            'mitchell and gee',
            'mitche1l and gee'
        ) AS jw,

        levenshtein(
            'mitchell and gee',
            'mitche1l and gee'
        ) AS lev
    """
).fetchone()


print(
    f"  Jaro-Winkler : {test_result[0]:.4f}"
)

print(
    f"  Levenshtein  : {test_result[1]}"
)


# ============================================================
# NAME SCORE EXPRESSION
# ============================================================

NAME_SCORE = """
GREATEST(

    jaro_winkler_similarity(
        s.name_core,
        t.name_core
    ),

    jaro_winkler_similarity(
        s.name_norm,
        t.name_norm
    ),

    jaro_winkler_similarity(
        s.name_translit,
        t.name_translit
    )

)
"""


# ============================================================
# ANALYZE ONE SOURCE
# ============================================================

def analyze_source(
    source_name,
    target_view,
    gt_view,
    total_gt
):

    print()
    print("=" * 70)
    print(f"S1 -> {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    results = []

    retrieved_union = {
        threshold: set()
        for threshold in THRESHOLDS
    }

    total_candidate_counts = {
        "postal": {
            threshold: 0
            for threshold in THRESHOLDS
        },

        "house": {
            threshold: 0
            for threshold in THRESHOLDS
        },
    }

    # --------------------------------------------------------
    # Batch loop
    # --------------------------------------------------------

    for batch_number, start in enumerate(
        range(
            0,
            TOTAL_S1,
            BATCH_SIZE
        ),
        start=1
    ):

        end = min(
            start + BATCH_SIZE,
            TOTAL_S1
        )

        batch_ids = s1_ids[start:end]

        print()
        print(
            f"Batch {batch_number}/{TOTAL_BATCHES}"
            f"  "
            f"S1 {start + 1:,}-{end:,}"
        )

        # ----------------------------------------------------
        # Create current S1 batch
        # ----------------------------------------------------

        DB.execute(
            "DROP TABLE IF EXISTS batch_s1"
        )

        DB.execute(
            """
            CREATE TEMP TABLE batch_ids (
                entity_id VARCHAR
            )
            """
        )

        # Insert IDs in one executemany call.
        DB.executemany(
            """
            INSERT INTO batch_ids
            VALUES (?)
            """,
            [
                (x,)
                for x in batch_ids
            ]
        )

        DB.execute(
            """
            CREATE TEMP TABLE batch_s1 AS

            SELECT s.*

            FROM sample_s1 s

            INNER JOIN batch_ids b

                ON s.entity_id = b.entity_id
            """
        )

        DB.execute(
            "DROP TABLE batch_ids"
        )

        # ----------------------------------------------------
        # Process each blocking route
        # ----------------------------------------------------

        routes = {

            "postal": """
                s.postal_code IS NOT NULL
                AND TRIM(s.postal_code) <> ''

                AND t.postal_code IS NOT NULL
                AND TRIM(t.postal_code) <> ''

                AND s.name_core IS NOT NULL
                AND TRIM(s.name_core) <> ''

                AND t.name_core IS NOT NULL
                AND TRIM(t.name_core) <> ''

                AND s.postal_code = t.postal_code
            """,

            "house": """
                s.house_number IS NOT NULL
                AND TRIM(s.house_number) <> ''

                AND t.house_number IS NOT NULL
                AND TRIM(t.house_number) <> ''

                AND s.name_core IS NOT NULL
                AND TRIM(s.name_core) <> ''

                AND t.name_core IS NOT NULL
                AND TRIM(t.name_core) <> ''

                AND s.house_number = t.house_number
            """,
        }

        for route_name, join_condition in routes.items():

            # ------------------------------------------------
            # Calculate fuzzy score once.
            # ------------------------------------------------

            candidate_sql = f"""
            SELECT

                s.entity_id AS s1_id,

                t.entity_id AS target_id,

                {NAME_SCORE}
                    AS name_score

            FROM batch_s1 s

            INNER JOIN {target_view} t

                ON {join_condition}
            """

            DB.execute(
                "DROP TABLE IF EXISTS v2_candidates"
            )

            DB.execute(
                f"""
                CREATE TEMP TABLE v2_candidates AS

                {candidate_sql}
                """
            )

            # ------------------------------------------------
            # Test thresholds
            # ------------------------------------------------

            for threshold in THRESHOLDS:

                candidate_count = DB.execute(
                    f"""
                    SELECT COUNT(*)

                    FROM v2_candidates

                    WHERE name_score >= {threshold}
                    """
                ).fetchone()[0]

                total_candidate_counts[
                    route_name
                ][threshold] += candidate_count

                # --------------------------------------------
                # True-pair hits
                # --------------------------------------------

                hits = DB.execute(
                    f"""
                    SELECT DISTINCT

                        c.s1_id,
                        c.target_id

                    FROM v2_candidates c

                    INNER JOIN {gt_view} g

                        ON c.s1_id = g.s1_id
                        AND c.target_id = g.target_id

                    WHERE c.name_score >= {threshold}
                    """
                ).fetchall()

                for s1_id, target_id in hits:

                    retrieved_union[
                        threshold
                    ].add(
                        (
                            str(s1_id),
                            str(target_id)
                        )
                    )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        DB.execute(
            "DROP TABLE IF EXISTS v2_candidates"
        )

        DB.execute(
            "DROP TABLE IF EXISTS batch_s1"
        )

        print(
            "  completed"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("-" * 70)
    print(f"RESULTS: S1 -> {source_name}")
    print("-" * 70)

    for threshold in THRESHOLDS:

        unique_hits = len(
            retrieved_union[threshold]
        )

        recall = (
            unique_hits / total_gt
            if total_gt > 0
            else 0
        )

        postal_candidates = (
            total_candidate_counts[
                "postal"
            ][threshold]
        )

        house_candidates = (
            total_candidate_counts[
                "house"
            ][threshold]
        )

        total_candidates = (
            postal_candidates
            + house_candidates
        )

        avg_candidates = (
            total_candidates / TOTAL_S1
        )

        print()
        print(
            f"Threshold {threshold:.2f}"
        )

        print(
            f"  Union true pairs : "
            f"{unique_hits:,}"
        )

        print(
            f"  Recall           : "
            f"{recall * 100:.4f}%"
        )

        print(
            f"  Postal candidates: "
            f"{postal_candidates:,}"
        )

        print(
            f"  House candidates : "
            f"{house_candidates:,}"
        )

        print(
            f"  Total candidates : "
            f"{total_candidates:,}"
        )

        print(
            f"  Avg/S1           : "
            f"{avg_candidates:.2f}"
        )

        results.append(
            {
                "target_source": source_name,
                "threshold": threshold,
                "postal_candidates": postal_candidates,
                "house_candidates": house_candidates,
                "total_candidates": total_candidates,
                "avg_candidates_per_s1": avg_candidates,
                "true_pairs_retrieved": unique_hits,
                "true_pairs_total": total_gt,
                "blocking_recall": recall,
            }
        )

    return results


# ============================================================
# RUN S2
# ============================================================

start_time = time.time()

results_s2 = analyze_source(
    "S2",
    "s2",
    "gt_s2",
    gt_s2_count
)


# ============================================================
# RUN S3
# ============================================================

results_s3 = analyze_source(
    "S3",
    "s3",
    "gt_s3",
    gt_s3_count
)


# ============================================================
# SAVE RESULTS
# ============================================================

all_results = (
    results_s2
    + results_s3
)

result_df = pd.DataFrame(
    all_results
)

output_file = (
    OUTPUT_DIR
    / "blocking_v2_benchmark.csv"
)

result_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 70)
print("BLOCKING V2 COMPLETE")
print("=" * 70)

print()
print(
    result_df.to_string(
        index=False
    )
)

print()
print(
    f"Results saved to:"
    f"\n{output_file}"
)

print()
print(
    f"Elapsed time: "
    f"{elapsed / 60:.2f} minutes"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is a benchmark only."
)

print(
    "Do NOT create the final candidate dataset yet."
)

print(
    "Choose the V2 threshold only after comparing "
    "recall versus candidate volume."
)