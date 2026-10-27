from pathlib import Path
import duckdb
import pandas as pd
import time
import os
import math
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Users\sanja\Downloads\amazon_ml")

S1_PATH = ROOT / "features" / "normalized" / "train_source1.parquet"
S2_PATH = ROOT / "features" / "normalized" / "train_source2.parquet"
S3_PATH = ROOT / "features" / "normalized" / "train_source3.parquet"

GT_PATH = ROOT / "data" / "ground_truth" / "train_ground_truth.tsv"

OUT_DIR = ROOT / "outputs" / "blocking_v3_40pct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = ROOT / "outputs" / "duckdb_temp_v3"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Only 40% of S1 is used for the benchmark.
S1_SAMPLE_FRACTION = 0.40

# Your machine currently has limited free RAM.
DUCKDB_MEMORY = "4GB"

# Use 4 CPU threads.
CPU_THREADS = 4

# Process S1 in batches.
BATCH_SIZE = 10_000

# Keep V2 threshold identical to previous benchmark.
NAME_JW_THRESHOLD = 0.80

# Reproducible sample.
RANDOM_SEED = 42

# ============================================================
# CONNECT DUCKDB
# ============================================================

con = duckdb.connect()

con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
con.execute(f"SET threads={CPU_THREADS}")
con.execute(f"SET temp_directory='{TEMP_DIR}'")
con.execute("SET preserve_insertion_order=false")

# ============================================================
# HELPERS
# ============================================================

def q(path):
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def name_token_key_sql(column):

    return f"""
        array_to_string(
            list_sort(
                list_distinct(
                    list_filter(
                        string_split(
                            regexp_replace(
                                lower(coalesce({column}, '')),
                                '[^a-z0-9]+',
                                ' ',
                                'g'
                            ),
                            ' '
                        ),
                        x -> length(trim(x)) >= 2
                    )
                )
            ),
            ' '
        )
    """


# ============================================================
# CREATE SOURCE VIEWS
# ============================================================

def create_views():

    print("Creating source views...")

    # --------------------------------------------------------
    # S1 FULL VIEW
    # --------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE VIEW s1_full AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            name_core,
            name_translit,
            address_norm,
            address_translit,
            postal_code,
            house_number,
            phone,
            domain,

            {name_token_key_sql("name_norm")}
                AS name_token_key

        FROM read_parquet({q(S1_PATH)})
    """)

    # --------------------------------------------------------
    # S2 FULL VIEW
    # --------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE VIEW s2 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            name_core,
            name_translit,
            address_norm,
            address_translit,
            postal_code,
            house_number,
            phone,
            domain,

            {name_token_key_sql("name_norm")}
                AS name_token_key

        FROM read_parquet({q(S2_PATH)})
    """)

    # --------------------------------------------------------
    # S3 FULL VIEW
    # --------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE VIEW s3 AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            name_core,
            name_translit,
            address_norm,
            address_translit,
            postal_code,
            house_number,
            phone,
            domain,

            {name_token_key_sql("name_norm")}
                AS name_token_key

        FROM read_parquet({q(S3_PATH)})
    """)

    print("Source views created.")


# ============================================================
# CREATE 40% S1 SAMPLE
# ============================================================

def create_s1_sample():

    print()
    print("=" * 70)
    print("CREATING 40% S1 SAMPLE")
    print("=" * 70)

    # Important:
    # Only S1 is sampled.
    # S2 and S3 remain FULL.

    con.execute("""
        CREATE OR REPLACE TABLE s1_sample AS
        SELECT *
        FROM s1_full
        USING SAMPLE 40%
        (reservoir, 42)
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM s1_sample"
    ).fetchone()[0]

    total = con.execute(
        "SELECT COUNT(*) FROM s1_full"
    ).fetchone()[0]

    percentage = count / total * 100

    print(f"Full S1 rows   : {total:,}")
    print(f"Sample S1 rows : {count:,}")
    print(f"Actual sample  : {percentage:.2f}%")

    return count


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print()
    print("Loading ground truth...")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW gt_raw AS
        SELECT *
        FROM read_csv(
            {q(GT_PATH)},
            delim='\\t',
            header=true,
            columns={{
                'source1_entity_id':'VARCHAR',
                'matched_entity_ids':'VARCHAR'
            }}
        )
    """)

    con.execute("""
        CREATE OR REPLACE TABLE gt_pairs AS
        SELECT
            source1_entity_id AS s1_id,
            trim(x.target_id) AS target_id
        FROM gt_raw,
        unnest(
            string_split(matched_entity_ids, ',')
        ) AS x(target_id)
        WHERE trim(x.target_id) <> ''
    """)

    total = con.execute(
        "SELECT COUNT(*) FROM gt_pairs"
    ).fetchone()[0]

    # Only GT belonging to sampled S1.
    con.execute("""
        CREATE OR REPLACE TABLE gt_sample AS
        SELECT g.*
        FROM gt_pairs g
        INNER JOIN s1_sample s
            ON g.s1_id = s.entity_id
    """)

    sample_count = con.execute(
        "SELECT COUNT(*) FROM gt_sample"
    ).fetchone()[0]

    print(f"Full GT pairs       : {total:,}")
    print(f"Sample GT pairs     : {sample_count:,}")

    return sample_count


# ============================================================
# BLOCKING QUERY
# ============================================================

def generate_candidates(target_view):

    query = f"""

    WITH

    ----------------------------------------------------------
    -- V1: EXACT NAME
    ----------------------------------------------------------

    exact_name AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_norm = t.name_norm

        WHERE
            s.name_norm IS NOT NULL
            AND length(s.name_norm) >= 4
    ),

    ----------------------------------------------------------
    -- V1: EXACT NAME CORE
    ----------------------------------------------------------

    exact_core AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_core = t.name_core

        WHERE
            s.name_core IS NOT NULL
            AND length(s.name_core) >= 4
    ),

    ----------------------------------------------------------
    -- V1: EXACT TRANSLITERATION
    ----------------------------------------------------------

    exact_translit AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_translit = t.name_translit

        WHERE
            s.name_translit IS NOT NULL
            AND length(s.name_translit) >= 4
    ),

    ----------------------------------------------------------
    -- V1: PHONE
    ----------------------------------------------------------

    exact_phone AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.phone = t.phone

        WHERE
            s.phone IS NOT NULL
            AND length(s.phone) >= 7
    ),

    ----------------------------------------------------------
    -- V1: NAME + POSTAL
    ----------------------------------------------------------

    name_postal AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_core = t.name_core
           AND s.postal_code = t.postal_code

        WHERE
            s.name_core IS NOT NULL
            AND s.postal_code IS NOT NULL
    ),

    ----------------------------------------------------------
    -- V1: NAME + HOUSE
    ----------------------------------------------------------

    name_house AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_core = t.name_core
           AND s.house_number = t.house_number

        WHERE
            s.name_core IS NOT NULL
            AND s.house_number IS NOT NULL
    ),

    ----------------------------------------------------------
    -- V2: POSTAL + FUZZY NAME
    ----------------------------------------------------------

    fuzzy_postal AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.postal_code = t.postal_code

        WHERE
            s.postal_code IS NOT NULL
            AND length(s.postal_code) >= 3

            AND greatest(

                jaro_winkler_similarity(
                    coalesce(s.name_core, ''),
                    coalesce(t.name_core, '')
                ),

                jaro_winkler_similarity(
                    coalesce(s.name_norm, ''),
                    coalesce(t.name_norm, '')
                ),

                jaro_winkler_similarity(
                    coalesce(s.name_translit, ''),
                    coalesce(t.name_translit, '')
                )

            ) >= {NAME_JW_THRESHOLD}
    ),

    ----------------------------------------------------------
    -- V2: HOUSE + FUZZY NAME
    ----------------------------------------------------------

    fuzzy_house AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.house_number = t.house_number

        WHERE
            s.house_number IS NOT NULL

            AND greatest(

                jaro_winkler_similarity(
                    coalesce(s.name_core, ''),
                    coalesce(t.name_core, '')
                ),

                jaro_winkler_similarity(
                    coalesce(s.name_norm, ''),
                    coalesce(t.name_norm, '')
                ),

                jaro_winkler_similarity(
                    coalesce(s.name_translit, ''),
                    coalesce(t.name_translit, '')
                )

            ) >= {NAME_JW_THRESHOLD}
    ),

    ----------------------------------------------------------
    -- V3: TOKEN ORDER NORMALIZATION
    --
    -- Example:
    --
    -- "f 5 better grading llc"
    -- "f 5 llc grading better"
    --
    -- become the same token key.
    ----------------------------------------------------------

    token_key AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_token_key = t.name_token_key

        WHERE
            s.name_token_key IS NOT NULL
            AND length(s.name_token_key) >= 6
    ),

    ----------------------------------------------------------
    -- V3: TOKEN KEY + POSTAL
    ----------------------------------------------------------

    token_postal AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_token_key = t.name_token_key
           AND s.postal_code = t.postal_code

        WHERE
            s.name_token_key IS NOT NULL
            AND s.postal_code IS NOT NULL
    ),

    ----------------------------------------------------------
    -- V3: TOKEN KEY + HOUSE
    ----------------------------------------------------------

    token_house AS (

        SELECT DISTINCT
            s.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM batch_s1 s

        INNER JOIN {target_view} t
            ON s.name_token_key = t.name_token_key
           AND s.house_number = t.house_number

        WHERE
            s.name_token_key IS NOT NULL
            AND s.house_number IS NOT NULL
    )

    ----------------------------------------------------------
    -- UNION ALL BLOCKING RULES
    ----------------------------------------------------------

    SELECT DISTINCT
        s1_id,
        target_id

    FROM (

        SELECT * FROM exact_name

        UNION ALL

        SELECT * FROM exact_core

        UNION ALL

        SELECT * FROM exact_translit

        UNION ALL

        SELECT * FROM exact_phone

        UNION ALL

        SELECT * FROM name_postal

        UNION ALL

        SELECT * FROM name_house

        UNION ALL

        SELECT * FROM fuzzy_postal

        UNION ALL

        SELECT * FROM fuzzy_house

        UNION ALL

        SELECT * FROM token_key

        UNION ALL

        SELECT * FROM token_postal

        UNION ALL

        SELECT * FROM token_house

    )

    """

    return con.execute(query).fetchdf()


# ============================================================
# ANALYZE SOURCE
# ============================================================

def analyze_source(source_name, target_view):

    print()
    print("=" * 70)
    print(f"S1 -> {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Get sampled S1 IDs
    # --------------------------------------------------------

    s1_ids = con.execute("""
        SELECT entity_id
        FROM s1_sample
        ORDER BY entity_id
    """).fetchdf()

    total_s1 = len(s1_ids)

    n_batches = math.ceil(
        total_s1 / BATCH_SIZE
    )

    print(f"Sampled S1 rows : {total_s1:,}")
    print(f"Batches         : {n_batches:,}")
    print()

    # --------------------------------------------------------
    # Temporary candidate files
    # --------------------------------------------------------

    source_temp_dir = (
        OUT_DIR / f"temp_{source_name.lower()}"
    )

    if source_temp_dir.exists():
        shutil.rmtree(source_temp_dir)

    source_temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total_candidates = 0
    total_true = 0
    total_gt = 0

    start_time = time.time()

    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    for batch_number, start_idx in enumerate(
        range(0, total_s1, BATCH_SIZE),
        start=1
    ):

        end_idx = min(
            start_idx + BATCH_SIZE,
            total_s1
        )

        batch_ids_df = s1_ids.iloc[
            start_idx:end_idx
        ]

        # ----------------------------------------------------
        # Register batch IDs
        # ----------------------------------------------------

        con.execute(
            "DROP TABLE IF EXISTS batch_s1"
        )

        con.register(
            "batch_ids_df",
            batch_ids_df
        )

        con.execute("""
            CREATE TEMP TABLE batch_s1 AS

            SELECT s.*

            FROM s1_sample s

            INNER JOIN batch_ids_df b
                ON s.entity_id = b.entity_id
        """)

        con.unregister(
            "batch_ids_df"
        )

        # ----------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------

        batch_candidates = generate_candidates(
            target_view
        )

        batch_candidate_count = len(
            batch_candidates
        )

        # ----------------------------------------------------
        # GT for this batch
        # ----------------------------------------------------

        con.register(
            "batch_candidates_df",
            batch_candidates
        )

        batch_gt = con.execute("""
            SELECT COUNT(*)

            FROM gt_sample g

            INNER JOIN batch_s1 b
                ON g.s1_id = b.entity_id
        """).fetchone()[0]

        batch_true = con.execute("""
            SELECT COUNT(*)

            FROM batch_candidates_df c

            INNER JOIN gt_sample g

                ON c.s1_id = g.s1_id
               AND c.target_id = g.target_id

            INNER JOIN batch_s1 b
                ON c.s1_id = b.entity_id
        """).fetchone()[0]

        # ----------------------------------------------------
        # Save batch to disk
        # ----------------------------------------------------

        batch_path = (
            source_temp_dir
            / f"batch_{batch_number:04d}.parquet"
        )

        con.execute(f"""
            COPY (
                SELECT *
                FROM batch_candidates_df
            )
            TO {q(batch_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        con.unregister(
            "batch_candidates_df"
        )

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        total_candidates += batch_candidate_count
        total_true += batch_true
        total_gt += batch_gt

        batch_recall = (
            batch_true / batch_gt
            if batch_gt > 0
            else 0
        )

        elapsed = time.time() - start_time

        print(
            f"Batch {batch_number:>3}/{n_batches} | "
            f"S1 {start_idx+1:,}-{end_idx:,} | "
            f"Candidates {batch_candidate_count:,} | "
            f"GT {batch_gt:,} | "
            f"True {batch_true:,} | "
            f"Recall {batch_recall:.2%} | "
            f"Elapsed {elapsed/60:.1f}m"
        )

        # Release DataFrame immediately.
        del batch_candidates
        del batch_ids_df

    # ========================================================
    # COMBINE BATCHES
    # ========================================================

    print()
    print("Combining candidate batches...")

    pattern = (
        source_temp_dir
        / "batch_*.parquet"
    )

    final_candidate_path = (
        OUT_DIR
        / f"v3_candidates_{source_name.lower()}.parquet"
    )

    # Deduplicate with DuckDB instead of pandas.
    con.execute(f"""
        COPY (

            SELECT DISTINCT
                s1_id,
                target_id

            FROM read_parquet(
                {q(pattern)}
            )

        )

        TO {q(final_candidate_path)}

        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # --------------------------------------------------------
    # Final counts
    # --------------------------------------------------------

    final_candidates = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet(
            {q(final_candidate_path)}
        )
    """).fetchone()[0]

    final_true = con.execute(f"""
        SELECT COUNT(*)

        FROM read_parquet(
            {q(final_candidate_path)}
        ) c

        INNER JOIN gt_sample g

            ON c.s1_id = g.s1_id
           AND c.target_id = g.target_id
    """).fetchone()[0]

    gt_count = con.execute("""
        SELECT COUNT(*)
        FROM gt_sample
        WHERE target_id LIKE ?
    """, [f"{source_name}-%"]).fetchone()[0]

    recall = (
        final_true / gt_count
        if gt_count > 0
        else 0
    )

    avg_candidates = (
        final_candidates / total_s1
    )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(f"FINAL V3 RESULT: S1 -> {source_name}")
    print("-" * 70)

    print(
        f"Sampled S1 entities       : {total_s1:,}"
    )

    print(
        f"Ground-truth pairs        : {gt_count:,}"
    )

    print(
        f"V3 candidate pairs        : {final_candidates:,}"
    )

    print(
        f"True pairs retrieved      : {final_true:,}"
    )

    print(
        f"V3 blocking recall        : {recall:.4%}"
    )

    print(
        f"Average candidates / S1   : {avg_candidates:.2f}"
    )

    print(
        f"Runtime                    : {elapsed/60:.2f} minutes"
    )

    print(
        f"Saved                      : {final_candidate_path}"
    )

    # --------------------------------------------------------
    # Remove temporary batch files
    # --------------------------------------------------------

    shutil.rmtree(
        source_temp_dir,
        ignore_errors=True
    )

    return {
        "source": source_name,
        "sample_fraction": S1_SAMPLE_FRACTION,
        "sample_s1": total_s1,
        "gt_pairs": gt_count,
        "candidate_pairs": final_candidates,
        "true_pairs": final_true,
        "recall": recall,
        "avg_candidates_per_s1": avg_candidates,
        "runtime_minutes": elapsed / 60,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print()
    print("=" * 70)
    print("BLOCKING V3 — 40% S1 BENCHMARK")
    print("=" * 70)

    print(f"S1 sample fraction : {S1_SAMPLE_FRACTION:.0%}")
    print(f"DuckDB memory      : {DUCKDB_MEMORY}")
    print(f"CPU threads        : {CPU_THREADS}")
    print(f"Batch size         : {BATCH_SIZE:,}")
    print(f"JW threshold       : {NAME_JW_THRESHOLD}")
    print(f"Random seed        : {RANDOM_SEED}")
    print()

    # --------------------------------------------------------
    # Create views
    # --------------------------------------------------------

    create_views()

    # --------------------------------------------------------
    # Create 40% sample
    # --------------------------------------------------------

    sample_count = create_s1_sample()

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt_sample_count = load_ground_truth()

    # --------------------------------------------------------
    # Run S2
    # --------------------------------------------------------

    result_s2 = analyze_source(
        "S2",
        "s2"
    )

    # --------------------------------------------------------
    # Run S3
    # --------------------------------------------------------

    result_s3 = analyze_source(
        "S3",
        "s3"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        result_s2,
        result_s3
    ])

    summary_path = (
        OUT_DIR / "v3_40pct_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    total_elapsed = (
        time.time() - overall_start
    )

    print()
    print("=" * 70)
    print("V3 40% BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(index=False)
    )

    print()
    print(
        f"Summary saved: {summary_path}"
    )

    print(
        f"Total runtime: {total_elapsed/60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()