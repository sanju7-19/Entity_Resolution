from pathlib import Path
import duckdb
import pandas as pd
import time
import shutil

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

RESULT_FILE = OUT_DIR / "combined_blocking_benchmark.csv"

# Memory-safe settings
BATCH_SIZE = 5_000
DUCKDB_MEMORY = "4GB"
DUCKDB_THREADS = 2
MAX_TEMP_SIZE = "30GB"

# V2 thresholds
THRESHOLDS = [0.80, 0.85, 0.90]

# ============================================================
# IMPORTANT
# ============================================================
#
# We are benchmarking:
#
# V1:
#   exact_name_norm
#   exact_name_core
#   exact_name_translit
#   exact_phone
#   name_norm_postal
#   name_core_postal
#   name_core_house
#   name_translit_postal
#   address_postal
#
# V2:
#   postal + fuzzy name
#   house + fuzzy name
#
# We deliberately EXCLUDE unrestricted exact_domain here.
#
# Earlier benchmark showed that exact_domain generated
# ~300 MILLION candidates on the 10% sample because common
# domains create enormous blocks.
#
# We will handle domain separately with frequency-aware blocking
# after this benchmark.
#
# ============================================================


# ============================================================
# HELPERS
# ============================================================

def configure(con):
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
    con.execute(f"SET max_temp_directory_size='{MAX_TEMP_SIZE}'")


def q(path):
    return path.as_posix().replace("'", "''")


def safe_name_score_sql(left_alias="s", right_alias="t"):
    """
    Maximum of three Jaro-Winkler similarities.

    NULLs are converted to 0 so GREATEST behaves predictably.
    """

    return f"""
        GREATEST(
            COALESCE(
                jaro_winkler_similarity(
                    {left_alias}.name_core,
                    {right_alias}.name_core
                ),
                0.0
            ),
            COALESCE(
                jaro_winkler_similarity(
                    {left_alias}.name_norm,
                    {right_alias}.name_norm
                ),
                0.0
            ),
            COALESCE(
                jaro_winkler_similarity(
                    {left_alias}.name_translit,
                    {right_alias}.name_translit
                ),
                0.0
            )
        )
    """


# ============================================================
# CREATE DUCKDB CONNECTION
# ============================================================

print("=" * 70)
print("COMBINED V1 + V2 BLOCKING BENCHMARK")
print("=" * 70)

print()
print("S1 sample :", S1_SAMPLE)
print("GT sample :", GT_SAMPLE)
print("S2        :", S2_PARQUET)
print("S3        :", S3_PARQUET)
print("Batch size:", BATCH_SIZE)
print("Memory    :", DUCKDB_MEMORY)
print("Threads   :", DUCKDB_THREADS)
print("Thresholds:", THRESHOLDS)
print()

start_time = time.time()

con = duckdb.connect()
configure(con)

# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print("Loading S1 sample...")

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

s1_count = con.execute("""
    SELECT COUNT(*)
    FROM s1
""").fetchone()[0]

print(f"S1 sample size: {s1_count:,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW gt_raw AS
    SELECT
        CAST(source1_entity_id AS VARCHAR) AS s1_id,
        CAST(target_entity_id AS VARCHAR) AS target_id
    FROM read_parquet('{q(GT_SAMPLE)}')
""")

gt_total = con.execute("""
    SELECT COUNT(*)
    FROM gt_raw
""").fetchone()[0]

print(f"Ground-truth links: {gt_total:,}")


# ============================================================
# LOAD TARGET SOURCES
# ============================================================

print("Loading target views...")

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


# ============================================================
# CLASSIFY GT INTO S2 / S3
# ============================================================

print("Classifying ground-truth targets...")

con.execute("""
    CREATE OR REPLACE TEMP VIEW gt_s2 AS
    SELECT
        g.s1_id,
        g.target_id
    FROM gt_raw g
    INNER JOIN s2 t
        ON g.target_id = t.entity_id
""")

con.execute("""
    CREATE OR REPLACE TEMP VIEW gt_s3 AS
    SELECT
        g.s1_id,
        g.target_id
    FROM gt_raw g
    INNER JOIN s3 t
        ON g.target_id = t.entity_id
""")

gt_s2_count = con.execute("""
    SELECT COUNT(*)
    FROM gt_s2
""").fetchone()[0]

gt_s3_count = con.execute("""
    SELECT COUNT(*)
    FROM gt_s3
""").fetchone()[0]

print()
print("Ground truth:")
print(f"  S2 true pairs : {gt_s2_count:,}")
print(f"  S3 true pairs : {gt_s3_count:,}")
print(f"  Total         : {gt_s2_count + gt_s3_count:,}")


# ============================================================
# CHECK SAMPLE
# ============================================================

print()
print("Checking similarity functions...")

jw = con.execute("""
    SELECT jaro_winkler_similarity(
        'mitchell and gee',
        'mitche1l and gee'
    )
""").fetchone()[0]

lev = con.execute("""
    SELECT levenshtein(
        'mitchell and gee',
        'mitche1l and gee'
    )
""").fetchone()[0]

print(f"  Jaro-Winkler : {jw:.4f}")
print(f"  Levenshtein  : {lev}")


# ============================================================
# GET BATCH IDS
# ============================================================

s1_ids = con.execute("""
    SELECT entity_id
    FROM s1
    ORDER BY entity_id
""").fetchdf()["entity_id"].tolist()

total_batches = (len(s1_ids) + BATCH_SIZE - 1) // BATCH_SIZE

print()
print(f"Total batches: {total_batches}")


# ============================================================
# V1 ROUTES
# ============================================================

V1_ROUTES = [
    "exact_name_norm",
    "exact_name_core",
    "exact_name_translit",
    "exact_phone",
    "name_norm_postal",
    "name_core_postal",
    "name_core_house",
    "name_translit_postal",
    "address_postal",
]


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(source_name, target_view, gt_view, gt_total):

    print()
    print("=" * 70)
    print(f"S1 -> {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Per threshold accumulators
    # --------------------------------------------------------

    stats = {}

    for threshold in THRESHOLDS:

        stats[threshold] = {
            "v1_candidate_count": 0,
            "v2_postal_candidate_count": 0,
            "v2_house_candidate_count": 0,

            "combined_candidate_count": 0,

            "v1_true_pairs": set(),
            "v2_true_pairs": set(),
            "combined_true_pairs": set(),
        }

    # --------------------------------------------------------
    # Batch loop
    # --------------------------------------------------------

    for batch_number in range(total_batches):

        start_idx = batch_number * BATCH_SIZE
        end_idx = min(
            start_idx + BATCH_SIZE,
            len(s1_ids)
        )

        batch_id_values = s1_ids[start_idx:end_idx]

        print(
            f"\nBatch {batch_number + 1}/{total_batches} "
            f"S1 {start_idx + 1:,}-{end_idx:,}"
        )

        # ----------------------------------------------------
        # Create batch ID table
        # ----------------------------------------------------

        con.execute("""
            DROP TABLE IF EXISTS batch_ids
        """)

        con.execute("""
            CREATE TEMP TABLE batch_ids (
                entity_id VARCHAR
            )
        """)

        # Insert IDs safely in chunks
        id_df = pd.DataFrame({
            "entity_id": batch_id_values
        })

        con.register("batch_id_df", id_df)

        con.execute("""
            INSERT INTO batch_ids
            SELECT entity_id
            FROM batch_id_df
        """)

        con.unregister("batch_id_df")

        # ----------------------------------------------------
        # Create batch S1
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
        # V1 CANDIDATES
        # ====================================================

        # We build one deduplicated V1 candidate table.
        #
        # This avoids counting the same candidate multiple
        # times when it is retrieved by multiple exact rules.
        #
        # IMPORTANT:
        # exact_domain is intentionally excluded here because
        # unrestricted domain blocks were extremely large.
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

        v1_count = con.execute("""
            SELECT COUNT(*)
            FROM v1_candidates
        """).fetchone()[0]

        # ----------------------------------------------------
        # V1 true pairs
        # ----------------------------------------------------

        gt_pairs_v1 = con.execute(f"""
            SELECT
                g.s1_id,
                g.target_id
            FROM {gt_view} g
            INNER JOIN v1_candidates c
                ON g.s1_id = c.s1_id
               AND g.target_id = c.target_id
        """).fetchall()

        v1_true_pairs = set(gt_pairs_v1)

        # ====================================================
        # V2 ROUTES
        # ====================================================

        name_score = safe_name_score_sql("s", "t")

        # Build V2 candidates once at each threshold.
        # Then union with V1.
        # ====================================================

        for threshold in THRESHOLDS:

            # ------------------------------------------------
            # V2 candidate table
            # ------------------------------------------------

            con.execute("""
                DROP TABLE IF EXISTS v2_candidates
            """)

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

                   AND (
                        {name_score}
                   ) >= {threshold}

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

                   AND (
                        {name_score}
                   ) >= {threshold}
            """

            con.execute(v2_sql)

            # ------------------------------------------------
            # V2 count
            # ------------------------------------------------

            v2_count = con.execute("""
                SELECT COUNT(*)
                FROM v2_candidates
            """).fetchone()[0]

            # ------------------------------------------------
            # V2 true pairs
            # ------------------------------------------------

            v2_true = con.execute(f"""
                SELECT
                    g.s1_id,
                    g.target_id
                FROM {gt_view} g
                INNER JOIN v2_candidates c
                    ON g.s1_id = c.s1_id
                   AND g.target_id = c.target_id
            """).fetchall()

            v2_true_pairs = set(v2_true)

            # ------------------------------------------------
            # Combined candidates
            # ------------------------------------------------

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

            combined_count = con.execute("""
                SELECT COUNT(*)
                FROM combined_candidates
            """).fetchone()[0]

            # ------------------------------------------------
            # Combined true pairs
            # ------------------------------------------------

            combined_true = con.execute(f"""
                SELECT
                    g.s1_id,
                    g.target_id
                FROM {gt_view} g
                INNER JOIN combined_candidates c
                    ON g.s1_id = c.s1_id
                   AND g.target_id = c.target_id
            """).fetchall()

            combined_true_pairs = set(combined_true)

            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            stats[threshold]["v1_candidate_count"] += v1_count
            stats[threshold]["combined_candidate_count"] += combined_count

            stats[threshold]["v2_postal_candidate_count"] += con.execute("""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        s.entity_id AS s1_id,
                        t.entity_id AS target_id
                    FROM batch_s1 s
                    INNER JOIN {target} t
                        ON s.postal_code IS NOT NULL
                       AND s.postal_code <> ''
                       AND t.postal_code IS NOT NULL
                       AND t.postal_code <> ''
                       AND s.postal_code = t.postal_code
                       AND ({score}) >= {threshold}
                )
            """.format(
                target=target_view,
                score=safe_name_score_sql("s", "t"),
                threshold=threshold
            )).fetchone()[0]

            stats[threshold]["v2_house_candidate_count"] += con.execute("""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        s.entity_id AS s1_id,
                        t.entity_id AS target_id
                    FROM batch_s1 s
                    INNER JOIN {target} t
                        ON s.house_number IS NOT NULL
                       AND s.house_number <> ''
                       AND t.house_number IS NOT NULL
                       AND t.house_number <> ''
                       AND s.house_number = t.house_number
                       AND ({score}) >= {threshold}
                )
            """.format(
                target=target_view,
                score=safe_name_score_sql("s", "t"),
                threshold=threshold
            )).fetchone()[0]

            stats[threshold]["v1_true_pairs"].update(v1_true_pairs)
            stats[threshold]["v2_true_pairs"].update(v2_true_pairs)
            stats[threshold]["combined_true_pairs"].update(
                combined_true_pairs
            )

        print("  completed")

        # ----------------------------------------------------
        # Clean batch tables
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS batch_s1")
        con.execute("DROP TABLE IF EXISTS v1_candidates")
        con.execute("DROP TABLE IF EXISTS v2_candidates")
        con.execute("DROP TABLE IF EXISTS combined_candidates")
        con.execute("DROP TABLE IF EXISTS batch_ids")

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    rows = []

    print()
    print("-" * 70)
    print(f"RESULTS: S1 -> {source_name}")
    print("-" * 70)

    for threshold in THRESHOLDS:

        st = stats[threshold]

        v1_true = len(st["v1_true_pairs"])
        v2_true = len(st["v2_true_pairs"])
        combined_true = len(st["combined_true_pairs"])

        v1_recall = (
            v1_true / gt_total
            if gt_total else 0
        )

        v2_recall = (
            v2_true / gt_total
            if gt_total else 0
        )

        combined_recall = (
            combined_true / gt_total
            if gt_total else 0
        )

        # Important:
        # This measures how many NEW true pairs V2 adds
        # beyond V1.
        incremental_true = combined_true - v1_true

        incremental_recall = (
            incremental_true / gt_total
            if gt_total else 0
        )

        # V2 candidate count is estimated as postal + house
        # route work. Combined candidate count is actually
        # deduplicated against V1.
        avg_combined = (
            st["combined_candidate_count"] / len(s1_ids)
        )

        print()
        print(f"Threshold {threshold:.2f}")

        print(
            f"  V1 candidates          : "
            f"{st['v1_candidate_count']:,}"
        )

        print(
            f"  V2 postal candidates  : "
            f"{st['v2_postal_candidate_count']:,}"
        )

        print(
            f"  V2 house candidates   : "
            f"{st['v2_house_candidate_count']:,}"
        )

        print(
            f"  Combined candidates   : "
            f"{st['combined_candidate_count']:,}"
        )

        print(
            f"  Avg combined/S1       : "
            f"{avg_combined:.2f}"
        )

        print(
            f"  V1 true pairs         : "
            f"{v1_true:,}"
        )

        print(
            f"  V1 recall             : "
            f"{v1_recall:.4%}"
        )

        print(
            f"  V2-only true pairs    : "
            f"{v2_true:,}"
        )

        print(
            f"  V2-only recall        : "
            f"{v2_recall:.4%}"
        )

        print(
            f"  NEW true pairs from V2: "
            f"{incremental_true:,}"
        )

        print(
            f"  Incremental recall    : "
            f"{incremental_recall:.4%}"
        )

        print(
            f"  COMBINED true pairs   : "
            f"{combined_true:,}"
        )

        print(
            f"  COMBINED recall       : "
            f"{combined_recall:.4%}"
        )

        rows.append({
            "target_source": source_name,
            "threshold": threshold,

            "v1_candidate_count": st["v1_candidate_count"],
            "v2_postal_candidates": st[
                "v2_postal_candidate_count"
            ],
            "v2_house_candidates": st[
                "v2_house_candidate_count"
            ],
            "combined_candidate_count": st[
                "combined_candidate_count"
            ],

            "avg_combined_candidates_per_s1": avg_combined,

            "v1_true_pairs": v1_true,
            "v1_recall": v1_recall,

            "v2_true_pairs": v2_true,
            "v2_recall": v2_recall,

            "v2_incremental_true_pairs": incremental_true,
            "v2_incremental_recall": incremental_recall,

            "combined_true_pairs": combined_true,
            "combined_recall": combined_recall,

            "ground_truth_pairs": gt_total,
        })

    return rows


# ============================================================
# RUN S2
# ============================================================

all_results = []

s2_results = process_source(
    source_name="S2",
    target_view="s2",
    gt_view="gt_s2",
    gt_total=gt_s2_count,
)

all_results.extend(s2_results)


# ============================================================
# RUN S3
# ============================================================

s3_results = process_source(
    source_name="S3",
    target_view="s3",
    gt_view="gt_s3",
    gt_total=gt_s3_count,
)

all_results.extend(s3_results)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print()
print("=" * 70)
print("COMBINED BLOCKING BENCHMARK COMPLETE")
print("=" * 70)

print()
print(results_df.to_string(index=False))

print()
print("Results saved to:")
print(RESULT_FILE)

elapsed = time.time() - start_time

print()
print(f"Elapsed time: {elapsed / 60:.2f} minutes")

print()
print("IMPORTANT:")
print("This benchmark measures V1 + V2 together.")
print("Do NOT create the final candidate dataset yet.")
print("Use COMBINED recall and candidate volume to choose")
print("the next blocking configuration.")
print()