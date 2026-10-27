"""
14_build_training_features.py

TRAINING FEATURE GENERATION

Important:
- Uses approximately 10% of S1 entities.
- Uses FULL S2 and FULL S3.
- Candidate generation is unchanged:
    V1 + V2 + V3
- Ground truth uses comma-separated target IDs.
- Processes S1 in batches.
- DuckDB memory limited to 4 GB.
- Does not force top-1 matching.
- S1 entities may have 0, 1, or multiple matches.

Output:
    outputs/training_features/train_s2_features.parquet
    outputs/training_features/train_s3_features.parquet
    outputs/training_features/train_features_summary.csv
"""

from pathlib import Path
import shutil
import time
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = ROOT / "features" / "normalized"
GT_FILE = ROOT / "data" / "ground_truth" / "train_ground_truth.tsv"

OUTPUT_DIR = ROOT / "outputs" / "training_features"
BATCH_DIR = OUTPUT_DIR / "_batches"
TEMP_DIR = ROOT / "outputs" / "duckdb_tmp_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

S1_FILE = NORMALIZED_DIR / "train_source1.parquet"
S2_FILE = NORMALIZED_DIR / "train_source2.parquet"
S3_FILE = NORMALIZED_DIR / "train_source3.parquet"


# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

BATCH_SIZE = 5000

DUCKDB_MEMORY = "4GB"
DUCKDB_THREADS = 4

FUZZY_THRESHOLD = 0.80


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

# Approximately 10% of S1.
#
# We deliberately use hash(entity_id) instead of DuckDB's
# USING SAMPLE syntax so this works reliably across DuckDB
# versions.
#
# This is deterministic.
S1_SAMPLE_PERCENT = 10

# 1 in every 10 hash buckets.
S1_SAMPLE_MOD = 10

# Deterministic train/validation split:
# sampled S1 IDs are divided approximately 80/20.
VALIDATION_MOD = 5


# ============================================================
# HELPERS
# ============================================================

def q(path):
    """
    Safely quote a Windows path for DuckDB SQL.
    """
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def configure(con):
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory={q(TEMP_DIR)}")
    con.execute("SET preserve_insertion_order=false")


def token_key_sql(alias):
    """
    Order-independent token representation.

    Example:

        ABC PHARMA PRIVATE LTD

    and

        LTD ABC PHARMA PRIVATE

    receive the same token key, assuming the same tokens.
    """

    return f"""
        array_to_string(
            list_sort(
                list_distinct(
                    list_filter(
                        string_split(
                            regexp_replace(
                                lower(coalesce({alias}.name_norm, '')),
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
# START
# ============================================================

start_time = time.time()

print("=" * 70)
print("STEP 14 - BUILD TRAINING FEATURES")
print("=" * 70)
print()

print("S1 sample             : approximately 10%")
print("S2                    : FULL")
print("S3                    : FULL")
print("Candidate generation  : V1 + V2 + V3")
print("RAM limit             :", DUCKDB_MEMORY)
print("Threads               :", DUCKDB_THREADS)
print("Batch size            :", BATCH_SIZE)
print("Fuzzy threshold       :", FUZZY_THRESHOLD)
print()


# ============================================================
# CONNECT DUCKDB
# ============================================================

con = duckdb.connect()

configure(con)


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

print("Loading normalized datasets...")

con.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT *
FROM read_parquet({q(S1_FILE)})
""")

con.execute(f"""
CREATE OR REPLACE VIEW s2 AS
SELECT *
FROM read_parquet({q(S2_FILE)})
""")

con.execute(f"""
CREATE OR REPLACE VIEW s3 AS
SELECT *
FROM read_parquet({q(S3_FILE)})
""")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE ground_truth AS
SELECT
    trim(source1_entity_id) AS s1_id,
    trim(matched_entity_ids) AS matched_ids
FROM read_csv(
    {q(GT_FILE)},
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id':'VARCHAR',
        'matched_entity_ids':'VARCHAR'
    }}
)
""")


# ============================================================
# EXPAND GROUND TRUTH
# ============================================================
#
# IMPORTANT:
#
# matched_entity_ids uses COMMAS:
#
# S2-123,S2-456,S3-789
#
# ============================================================

con.execute("""
CREATE OR REPLACE TEMP TABLE gt_pairs AS
SELECT
    s1_id,
    trim(target_id) AS target_id
FROM ground_truth,
UNNEST(string_split(matched_ids, ',')) AS t(target_id)
WHERE trim(target_id) <> ''
""")

gt_count = con.execute("""
SELECT COUNT(*)
FROM gt_pairs
""").fetchone()[0]

print(f"Ground-truth pairs: {gt_count:,}")

if gt_count < 7000000:
    print()
    print("WARNING:")
    print("Ground-truth pair count is lower than expected.")
    print("Expected approximately 7.64 million.")
    print("Check ground truth before continuing.")
    print()


# ============================================================
# CREATE 10% S1 SAMPLE
# ============================================================
#
# Deterministic hash sampling.
#
# abs(hash(entity_id)) % 10 = 0
#
# gives approximately 10%.
#
# This avoids DuckDB USING SAMPLE syntax completely.
# ============================================================

print("Creating approximately 10% S1 sample...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_ids AS
SELECT
    entity_id,

    CASE
        WHEN abs(hash(entity_id)) % {VALIDATION_MOD} = 0
        THEN 1
        ELSE 0
    END AS is_validation

FROM s1

WHERE abs(hash(entity_id)) % {S1_SAMPLE_MOD} = 0
""")


# ============================================================
# SAMPLE COUNTS
# ============================================================

total_s1 = con.execute("""
SELECT COUNT(*)
FROM s1_ids
""").fetchone()[0]

validation_s1 = con.execute("""
SELECT COUNT(*)
FROM s1_ids
WHERE is_validation = 1
""").fetchone()[0]

training_s1 = total_s1 - validation_s1

print()
print(f"S1 sampled entities : {total_s1:,}")
print(f"Validation S1       : {validation_s1:,}")
print(f"Training S1         : {training_s1:,}")
print()


# ============================================================
# BATCH PROCESSING
# ============================================================

batch_files_s2 = []
batch_files_s3 = []

offset = 0
batch_number = 0


while offset < total_s1:

    batch_number += 1

    batch_end = min(
        offset + BATCH_SIZE,
        total_s1
    )

    print("-" * 70)

    print(
        f"Batch {batch_number} | "
        f"S1 rows {offset:,} - {batch_end:,}"
    )


    # ========================================================
    # CURRENT S1 BATCH
    # ========================================================

    con.execute("""
    DROP TABLE IF EXISTS batch_s1
    """)

    con.execute(f"""
    CREATE TEMP TABLE batch_s1 AS

    SELECT
        s.*,
        ids.is_validation

    FROM s1 s

    INNER JOIN (
        SELECT
            entity_id,
            is_validation

        FROM s1_ids

        ORDER BY entity_id

        LIMIT {BATCH_SIZE}

        OFFSET {offset}
    ) ids

    ON s.entity_id = ids.entity_id
    """)

    batch_count = con.execute("""
    SELECT COUNT(*)
    FROM batch_s1
    """).fetchone()[0]

    if batch_count == 0:
        break


    # ========================================================
    # PROCESS S2 AND S3
    # ========================================================

    for target_name, target_view, output_list in [

        ("s2", "s2", batch_files_s2),

        ("s3", "s3", batch_files_s3),

    ]:

        print(
            f"  Generating candidates: "
            f"S1 -> {target_name}"
        )


        # ====================================================
        # CANDIDATE GENERATION
        #
        # KEEPING V1 + V2 + V3 LOGIC
        # ====================================================

        candidate_sql = f"""

        WITH

        b AS (

            SELECT
                *,
                {token_key_sql("b")} AS token_key

            FROM batch_s1 b
        ),

        t AS (

            SELECT
                *,
                {token_key_sql("t")} AS token_key

            FROM {target_view} t
        ),

        candidates AS (

            /* ==============================================
               V1.1 EXACT NORMALIZED NAME
               ============================================== */

            SELECT
                b.entity_id AS s1_id,
                t.entity_id AS target_id,
                'v1_exact_name_norm' AS blocking_rule

            FROM b

            INNER JOIN t

                ON b.name_norm <> ''
               AND b.name_norm = t.name_norm


            UNION ALL


            /* ==============================================
               V1.2 EXACT CORE NAME
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v1_exact_name_core'

            FROM b

            INNER JOIN t

                ON b.name_core <> ''
               AND b.name_core = t.name_core


            UNION ALL


            /* ==============================================
               V1.3 EXACT TRANSLITERATED NAME
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v1_exact_name_translit'

            FROM b

            INNER JOIN t

                ON b.name_translit <> ''
               AND b.name_translit = t.name_translit


            UNION ALL


            /* ==============================================
               V1.4 EXACT PHONE
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v1_exact_phone'

            FROM b

            INNER JOIN t

                ON b.phone <> ''
               AND b.phone = t.phone


            UNION ALL


            /* ==============================================
               V1.5 NAME + POSTAL
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v1_name_postal'

            FROM b

            INNER JOIN t

                ON b.name_norm <> ''
               AND b.name_norm = t.name_norm

               AND b.postal_code <> ''
               AND b.postal_code = t.postal_code


            UNION ALL


            /* ==============================================
               V1.6 CORE NAME + POSTAL
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v1_core_postal'

            FROM b

            INNER JOIN t

                ON b.name_core <> ''
               AND b.name_core = t.name_core

               AND b.postal_code <> ''
               AND b.postal_code = t.postal_code


            UNION ALL


            /* ==============================================
               V1.7 CORE NAME + HOUSE
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v1_core_house'

            FROM b

            INNER JOIN t

                ON b.name_core <> ''
               AND b.name_core = t.name_core

               AND b.house_number <> ''
               AND b.house_number = t.house_number


            UNION ALL


            /* ==============================================
               V2.1 EXACT POSTAL + FUZZY NAME
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v2_fuzzy_postal'

            FROM b

            INNER JOIN t

                ON b.postal_code <> ''
               AND b.postal_code = t.postal_code

               AND greatest(

                    jaro_winkler_similarity(
                        coalesce(b.name_core, ''),
                        coalesce(t.name_core, '')
                    ),

                    jaro_winkler_similarity(
                        coalesce(b.name_norm, ''),
                        coalesce(t.name_norm, '')
                    ),

                    jaro_winkler_similarity(
                        coalesce(b.name_translit, ''),
                        coalesce(t.name_translit, '')
                    )

               ) >= {FUZZY_THRESHOLD}


            UNION ALL


            /* ==============================================
               V2.2 EXACT HOUSE + FUZZY NAME
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v2_fuzzy_house'

            FROM b

            INNER JOIN t

                ON b.house_number <> ''
               AND b.house_number = t.house_number

               AND greatest(

                    jaro_winkler_similarity(
                        coalesce(b.name_core, ''),
                        coalesce(t.name_core, '')
                    ),

                    jaro_winkler_similarity(
                        coalesce(b.name_norm, ''),
                        coalesce(t.name_norm, '')
                    ),

                    jaro_winkler_similarity(
                        coalesce(b.name_translit, ''),
                        coalesce(t.name_translit, '')
                    )

               ) >= {FUZZY_THRESHOLD}


            UNION ALL


            /* ==============================================
               V3.1 TOKEN-ORDER-INDEPENDENT NAME
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v3_token_name'

            FROM b

            INNER JOIN t

                ON b.token_key <> ''
               AND b.token_key = t.token_key


            UNION ALL


            /* ==============================================
               V3.2 TOKEN KEY + POSTAL
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v3_token_postal'

            FROM b

            INNER JOIN t

                ON b.token_key <> ''
               AND b.token_key = t.token_key

               AND b.postal_code <> ''
               AND b.postal_code = t.postal_code


            UNION ALL


            /* ==============================================
               V3.3 TOKEN KEY + HOUSE
               ============================================== */

            SELECT
                b.entity_id,
                t.entity_id,
                'v3_token_house'

            FROM b

            INNER JOIN t

                ON b.token_key <> ''
               AND b.token_key = t.token_key

               AND b.house_number <> ''
               AND b.house_number = t.house_number
        ),


        /* ==================================================
           CONVERT BLOCKING RULES INTO FEATURES
           ================================================== */

        candidate_flags AS (

            SELECT

                s1_id,

                target_id,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_exact_name_norm'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_exact_name_norm,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_exact_name_core'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_exact_name_core,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_exact_name_translit'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_exact_name_translit,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_exact_phone'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_exact_phone,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_name_postal'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_name_postal,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_core_postal'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_core_postal,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v1_core_house'
                        THEN 1
                        ELSE 0
                    END
                ) AS v1_core_house,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v2_fuzzy_postal'
                        THEN 1
                        ELSE 0
                    END
                ) AS v2_fuzzy_postal,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v2_fuzzy_house'
                        THEN 1
                        ELSE 0
                    END
                ) AS v2_fuzzy_house,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v3_token_name'
                        THEN 1
                        ELSE 0
                    END
                ) AS v3_token_name,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v3_token_postal'
                        THEN 1
                        ELSE 0
                    END
                ) AS v3_token_postal,


                MAX(
                    CASE
                        WHEN blocking_rule =
                             'v3_token_house'
                        THEN 1
                        ELSE 0
                    END
                ) AS v3_token_house

            FROM candidates

            GROUP BY
                s1_id,
                target_id
        )


        SELECT *

        FROM candidate_flags
        """


        # ====================================================
        # MATERIALIZE CANDIDATES
        # ====================================================

        con.execute("""
        DROP TABLE IF EXISTS candidate_flags
        """)

        con.execute(
            f"""
            CREATE TEMP TABLE candidate_flags AS

            {candidate_sql}
            """
        )


        candidate_count = con.execute("""
        SELECT COUNT(*)
        FROM candidate_flags
        """).fetchone()[0]


        print(
            f"    Candidates: "
            f"{candidate_count:,}"
        )


        if candidate_count == 0:

            print(
                "    No candidates in this batch."
            )

            continue


        # ====================================================
        # PAIR FEATURES
        # ====================================================

        feature_sql = f"""

        WITH pairs AS (

            SELECT

                c.*,


                /* ------------------------------------------
                   S1 FIELDS
                   ------------------------------------------ */

                s1.name_norm AS s1_name_norm,

                s1.name_core AS s1_name_core,

                s1.name_translit AS s1_name_translit,

                s1.address_norm AS s1_address_norm,

                s1.postal_code AS s1_postal_code,

                s1.house_number AS s1_house_number,

                s1.phone AS s1_phone,

                s1.domain AS s1_domain,

                s1.country_norm AS s1_country_norm,

                s1.name_script AS s1_name_script,

                s1.address_script AS s1_address_script,

                s1.script_type AS s1_script_type,

                s1.is_validation,


                /* ------------------------------------------
                   TARGET FIELDS
                   ------------------------------------------ */

                t.name_norm AS target_name_norm,

                t.name_core AS target_name_core,

                t.name_translit AS target_name_translit,

                t.address_norm AS target_address_norm,

                t.postal_code AS target_postal_code,

                t.house_number AS target_house_number,

                t.phone AS target_phone,

                t.domain AS target_domain,

                t.country_norm AS target_country_norm,

                t.name_script AS target_name_script,

                t.address_script AS target_address_script,

                t.script_type AS target_script_type


            FROM candidate_flags c


            INNER JOIN batch_s1 s1

                ON c.s1_id = s1.entity_id


            INNER JOIN {target_view} t

                ON c.target_id = t.entity_id
        ),


        scored AS (

            SELECT

                *,


                /* ==========================================
                   NAME JARO-WINKLER
                   ========================================== */

                jaro_winkler_similarity(

                    coalesce(
                        s1_name_norm,
                        ''
                    ),

                    coalesce(
                        target_name_norm,
                        ''
                    )

                ) AS name_jw_norm,


                jaro_winkler_similarity(

                    coalesce(
                        s1_name_core,
                        ''
                    ),

                    coalesce(
                        target_name_core,
                        ''
                    )

                ) AS name_jw_core,


                jaro_winkler_similarity(

                    coalesce(
                        s1_name_translit,
                        ''
                    ),

                    coalesce(
                        target_name_translit,
                        ''
                    )

                ) AS name_jw_translit,


                greatest(

                    jaro_winkler_similarity(
                        coalesce(
                            s1_name_norm,
                            ''
                        ),
                        coalesce(
                            target_name_norm,
                            ''
                        )
                    ),

                    jaro_winkler_similarity(
                        coalesce(
                            s1_name_core,
                            ''
                        ),
                        coalesce(
                            target_name_core,
                            ''
                        )
                    ),

                    jaro_winkler_similarity(
                        coalesce(
                            s1_name_translit,
                            ''
                        ),
                        coalesce(
                            target_name_translit,
                            ''
                        )
                    )

                ) AS name_jw_max,


                /* ==========================================
                   ADDRESS SIMILARITY
                   ========================================== */

                jaro_winkler_similarity(

                    coalesce(
                        s1_address_norm,
                        ''
                    ),

                    coalesce(
                        target_address_norm,
                        ''
                    )

                ) AS address_jw,


                /* ==========================================
                   EXACT STRUCTURED FEATURES
                   ========================================== */

                CASE

                    WHEN s1_postal_code <> ''
                     AND s1_postal_code =
                         target_postal_code

                    THEN 1

                    ELSE 0

                END AS postal_exact,


                CASE

                    WHEN s1_house_number <> ''
                     AND s1_house_number =
                         target_house_number

                    THEN 1

                    ELSE 0

                END AS house_exact,


                CASE

                    WHEN s1_phone <> ''
                     AND s1_phone =
                         target_phone

                    THEN 1

                    ELSE 0

                END AS phone_exact,


                CASE

                    WHEN s1_domain <> ''
                     AND s1_domain =
                         target_domain

                    THEN 1

                    ELSE 0

                END AS domain_exact,


                CASE

                    WHEN s1_country_norm <> ''
                     AND s1_country_norm =
                         target_country_norm

                    THEN 1

                    ELSE 0

                END AS country_exact,


                CASE

                    WHEN s1_name_script <> ''
                     AND s1_name_script =
                         target_name_script

                    THEN 1

                    ELSE 0

                END AS name_script_exact,


                CASE

                    WHEN s1_address_script <> ''
                     AND s1_address_script =
                         target_address_script

                    THEN 1

                    ELSE 0

                END AS address_script_exact,


                CASE

                    WHEN s1_script_type <> ''
                     AND s1_script_type =
                         target_script_type

                    THEN 1

                    ELSE 0

                END AS script_type_exact,


                /* ==========================================
                   LENGTH FEATURES
                   ========================================== */

                abs(

                    length(
                        coalesce(
                            s1_name_norm,
                            ''
                        )
                    )

                    -

                    length(
                        coalesce(
                            target_name_norm,
                            ''
                        )
                    )

                ) AS name_length_diff,


                abs(

                    length(
                        coalesce(
                            s1_address_norm,
                            ''
                        )
                    )

                    -

                    length(
                        coalesce(
                            target_address_norm,
                            ''
                        )
                    )

                ) AS address_length_diff,


                /* ==========================================
                   GROUND TRUTH LABEL
                   ========================================== */

                CASE

                    WHEN gt.target_id IS NOT NULL

                    THEN 1

                    ELSE 0

                END AS label


            FROM pairs


            LEFT JOIN gt_pairs gt

                ON pairs.s1_id =
                   gt.s1_id

               AND pairs.target_id =
                   gt.target_id
        )


        SELECT

            s1_id,

            target_id,

            '{target_name}' AS target_source,


            /* NAME */

            name_jw_norm,

            name_jw_core,

            name_jw_translit,

            name_jw_max,


            /* ADDRESS */

            address_jw,


            /* STRUCTURED */

            postal_exact,

            house_exact,

            phone_exact,

            domain_exact,

            country_exact,


            /* SCRIPT */

            name_script_exact,

            address_script_exact,

            script_type_exact,


            /* LENGTH */

            name_length_diff,

            address_length_diff,


            /* V1 */

            v1_exact_name_norm,

            v1_exact_name_core,

            v1_exact_name_translit,

            v1_exact_phone,

            v1_name_postal,

            v1_core_postal,

            v1_core_house,


            /* V2 */

            v2_fuzzy_postal,

            v2_fuzzy_house,


            /* V3 */

            v3_token_name,

            v3_token_postal,

            v3_token_house,


            /* SPLIT */

            is_validation,


            /* LABEL */

            label


        FROM scored
        """


        feature_table = (
            f"features_{target_name}"
        )


        con.execute(
            f"""
            DROP TABLE IF EXISTS
            {feature_table}
            """
        )


        con.execute(
            f"""
            CREATE TEMP TABLE
            {feature_table} AS

            {feature_sql}
            """
        )


        # ====================================================
        # COUNTS
        # ====================================================

        feature_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {feature_table}
            """
        ).fetchone()[0]


        positive_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {feature_table}
            WHERE label = 1
            """
        ).fetchone()[0]


        print(
            f"    Features: "
            f"{feature_count:,} | "
            f"positives: "
            f"{positive_count:,}"
        )


        # ====================================================
        # SAVE BATCH
        # ====================================================

        batch_file = (
            BATCH_DIR
            /
            f"{target_name}_batch_"
            f"{batch_number:05d}.parquet"
        )


        con.execute(
            f"""
            COPY {feature_table}

            TO {q(batch_file)}

            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )


        output_list.append(
            batch_file
        )


        # ====================================================
        # CLEAN TEMP TABLES
        # ====================================================

        con.execute(
            f"""
            DROP TABLE IF EXISTS
            {feature_table}
            """
        )

        con.execute("""
        DROP TABLE IF EXISTS candidate_flags
        """)


    # ========================================================
    # CLEAN CURRENT BATCH
    # ========================================================

    con.execute("""
    DROP TABLE IF EXISTS batch_s1
    """)


    offset += BATCH_SIZE


    elapsed = (
        time.time()
        -
        start_time
    )


    print(
        f"  Elapsed: "
        f"{elapsed / 60:.1f} minutes"
    )


# ============================================================
# COMBINE BATCH PARQUETS
# ============================================================

print()
print("=" * 70)
print("COMBINING FEATURE FILES")
print("=" * 70)


def combine_files(
    target_name,
    output_name
):

    pattern = (
        BATCH_DIR
        /
        f"{target_name}_batch_*.parquet"
    )


    output_path = (
        OUTPUT_DIR
        /
        output_name
    )


    print()
    print(
        f"Combining {target_name}..."
    )


    con.execute(
        f"""
        COPY (

            SELECT *

            FROM read_parquet(
                {q(pattern)}
            )

        )

        TO {q(output_path)}

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )


    total_rows = con.execute(
        f"""
        SELECT COUNT(*)

        FROM read_parquet(
            {q(output_path)}
        )
        """
    ).fetchone()[0]


    positive_rows = con.execute(
        f"""
        SELECT COUNT(*)

        FROM read_parquet(
            {q(output_path)}
        )

        WHERE label = 1
        """
    ).fetchone()[0]


    validation_rows = con.execute(
        f"""
        SELECT COUNT(*)

        FROM read_parquet(
            {q(output_path)}
        )

        WHERE is_validation = 1
        """
    ).fetchone()[0]


    validation_positive = con.execute(
        f"""
        SELECT COUNT(*)

        FROM read_parquet(
            {q(output_path)}
        )

        WHERE is_validation = 1
          AND label = 1
        """
    ).fetchone()[0]


    print(
        f"  Rows       : "
        f"{total_rows:,}"
    )

    print(
        f"  Positives  : "
        f"{positive_rows:,}"
    )

    print(
        f"  Validation : "
        f"{validation_rows:,}"
    )

    print(
        f"  Val +      : "
        f"{validation_positive:,}"
    )


    return output_path


# ============================================================
# COMBINE S2
# ============================================================

s2_output = combine_files(
    "s2",
    "train_s2_features.parquet"
)


# ============================================================
# COMBINE S3
# ============================================================

s3_output = combine_files(
    "s3",
    "train_s3_features.parquet"
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_path = (
    OUTPUT_DIR
    /
    "train_features_summary.csv"
)


con.execute(
    f"""
    COPY (

        SELECT

            'S2' AS target_source,

            COUNT(*) AS total_pairs,

            SUM(label)
                AS positive_pairs,

            COUNT(*) - SUM(label)
                AS negative_pairs,

            SUM(
                CASE
                    WHEN is_validation = 1
                    THEN 1
                    ELSE 0
                END
            ) AS validation_pairs,

            SUM(
                CASE
                    WHEN is_validation = 1
                     AND label = 1
                    THEN 1
                    ELSE 0
                END
            ) AS validation_positive_pairs

        FROM read_parquet(
            {q(s2_output)}
        )


        UNION ALL


        SELECT

            'S3' AS target_source,

            COUNT(*) AS total_pairs,

            SUM(label)
                AS positive_pairs,

            COUNT(*) - SUM(label)
                AS negative_pairs,

            SUM(
                CASE
                    WHEN is_validation = 1
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN is_validation = 1
                     AND label = 1
                    THEN 1
                    ELSE 0
                END
            )

        FROM read_parquet(
            {q(s3_output)}
        )

    )

    TO {q(summary_path)}

    (
        HEADER,
        DELIMITER ','
    )
    """
)


# ============================================================
# CLEAN TEMPORARY FILES
# ============================================================

print()
print("Cleaning temporary batch files...")


try:

    shutil.rmtree(
        BATCH_DIR
    )

except Exception as e:

    print(
        "Warning: could not remove "
        "batch directory:",
        e
    )


try:

    shutil.rmtree(
        TEMP_DIR
    )

except Exception as e:

    print(
        "Warning: could not remove "
        "DuckDB temp directory:",
        e
    )


# ============================================================
# FINAL
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


print()
print("=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print(
    f"Runtime: "
    f"{elapsed / 60:.2f} minutes"
)

print()

print(
    "Created:"
)

print(
    f"  {s2_output}"
)

print(
    f"  {s3_output}"
)

print(
    f"  {summary_path}"
)

print()
print(
    "Next step:"
)
print(
    "  15_train_lightgbm.py"
)
print()