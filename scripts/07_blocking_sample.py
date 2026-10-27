from pathlib import Path
import re
import duckdb
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NORMALIZED = ROOT / "features" / "normalized"
GROUND_TRUTH = ROOT / "data" / "ground_truth" / "train_ground_truth.tsv"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

S1_FILE = NORMALIZED / "train_source1.parquet"
S2_FILE = NORMALIZED / "train_source2.parquet"
S3_FILE = NORMALIZED / "train_source3.parquet"

GT_PAIRS_FILE = OUTPUT / "ground_truth_pairs.parquet"

RESULT_FILE = OUTPUT / "blocking_sample_benchmark.csv"

SAMPLE_S1_FILE = OUTPUT / "blocking_sample_s1.parquet"
SAMPLE_GT_FILE = OUTPUT / "blocking_sample_gt.parquet"


# ============================================================
# RESOURCE LIMITS
# ============================================================

SAMPLE_FRACTION = 0.10

RANDOM_SEED = 42

DUCKDB_THREADS = 2

DUCKDB_MEMORY = "6GB"


# ============================================================
# BLOCKING RULES
# ============================================================

RULES = {

    "exact_name_norm": {
        "columns": ["name_norm"],
        "description": "Exact normalized business name",
    },

    "exact_name_core": {
        "columns": ["name_core"],
        "description": "Exact core business name",
    },

    "exact_name_translit": {
        "columns": ["name_translit"],
        "description": "Exact transliterated business name",
    },

    "exact_domain": {
        "columns": ["domain"],
        "description": "Exact domain",
    },

    "exact_phone": {
        "columns": ["phone"],
        "description": "Exact phone",
    },

    "name_norm_postal": {
        "columns": ["name_norm", "postal_code"],
        "description": "Normalized name + postal code",
    },

    "name_core_postal": {
        "columns": ["name_core", "postal_code"],
        "description": "Core name + postal code",
    },

    "name_core_house": {
        "columns": ["name_core", "house_number"],
        "description": "Core name + house number",
    },

    "name_translit_postal": {
        "columns": ["name_translit", "postal_code"],
        "description": "Transliterated name + postal code",
    },

    "address_postal": {
        "columns": ["address_norm", "postal_code"],
        "description": "Normalized address + postal code",
    },
}


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def sql_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def parse_match_ids(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    value = value.strip("[](){}")

    value = value.replace('"', "")
    value = value.replace("'", "")

    parts = re.split(
        r"[,;|\s]+",
        value
    )

    return [
        x.strip()
        for x in parts
        if x.strip()
        and x.lower()
        not in {"nan", "none", "null"}
    ]


# ============================================================
# DUCKDB
# ============================================================

def create_connection():

    con = duckdb.connect()

    con.execute(
        f"SET threads={DUCKDB_THREADS}"
    )

    con.execute(
        f"SET memory_limit='{DUCKDB_MEMORY}'"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    con.execute(
        "SET enable_progress_bar=false"
    )

    return con


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

def prepare_ground_truth():

    if GT_PAIRS_FILE.exists():

        print()
        print("Ground-truth pairs already exist.")

        return

    print()
    print("=" * 70)
    print("CREATING GROUND-TRUTH PAIRS")
    print("=" * 70)

    reader = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        keep_default_na=False,
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None

    total = 0

    for chunk_no, chunk in enumerate(
        reader,
        start=1
    ):

        rows = []

        for _, row in chunk.iterrows():

            s1_id = str(
                row["source1_entity_id"]
            ).strip()

            matches = parse_match_ids(
                row["matched_entity_ids"]
            )

            for target_id in matches:

                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "target_entity_id": target_id,
                    }
                )

        if rows:

            df = pd.DataFrame(rows)

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    GT_PAIRS_FILE,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

            total += len(df)

        print(
            f"chunk {chunk_no:>3}: "
            f"{len(rows):,} links"
        )

    if writer:
        writer.close()

    print()
    print(
        f"Ground-truth links: {total:,}"
    )


# ============================================================
# CREATE 10% S1 SAMPLE
# ============================================================

def create_s1_sample(con):

    print()
    print("=" * 70)
    print("STEP 1: CREATING 10% S1 SAMPLE")
    print("=" * 70)

    if SAMPLE_S1_FILE.exists():

        print(
            "Sample already exists:"
        )

        print(
            f"  {SAMPLE_S1_FILE}"
        )

        return

    # --------------------------------------------------------
    # SAMPLE ONLY S1.
    #
    # We deliberately keep the complete S1 records selected.
    # --------------------------------------------------------

    con.execute(
        f"""
        COPY (

            SELECT *

            FROM read_parquet(
                '{sql_path(S1_FILE)}'
            )

            USING SAMPLE
                {SAMPLE_FRACTION * 100}%
            (
                reservoir,
                {RANDOM_SEED}
            )

        )
        TO '{sql_path(SAMPLE_S1_FILE)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)

        FROM read_parquet(
            '{sql_path(SAMPLE_S1_FILE)}'
        );
        """
    ).fetchone()[0]

    print()
    print(
        f"Sampled S1 rows: {count:,}"
    )

    print(
        f"Saved: {SAMPLE_S1_FILE}"
    )


# ============================================================
# CREATE SAMPLED GROUND TRUTH
# ============================================================

def create_sample_ground_truth(con):

    print()
    print("=" * 70)
    print("STEP 2: CREATING SAMPLED GROUND TRUTH")
    print("=" * 70)

    if SAMPLE_GT_FILE.exists():

        print(
            "Sample ground truth already exists."
        )

        return

    con.execute(
        f"""
        COPY (

            SELECT
                g.source1_entity_id,
                g.target_entity_id

            FROM read_parquet(
                '{sql_path(GT_PAIRS_FILE)}'
            ) g

            INNER JOIN read_parquet(
                '{sql_path(SAMPLE_S1_FILE)}'
            ) s

                ON CAST(
                    g.source1_entity_id
                    AS VARCHAR
                )
                =
                CAST(
                    s.entity_id
                    AS VARCHAR
                )

        )
        TO '{sql_path(SAMPLE_GT_FILE)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)

        FROM read_parquet(
            '{sql_path(SAMPLE_GT_FILE)}'
        );
        """
    ).fetchone()[0]

    print()
    print(
        f"Sample ground-truth links: "
        f"{count:,}"
    )

    print(
        f"Saved: {SAMPLE_GT_FILE}"
    )


# ============================================================
# SQL KEY
# ============================================================

def key_expression(alias, column):

    return f"""
        NULLIF(
            TRIM(
                LOWER(
                    CAST(
                        {alias}.{column}
                        AS VARCHAR
                    )
                )
            ),
            ''
        )
    """


def composite_key(alias, columns):

    pieces = []

    for column in columns:

        pieces.append(
            f"""
            COALESCE(
                NULLIF(
                    TRIM(
                        LOWER(
                            CAST(
                                {alias}.{column}
                                AS VARCHAR
                            )
                        )
                    ),
                    ''
                ),
                '∅'
            )
            """
        )

    return " || '§' || ".join(
        pieces
    )


def rule_expression(rule_name, alias):

    columns = RULES[
        rule_name
    ]["columns"]

    if len(columns) == 1:

        return key_expression(
            alias,
            columns[0]
        )

    return composite_key(
        alias,
        columns
    )


# ============================================================
# CREATE SAMPLE VIEWS
# ============================================================

def create_sample_views(con):

    con.execute(
        f"""
        CREATE OR REPLACE VIEW sample_s1 AS

        SELECT *

        FROM read_parquet(
            '{sql_path(SAMPLE_S1_FILE)}'
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW sample_gt AS

        SELECT *

        FROM read_parquet(
            '{sql_path(SAMPLE_GT_FILE)}'
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s2 AS

        SELECT *

        FROM read_parquet(
            '{sql_path(S2_FILE)}'
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s3 AS

        SELECT *

        FROM read_parquet(
            '{sql_path(S3_FILE)}'
        );
        """
    )


# ============================================================
# DETERMINE TARGET SOURCE
# ============================================================

def classify_sample_gt(con):

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        sample_gt_classified AS

        SELECT

            g.source1_entity_id,

            g.target_entity_id,

            CASE

                WHEN s2.entity_id IS NOT NULL
                 AND s3.entity_id IS NOT NULL
                    THEN 'BOTH'

                WHEN s2.entity_id IS NOT NULL
                    THEN 'S2'

                WHEN s3.entity_id IS NOT NULL
                    THEN 'S3'

                ELSE 'UNKNOWN'

            END AS target_source

        FROM sample_gt g

        LEFT JOIN s2

            ON CAST(
                g.target_entity_id
                AS VARCHAR
            )
            =
            CAST(
                s2.entity_id
                AS VARCHAR
            )

        LEFT JOIN s3

            ON CAST(
                g.target_entity_id
                AS VARCHAR
            )
            =
            CAST(
                s3.entity_id
                AS VARCHAR
            );
        """
    )


# ============================================================
# CANDIDATE COUNT
# ============================================================

def candidate_count(
    con,
    target_view,
    rule_name
):

    s1_key = rule_expression(
        rule_name,
        "s"
    )

    target_key = rule_expression(
        rule_name,
        "t"
    )

    result = con.execute(
        f"""
        WITH

        sample_frequency AS (

            SELECT

                key,

                COUNT(*) AS s1_count

            FROM (

                SELECT

                    {s1_key}
                    AS key

                FROM sample_s1 s

            )

            WHERE key IS NOT NULL

            GROUP BY key
        ),

        target_frequency AS (

            SELECT

                key,

                COUNT(*) AS target_count

            FROM (

                SELECT

                    {target_key}
                    AS key

                FROM {target_view} t

            )

            WHERE key IS NOT NULL

            GROUP BY key
        )

        SELECT

            COALESCE(
                SUM(
                    sample_frequency.s1_count
                    *
                    target_frequency.target_count
                ),
                0
            )

        FROM sample_frequency

        INNER JOIN target_frequency

            ON sample_frequency.key
             =
               target_frequency.key;
        """
    ).fetchone()[0]

    return int(
        result or 0
    )


# ============================================================
# TRUE PAIR RECALL
# ============================================================

def true_pair_recall(
    con,
    target_source,
    target_view,
    rule_name
):

    s1_key = rule_expression(
        rule_name,
        "s"
    )

    target_key = rule_expression(
        rule_name,
        "t"
    )

    total = con.execute(
        f"""
        SELECT COUNT(*)

        FROM sample_gt_classified

        WHERE target_source =
              {sql_quote(target_source)};
        """
    ).fetchone()[0]

    if total == 0:

        return 0, 0.0

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only sampled GT pairs are evaluated.
    #
    # We do NOT generate candidate pairs.
    # --------------------------------------------------------

    retrieved = con.execute(
        f"""
        SELECT COUNT(*)

        FROM sample_gt_classified g

        INNER JOIN sample_s1 s

            ON CAST(
                g.source1_entity_id
                AS VARCHAR
            )
            =
            CAST(
                s.entity_id
                AS VARCHAR
            )

        INNER JOIN {target_view} t

            ON CAST(
                g.target_entity_id
                AS VARCHAR
            )
            =
            CAST(
                t.entity_id
                AS VARCHAR
            )

        WHERE g.target_source =
              {sql_quote(target_source)}

          AND {s1_key}
              IS NOT NULL

          AND {target_key}
              IS NOT NULL

          AND {s1_key}
              =
              {target_key};
        """
    ).fetchone()[0]

    retrieved = int(
        retrieved or 0
    )

    recall = (
        retrieved / total
        if total
        else 0.0
    )

    return retrieved, recall


# ============================================================
# UNION RECALL
# ============================================================

def union_recall(
    con,
    target_source,
    target_view
):

    total = con.execute(
        f"""
        SELECT COUNT(*)

        FROM sample_gt_classified

        WHERE target_source =
              {sql_quote(target_source)};
        """
    ).fetchone()[0]

    if total == 0:

        return 0, 0.0

    conditions = []

    for rule_name in RULES:

        s1_key = rule_expression(
            rule_name,
            "s"
        )

        target_key = rule_expression(
            rule_name,
            "t"
        )

        conditions.append(
            f"""
            (
                {s1_key} IS NOT NULL
                AND
                {target_key} IS NOT NULL
                AND
                {s1_key} = {target_key}
            )
            """
        )

    condition = "\nOR\n".join(
        conditions
    )

    retrieved = con.execute(
        f"""
        SELECT COUNT(*)

        FROM sample_gt_classified g

        INNER JOIN sample_s1 s

            ON CAST(
                g.source1_entity_id
                AS VARCHAR
            )
            =
            CAST(
                s.entity_id
                AS VARCHAR
            )

        INNER JOIN {target_view} t

            ON CAST(
                g.target_entity_id
                AS VARCHAR
            )
            =
            CAST(
                t.entity_id
                AS VARCHAR
            )

        WHERE g.target_source =
              {sql_quote(target_source)}

          AND (

              {condition}

          );
        """
    ).fetchone()[0]

    retrieved = int(
        retrieved or 0
    )

    recall = (
        retrieved / total
        if total
        else 0.0
    )

    return retrieved, recall


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(con):

    print()
    print("=" * 70)
    print("STEP 3: BLOCKING BENCHMARK")
    print("=" * 70)

    results = []

    sample_count = con.execute(
        "SELECT COUNT(*) FROM sample_s1"
    ).fetchone()[0]

    print()
    print(
        f"Development S1 sample: "
        f"{sample_count:,}"
    )

    for target_source in [
        "S2",
        "S3",
    ]:

        target_view = (
            "s2"
            if target_source == "S2"
            else "s3"
        )

        print()
        print("=" * 70)
        print(
            f"S1 -> {target_source}"
        )
        print("=" * 70)

        total_true = con.execute(
            f"""
            SELECT COUNT(*)

            FROM sample_gt_classified

            WHERE target_source =
                  {sql_quote(target_source)};
            """
        ).fetchone()[0]

        print()
        print(
            f"Sample true pairs: "
            f"{total_true:,}"
        )

        for number, rule_name in enumerate(
            RULES,
            start=1
        ):

            description = RULES[
                rule_name
            ]["description"]

            print()
            print(
                f"[{number}/10] "
                f"{rule_name}"
            )

            print(
                f"  {description}"
            )

            candidates = candidate_count(
                con,
                target_view,
                rule_name
            )

            retrieved, recall = (
                true_pair_recall(
                    con,
                    target_source,
                    target_view,
                    rule_name
                )
            )

            avg = (
                candidates / sample_count
                if sample_count
                else 0
            )

            print(
                f"  Recall      : "
                f"{recall:.4%}"
            )

            print(
                f"  True pairs  : "
                f"{retrieved:,} / "
                f"{total_true:,}"
            )

            print(
                f"  Candidates  : "
                f"{candidates:,}"
            )

            print(
                f"  Avg/S1      : "
                f"{avg:.2f}"
            )

            results.append(
                {
                    "target_source":
                        target_source,

                    "rule":
                        rule_name,

                    "description":
                        description,

                    "true_pairs":
                        int(total_true),

                    "true_pairs_retrieved":
                        int(retrieved),

                    "blocking_recall":
                        float(recall),

                    "candidate_pairs":
                        int(candidates),

                    "avg_candidates_per_s1":
                        float(avg),
                }
            )

        # ----------------------------------------------------
        # UNION
        # ----------------------------------------------------

        retrieved, recall = union_recall(
            con,
            target_source,
            target_view
        )

        print()
        print(
            "  ALL RULES UNION"
        )

        print(
            f"  Recall      : "
            f"{recall:.4%}"
        )

        print(
            f"  True pairs  : "
            f"{retrieved:,} / "
            f"{total_true:,}"
        )

        results.append(
            {
                "target_source":
                    target_source,

                "rule":
                    "ALL_EXACT_RULES_UNION",

                "description":
                    "Union of all exact blocking rules",

                "true_pairs":
                    int(total_true),

                "true_pairs_retrieved":
                    int(retrieved),

                "blocking_recall":
                    float(recall),

                "candidate_pairs":
                    None,

                "avg_candidates_per_s1":
                    None,
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("10% MEMORY-SAFE BLOCKING BENCHMARK")
    print("=" * 70)

    print()
    print(
        f"Sample fraction : "
        f"{SAMPLE_FRACTION:.0%}"
    )

    print(
        f"Random seed     : "
        f"{RANDOM_SEED}"
    )

    print(
        f"RAM limit       : "
        f"{DUCKDB_MEMORY}"
    )

    print(
        f"DuckDB threads  : "
        f"{DUCKDB_THREADS}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required = [
        S1_FILE,
        S2_FILE,
        S3_FILE,
        GROUND_TRUTH,
    ]

    for file in required:

        if not file.exists():

            raise FileNotFoundError(
                f"\nMissing file:\n{file}"
            )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    prepare_ground_truth()

    # --------------------------------------------------------
    # DuckDB
    # --------------------------------------------------------

    con = create_connection()

    try:

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        create_s1_sample(
            con
        )

        create_sample_ground_truth(
            con
        )

        # ----------------------------------------------------
        # Views
        # ----------------------------------------------------

        create_sample_views(
            con
        )

        # ----------------------------------------------------
        # Classify
        # ----------------------------------------------------

        classify_sample_gt(
            con
        )

        # ----------------------------------------------------
        # Benchmark
        # ----------------------------------------------------

        result_df = run_benchmark(
            con
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        result_df.to_csv(
            RESULT_FILE,
            index=False
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        print()

        print(
            result_df[
                [
                    "target_source",
                    "rule",
                    "blocking_recall",
                    "candidate_pairs",
                    "avg_candidates_per_s1",
                ]
            ]
            .sort_values(
                [
                    "target_source",
                    "blocking_recall",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .to_string(
                index=False
            )
        )

        print()
        print(
            "Results saved:"
        )

        print(
            f"  {RESULT_FILE}"
        )

        print()
        print(
            "10% blocking benchmark completed."
        )

    finally:

        con.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()