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
RESULT_FILE = OUTPUT / "blocking_benchmark.csv"


# ============================================================
# MEMORY-SAFE DUCKDB SETTINGS
# ============================================================

DUCKDB_THREADS = 2

# Keep below physical RAM so Windows + VS Code + Python
# still have memory available.
DUCKDB_MEMORY = "7GB"


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
    """
    Convert Windows Path to a DuckDB-safe SQL path.
    """
    return str(path).replace("\\", "/").replace("'", "''")


def quote_sql_string(value: str) -> str:
    """
    Safely quote a string for SQL.
    """
    return "'" + str(value).replace("'", "''") + "'"


def parse_match_ids(value):
    """
    Convert matched_entity_ids into a list.

    Supports:
        123,456
        123 456
        123;456
        123|456
        [123,456]
        ["123","456"]
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    value = value.strip("[](){}")
    value = value.replace('"', "")
    value = value.replace("'", "")

    parts = re.split(r"[,;|\s]+", value)

    result = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if part.lower() in {
            "nan",
            "none",
            "null",
        }:
            continue

        result.append(part)

    return result


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

def prepare_ground_truth():

    print()
    print("=" * 70)
    print("STEP 1: PREPARING GROUND TRUTH")
    print("=" * 70)

    if GT_PAIRS_FILE.exists():

        print()
        print("Ground-truth pairs already exist:")
        print(f"  {GT_PAIRS_FILE}")
        print("Reusing existing file.")

        return

    print()
    print("Reading:")
    print(f"  {GROUND_TRUTH}")

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

    total_s1_rows = 0
    total_links = 0

    for chunk_number, chunk in enumerate(reader, start=1):

        if "source1_entity_id" not in chunk.columns:
            raise ValueError(
                "Missing column: source1_entity_id"
            )

        if "matched_entity_ids" not in chunk.columns:
            raise ValueError(
                "Missing column: matched_entity_ids"
            )

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

        total_s1_rows += len(chunk)
        total_links += len(rows)

        if rows:

            output_chunk = pd.DataFrame(rows)

            table = pa.Table.from_pandas(
                output_chunk,
                preserve_index=False,
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    GT_PAIRS_FILE,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

        print(
            f"  chunk {chunk_number:>3} | "
            f"S1 rows: {len(chunk):>9,} | "
            f"links: {len(rows):>9,}"
        )

        # Explicitly release chunk objects.
        del rows
        del chunk

    if writer is not None:
        writer.close()

    print()
    print(f"Ground-truth S1 rows : {total_s1_rows:,}")
    print(f"Ground-truth links   : {total_links:,}")
    print(f"Saved to             : {GT_PAIRS_FILE}")


# ============================================================
# DUCKDB CONNECTION
# ============================================================

def create_connection():

    print()
    print("=" * 70)
    print("STEP 2: STARTING MEMORY-SAFE DUCKDB")
    print("=" * 70)

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

    # Reduce unnecessary temporary file creation.
    con.execute(
        "SET enable_progress_bar=false"
    )

    print(
        f"Threads       : {DUCKDB_THREADS}"
    )

    print(
        f"Memory limit  : {DUCKDB_MEMORY}"
    )

    # --------------------------------------------------------
    # Views
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s1 AS
        SELECT *
        FROM read_parquet(
            '{sql_path(S1_FILE)}'
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

    con.execute(
        f"""
        CREATE OR REPLACE VIEW gt AS
        SELECT *
        FROM read_parquet(
            '{sql_path(GT_PAIRS_FILE)}'
        );
        """
    )

    return con


# ============================================================
# BASIC DATA COUNTS
# ============================================================

def print_basic_counts(con):

    print()
    print("=" * 70)
    print("STEP 3: DATA COUNTS")
    print("=" * 70)

    for source in ["s1", "s2", "s3"]:

        count = con.execute(
            f"SELECT COUNT(*) FROM {source}"
        ).fetchone()[0]

        print(
            f"{source.upper():<4}: {count:,} rows"
        )

    gt_count = con.execute(
        "SELECT COUNT(*) FROM gt"
    ).fetchone()[0]

    print(
        f"GT links: {gt_count:,}"
    )


# ============================================================
# CLASSIFY GROUND TRUTH
# ============================================================

def classify_ground_truth_targets(con):

    print()
    print("=" * 70)
    print("STEP 4: CLASSIFYING GROUND-TRUTH TARGET IDS")
    print("=" * 70)

    # --------------------------------------------------------
    # Instead of joining GT to S2 AND S3 simultaneously,
    # classify in two smaller passes.
    # --------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_s2_ids AS

        SELECT DISTINCT
            CAST(entity_id AS VARCHAR) AS target_entity_id

        FROM s2;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_s3_ids AS

        SELECT DISTINCT
            CAST(entity_id AS VARCHAR) AS target_entity_id

        FROM s3;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_classified AS

        SELECT
            g.source1_entity_id,
            g.target_entity_id,

            CASE

                WHEN s2.target_entity_id IS NOT NULL
                 AND s3.target_entity_id IS NOT NULL
                    THEN 'BOTH'

                WHEN s2.target_entity_id IS NOT NULL
                    THEN 'S2'

                WHEN s3.target_entity_id IS NOT NULL
                    THEN 'S3'

                ELSE 'UNKNOWN'

            END AS target_source

        FROM gt g

        LEFT JOIN gt_s2_ids s2
            ON CAST(g.target_entity_id AS VARCHAR)
             = s2.target_entity_id

        LEFT JOIN gt_s3_ids s3
            ON CAST(g.target_entity_id AS VARCHAR)
             = s3.target_entity_id;
        """
    )

    result = con.execute(
        """
        SELECT
            target_source,
            COUNT(*) AS links

        FROM gt_classified

        GROUP BY target_source

        ORDER BY target_source;
        """
    ).fetchdf()

    print()
    print(result.to_string(index=False))

    unknown = con.execute(
        """
        SELECT COUNT(*)
        FROM gt_classified
        WHERE target_source = 'UNKNOWN';
        """
    ).fetchone()[0]

    both = con.execute(
        """
        SELECT COUNT(*)
        FROM gt_classified
        WHERE target_source = 'BOTH';
        """
    ).fetchone()[0]

    if unknown:

        print()
        print(
            f"WARNING: {unknown:,} "
            f"UNKNOWN target links."
        )

    if both:

        print()
        print(
            f"WARNING: {both:,} "
            f"target IDs occur in BOTH sources."
        )


# ============================================================
# MATCH CATEGORIES
# ============================================================

def create_entity_categories(con):

    print()
    print("=" * 70)
    print("STEP 5: S1 MATCH CATEGORIES")
    print("=" * 70)

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_entity_stats AS

        SELECT

            s.entity_id
                AS source1_entity_id,

            COUNT(g.target_entity_id)
                AS true_match_count,

            CASE

                WHEN COUNT(g.target_entity_id) = 0
                    THEN 'NO_MATCH'

                WHEN COUNT(g.target_entity_id) = 1
                    THEN 'SINGLETON'

                ELSE 'MULTI_MATCH'

            END AS match_category

        FROM s1 s

        LEFT JOIN gt g

            ON CAST(s.entity_id AS VARCHAR)
             = CAST(g.source1_entity_id AS VARCHAR)

        GROUP BY s.entity_id;
        """
    )

    result = con.execute(
        """
        SELECT
            match_category,
            COUNT(*) AS s1_entities

        FROM gt_entity_stats

        GROUP BY match_category

        ORDER BY
            CASE match_category

                WHEN 'NO_MATCH'
                    THEN 1

                WHEN 'SINGLETON'
                    THEN 2

                WHEN 'MULTI_MATCH'
                    THEN 3

            END;
        """
    ).fetchdf()

    print()
    print(result.to_string(index=False))


# ============================================================
# CREATE SQL KEY EXPRESSION
# ============================================================

def column_key(alias, column):

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

    expressions = []

    for column in columns:

        expressions.append(
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
        expressions
    )


def rule_key(rule_name, alias):

    columns = RULES[rule_name]["columns"]

    if len(columns) == 1:

        return column_key(
            alias,
            columns[0],
        )

    return composite_key(
        alias,
        columns,
    )


# ============================================================
# CANDIDATE COUNT USING FREQUENCIES
# ============================================================

def calculate_candidate_count(
    con,
    target_view,
    rule_name,
):

    s1_key = rule_key(
        rule_name,
        "s",
    )

    target_key = rule_key(
        rule_name,
        "t",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT create:
    #
    #     S1 x target
    #
    # Instead:
    #
    #     frequency(S1 key)
    #              x
    #     frequency(target key)
    #
    # --------------------------------------------------------

    candidate_count = con.execute(
        f"""
        WITH

        s1_frequency AS (

            SELECT
                key,
                COUNT(*) AS s1_count

            FROM (

                SELECT
                    {s1_key} AS key

                FROM s1 s

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
                    {target_key} AS key

                FROM {target_view} t

            )

            WHERE key IS NOT NULL

            GROUP BY key
        )

        SELECT
            COALESCE(
                SUM(
                    s1_frequency.s1_count
                    *
                    target_frequency.target_count
                ),
                0
            )

        FROM s1_frequency

        INNER JOIN target_frequency

            ON s1_frequency.key
             = target_frequency.key;
        """
    ).fetchone()[0]

    return int(
        candidate_count or 0
    )


# ============================================================
# TRUE PAIR RECALL
# ============================================================

def calculate_true_pair_recall(
    con,
    target_source,
    target_view,
    rule_name,
):

    rule_s1_key = rule_key(
        rule_name,
        "s",
    )

    rule_target_key = rule_key(
        rule_name,
        "t",
    )

    total_true = con.execute(
        f"""
        SELECT COUNT(*)

        FROM gt_classified

        WHERE target_source =
              {quote_sql_string(target_source)};
        """
    ).fetchone()[0]

    if total_true == 0:

        return 0, 0.0

    # --------------------------------------------------------
    # CRITICAL:
    #
    # This joins ONLY the ground-truth pairs.
    #
    # It does NOT generate arbitrary S1 x target pairs.
    # --------------------------------------------------------

    retrieved = con.execute(
        f"""
        SELECT COUNT(*)

        FROM gt_classified g

        INNER JOIN s1 s

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
              {quote_sql_string(target_source)}

          AND {rule_s1_key}
              IS NOT NULL

          AND {rule_target_key}
              IS NOT NULL

          AND {rule_s1_key}
              =
              {rule_target_key};
        """
    ).fetchone()[0]

    retrieved = int(
        retrieved or 0
    )

    recall = (
        retrieved / total_true
        if total_true
        else 0.0
    )

    return retrieved, recall


# ============================================================
# UNION RECALL
# ============================================================

def calculate_union_recall(
    con,
    target_source,
    target_view,
):

    total_true = con.execute(
        f"""
        SELECT COUNT(*)

        FROM gt_classified

        WHERE target_source =
              {quote_sql_string(target_source)};
        """
    ).fetchone()[0]

    if total_true == 0:

        return 0, 0.0

    # --------------------------------------------------------
    # Build one OR condition.
    #
    # This still evaluates ONLY ground-truth pairs.
    #
    # It never creates all candidate pairs.
    # --------------------------------------------------------

    conditions = []

    for rule_name in RULES:

        s1_key = rule_key(
            rule_name,
            "s",
        )

        target_key = rule_key(
            rule_name,
            "t",
        )

        conditions.append(
            f"""
            (
                {s1_key} IS NOT NULL

                AND

                {target_key} IS NOT NULL

                AND

                {s1_key}
                =
                {target_key}
            )
            """
        )

    union_condition = "\nOR\n".join(
        conditions
    )

    retrieved = con.execute(
        f"""
        SELECT COUNT(*)

        FROM gt_classified g

        INNER JOIN s1 s

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
              {quote_sql_string(target_source)}

          AND (
                {union_condition}
          );
        """
    ).fetchone()[0]

    retrieved = int(
        retrieved or 0
    )

    recall = (
        retrieved / total_true
        if total_true
        else 0.0
    )

    return retrieved, recall


# ============================================================
# BENCHMARK ONE SOURCE
# ============================================================

def benchmark_source(
    con,
    target_source,
):

    target_view = (
        "s2"
        if target_source == "S2"
        else "s3"
    )

    print()
    print("=" * 70)
    print(
        f"BLOCKING BENCHMARK: S1 -> {target_source}"
    )
    print("=" * 70)

    total_true_pairs = con.execute(
        f"""
        SELECT COUNT(*)

        FROM gt_classified

        WHERE target_source =
              {quote_sql_string(target_source)};
        """
    ).fetchone()[0]

    s1_count = con.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    print()
    print(
        f"True {target_source} links: "
        f"{total_true_pairs:,}"
    )

    print(
        f"S1 entities: "
        f"{s1_count:,}"
    )

    results = []

    for index, (
        rule_name,
        rule_info,
    ) in enumerate(
        RULES.items(),
        start=1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"[{index}/{len(RULES)}] "
            f"Testing: {rule_name}"
        )

        print(
            f"  {rule_info['description']}"
        )

        # ----------------------------------------------------
        # Candidate count
        # ----------------------------------------------------

        candidate_count = (
            calculate_candidate_count(
                con,
                target_view,
                rule_name,
            )
        )

        # ----------------------------------------------------
        # True-pair recall
        # ----------------------------------------------------

        (
            true_retrieved,
            recall,
        ) = calculate_true_pair_recall(
            con,
            target_source,
            target_view,
            rule_name,
        )

        avg_candidates = (
            candidate_count / s1_count
            if s1_count
            else 0.0
        )

        result = {

            "target_source":
                target_source,

            "rule":
                rule_name,

            "description":
                rule_info["description"],

            "true_pairs":
                int(total_true_pairs),

            "true_pairs_retrieved":
                int(true_retrieved),

            "blocking_recall":
                float(recall),

            "candidate_pairs":
                int(candidate_count),

            "avg_candidates_per_s1":
                float(avg_candidates),
        }

        results.append(
            result
        )

        print()
        print(
            f"  True pairs retrieved : "
            f"{true_retrieved:,} / "
            f"{total_true_pairs:,}"
        )

        print(
            f"  Blocking recall      : "
            f"{recall:.4%}"
        )

        print(
            f"  Candidate pairs      : "
            f"{candidate_count:,}"
        )

        print(
            f"  Avg candidates/S1    : "
            f"{avg_candidates:.2f}"
        )

        # ----------------------------------------------------
        # Force DuckDB to clean temporary state.
        # ----------------------------------------------------

        con.execute(
            "CHECKPOINT"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MEMORY-SAFE BLOCKING BENCHMARK")
    print("=" * 70)

    print()
    print(
        "This benchmark does NOT materialize "
        "the full candidate pair space."
    )

    print()
    print(
        f"RAM limit: {DUCKDB_MEMORY}"
    )

    print(
        f"DuckDB threads: {DUCKDB_THREADS}"
    )

    print()
    print("Normalized files:")

    print(
        f"  S1: {S1_FILE}"
    )

    print(
        f"  S2: {S2_FILE}"
    )

    print(
        f"  S3: {S3_FILE}"
    )

    print()
    print("Ground truth:")

    print(
        f"  {GROUND_TRUTH}"
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    required_files = [

        S1_FILE,
        S2_FILE,
        S3_FILE,
        GROUND_TRUTH,
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n"
                f"{file}"
            )

    # ========================================================
    # PREPARE GROUND TRUTH
    # ========================================================

    prepare_ground_truth()

    # ========================================================
    # CONNECTION
    # ========================================================

    con = create_connection()

    try:

        # ====================================================
        # BASIC COUNTS
        # ====================================================

        print_basic_counts(
            con
        )

        # ====================================================
        # CLASSIFY TARGETS
        # ====================================================

        classify_ground_truth_targets(
            con
        )

        # ====================================================
        # ENTITY CATEGORIES
        # ====================================================

        create_entity_categories(
            con
        )

        # ====================================================
        # BENCHMARK S2 + S3
        # ====================================================

        all_results = []

        for target_source in [
            "S2",
            "S3",
        ]:

            results = benchmark_source(
                con,
                target_source,
            )

            all_results.extend(
                results
            )

            # ----------------------------------------------
            # UNION RECALL
            # ----------------------------------------------

            target_view = (
                "s2"
                if target_source == "S2"
                else "s3"
            )

            print()
            print("=" * 70)
            print(
                f"ALL RULES UNION: S1 -> "
                f"{target_source}"
            )
            print("=" * 70)

            (
                union_retrieved,
                union_recall,
            ) = calculate_union_recall(
                con,
                target_source,
                target_view,
            )

            total_true = con.execute(
                f"""
                SELECT COUNT(*)

                FROM gt_classified

                WHERE target_source =
                      {quote_sql_string(target_source)};
                """
            ).fetchone()[0]

            print()
            print(
                f"True pairs          : "
                f"{total_true:,}"
            )

            print(
                f"Retrieved by union  : "
                f"{union_retrieved:,}"
            )

            print(
                f"Union blocking recall: "
                f"{union_recall:.4%}"
            )

            all_results.append(
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
                        int(union_retrieved),

                    "blocking_recall":
                        float(union_recall),

                    "candidate_pairs":
                        None,

                    "avg_candidates_per_s1":
                        None,
                }
            )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        result_df = pd.DataFrame(
            all_results
        )

        result_df.to_csv(
            RESULT_FILE,
            index=False,
        )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print()
        print("=" * 70)
        print("FINAL BLOCKING BENCHMARK")
        print("=" * 70)

        display_columns = [

            "target_source",

            "rule",

            "blocking_recall",

            "candidate_pairs",

            "avg_candidates_per_s1",
        ]

        summary = (
            result_df[
                display_columns
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
        )

        print()

        print(
            summary.to_string(
                index=False
            )
        )

        print()
        print(
            f"Results saved to:"
        )

        print(
            f"  {RESULT_FILE}"
        )

        print()
        print(
            "Benchmark completed successfully."
        )

    finally:

        # ----------------------------------------------------
        # Close connection.
        # ----------------------------------------------------

        con.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()