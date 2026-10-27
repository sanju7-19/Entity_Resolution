"""
STEP 15 - TRAIN LIGHTGBM ENTITY-RESOLUTION MODELS

Fixed version:
1. Does NOT use DuckDB USING SAMPLE reservoir(...).
2. Does NOT trust Step-14 is_validation because the 10% S1 sample
   and the old hash-based validation rule overlapped.
3. Creates a fresh deterministic S1-level train/validation split.
4. Keeps all positives.
5. Keeps hard negatives up to 6x training positives.
6. Adds easy negatives up to 4x training positives.
7. Tunes the threshold on the challenge's MACRO F0.5 metric per S1 entity.
8. Trains separate S1->S2 and S1->S3 models.

Run from:
C:\\Users\\sanja\\Downloads\\amazon_ml

python -u scripts\\15_train_lightgbm.py
"""

from pathlib import Path
import json
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "outputs" / "training_features"
S2_FEATURE_FILE = FEATURE_DIR / "train_s2_features.parquet"
S3_FEATURE_FILE = FEATURE_DIR / "train_s3_features.parquet"

GROUND_TRUTH = ROOT / "data" / "ground_truth" / "train_ground_truth.tsv"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "model_validation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_MEMORY = "4GB"
DUCKDB_THREADS = 4

HARD_NEGATIVE_MULTIPLIER = 4
EASY_NEGATIVE_MULTIPLIER = 4

VALIDATION_MOD = 5
VALIDATION_BUCKET = 0

RANDOM_SEED = 42

FEATURE_PREFIXES = (
    "name_jw",
    "address_jw",
    "postal_",
    "house_",
    "phone_",
    "domain_",
    "country_",
    "name_script_",
    "address_script_",
    "script_type_",
    "name_length_",
    "address_length_",
    "v1_",
    "v2_",
    "v3_",
)


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    """Windows path -> DuckDB-friendly path."""
    return str(path).replace("\\", "/")


def make_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    return con


def choose_column(columns, candidates, description):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise RuntimeError(
        f"Could not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{columns}"
    )


def inspect_feature_file(con, feature_file: Path, target_source: str):
    path = sql_path(feature_file)

    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')"
    ).fetchdf()

    columns = schema["column_name"].tolist()

    s1_col = choose_column(
        columns,
        [
            "s1_entity_id",
            "source1_entity_id",
            "s1_id",
        ],
        "Source-1 ID column",
    )

    if target_source == "S2":
        target_candidates = [
            "s2_entity_id",
            "target_entity_id",
            "s2_id",
            "target_id",
        ]
    else:
        target_candidates = [
            "s3_entity_id",
            "target_entity_id",
            "s3_id",
            "target_id",
        ]

    target_col = choose_column(
        columns,
        target_candidates,
        f"{target_source} target ID column",
    )

    label_col = choose_column(
        columns,
        ["label"],
        "label column",
    )

    # Numeric feature columns are selected by the feature naming scheme
    # created in Step 14.
    feature_cols = []

    for col in columns:
        lc = col.lower()

        if col in {s1_col, target_col, label_col, "is_validation"}:
            continue

        if lc.startswith(FEATURE_PREFIXES):
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError(
            f"No model feature columns were detected in {feature_file}"
        )

    # Verify the selected columns are numeric.
    type_map = dict(zip(schema["column_name"], schema["column_type"]))

    non_numeric = []
    numeric_features = []

    for col in feature_cols:
        dtype = str(type_map[col]).upper()

        if any(
            x in dtype
            for x in [
                "BOOLEAN",
                "TINYINT",
                "SMALLINT",
                "INTEGER",
                "BIGINT",
                "HUGEINT",
                "FLOAT",
                "DOUBLE",
                "DECIMAL",
            ]
        ):
            numeric_features.append(col)
        else:
            non_numeric.append((col, dtype))

    if non_numeric:
        print("Ignoring non-numeric feature columns:")
        for col, dtype in non_numeric:
            print(f"  {col}: {dtype}")

    feature_cols = numeric_features

    if not feature_cols:
        raise RuntimeError("No numeric model features remain.")

    return {
        "columns": columns,
        "s1_col": s1_col,
        "target_col": target_col,
        "label_col": label_col,
        "feature_cols": feature_cols,
    }


def coalesce_feature(cols, default="0.0"):
    if not cols:
        return default

    return "GREATEST(" + ", ".join(
        f"COALESCE(CAST({c} AS DOUBLE), 0.0)" for c in cols
    ) + ")"


def build_hard_negative_expression(feature_cols):
    terms = []

    name_jw = [
        c for c in feature_cols
        if c.lower() in {
            "name_jw_max",
            "name_jw_norm",
            "name_jw_core",
            "name_jw_translit",
        }
    ]

    if "name_jw_max" in feature_cols:
        terms.append(
            "COALESCE(CAST(name_jw_max AS DOUBLE), 0.0) >= 0.70"
        )
    elif name_jw:
        terms.append(
            f"{coalesce_feature(name_jw)} >= 0.70"
        )

    exact_signal_names = [
        "postal_exact",
        "house_exact",
        "phone_exact",
        "domain_exact",
    ]

    for c in exact_signal_names:
        if c in feature_cols:
            terms.append(
                f"COALESCE(CAST({c} AS DOUBLE), 0.0) >= 1.0"
            )

    v3_cols = [
        c for c in feature_cols
        if c.lower().startswith("v3_")
    ]

    for c in v3_cols:
        terms.append(
            f"COALESCE(CAST({c} AS DOUBLE), 0.0) >= 1.0"
        )

    if not terms:
        # Safe fallback: no negative is treated as hard.
        return "FALSE"

    return "(" + "\n OR ".join(terms) + ")"


def count_training_rows(
    con,
    feature_file,
    s1_col,
    label_col,
    train_split,
):
    path = sql_path(feature_file)

    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            SUM(CASE WHEN {label_col}=1 THEN 1 ELSE 0 END) AS positives,
            SUM(CASE WHEN {label_col}=0 THEN 1 ELSE 0 END) AS negatives
        FROM read_parquet('{path}')
        WHERE {train_split}
        """
    ).fetchone()

    return int(row[0]), int(row[1] or 0), int(row[2] or 0)


def load_training_data(
    con,
    feature_file,
    s1_col,
    target_col,
    label_col,
    feature_cols,
    hard_expr,
):
    path = sql_path(feature_file)

    # IMPORTANT:
    # We intentionally create a fresh split here.
    #
    # Step 14 sampled S1 using:
    #     hash(entity_id) % 10 = 0
    #
    # and also defined validation using:
    #     hash(entity_id) % 5 = 0
    #
    # Therefore every sampled S1 was validation.
    #
    # We use a salted hash so the split is independent.
    train_split = (
        f"(hash(CAST({s1_col} AS VARCHAR) || "
        f"'|step15_split_v2') % {VALIDATION_MOD}) "
        f"<> {VALIDATION_BUCKET}"
    )

    select_features = ",\n            ".join(feature_cols)

    # Count training positives first so the negative caps are data-driven.
    positive_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{path}')
        WHERE {train_split}
          AND {label_col}=1
        """
    ).fetchone()[0]

    positive_count = int(positive_count)

    if positive_count == 0:
        raise RuntimeError(
            "Training positives are zero. "
            "The independent S1 split did not work."
        )

    hard_limit = positive_count * HARD_NEGATIVE_MULTIPLIER
    easy_limit = positive_count * EASY_NEGATIVE_MULTIPLIER

    print(f"Training positives available : {positive_count:,}")
    print(f"Hard-negative cap             : {hard_limit:,}")
    print(f"Easy-negative cap             : {easy_limit:,}")

    base_columns = f"""
            CAST({s1_col} AS VARCHAR) AS s1_id,
            CAST({target_col} AS VARCHAR) AS target_id,
            {select_features},
            CAST({label_col} AS INTEGER) AS label
    """

    # No USING SAMPLE.
    #
    # Hard negatives and easy negatives are selected deterministically
    # using ROW_NUMBER + hash, so reruns produce the same data.
    query = f"""
        WITH base AS (
            SELECT
                {base_columns}
            FROM read_parquet('{path}')
            WHERE {train_split}
        ),

        positives AS (
            SELECT *
            FROM base
            WHERE label = 1
        ),

        hard_pool AS (
            SELECT *
            FROM base
            WHERE label = 0
              AND {hard_expr}
        ),

        hard_negatives AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        ORDER BY hash(s1_id || '|' || target_id || '|hard')
                    ) AS rn
                FROM hard_pool
            )
            WHERE rn <= {hard_limit}
        ),

        easy_pool AS (
            SELECT *
            FROM base
            WHERE label = 0
              AND NOT ({hard_expr})
        ),

        easy_negatives AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        ORDER BY hash(s1_id || '|' || target_id || '|easy')
                    ) AS rn
                FROM easy_pool
            )
            WHERE rn <= {easy_limit}
        )

        SELECT *
        FROM positives

        UNION ALL

        SELECT * EXCLUDE (rn)
        FROM hard_negatives

        UNION ALL

        SELECT * EXCLUDE (rn)
        FROM easy_negatives
    """

    print()
    print("Sampling training pairs...")

    df = con.execute(query).fetchdf()

    if df.empty:
        raise RuntimeError(
            "Training query returned ZERO rows."
        )

    df["label"] = df["label"].astype(np.int8)

    # Defensive numeric conversion.
    for col in feature_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0.0).astype(np.float32)

    positives = int(df["label"].sum())
    negatives = int(len(df) - positives)

    print(f"Training rows selected : {len(df):,}")
    print(f"Training positives     : {positives:,}")
    print(f"Training negatives     : {negatives:,}")

    return df, train_split


def load_validation_data(
    con,
    feature_file,
    s1_col,
    target_col,
    label_col,
    feature_cols,
):
    path = sql_path(feature_file)

    # Same salted split used by training-data preparation.
    validation_split = (
        f"(hash(CAST({s1_col} AS VARCHAR) || "
        f"'|step15_split_v2') % {VALIDATION_MOD}) "
        f"= {VALIDATION_BUCKET}"
    )

    select_features = ",\n            ".join(feature_cols)

    query = f"""
        SELECT
            CAST({s1_col} AS VARCHAR) AS s1_id,
            CAST({target_col} AS VARCHAR) AS target_id,
            {select_features},
            CAST({label_col} AS INTEGER) AS label
        FROM read_parquet('{path}')
        WHERE {validation_split}
    """

    print()
    print("Loading validation data...")

    df = con.execute(query).fetchdf()

    if df.empty:
        raise RuntimeError(
            "Validation query returned ZERO rows."
        )

    df["label"] = df["label"].astype(np.int8)

    for col in feature_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0.0).astype(np.float32)

    print(f"Validation pairs    : {len(df):,}")
    print(f"Validation positives: {int(df['label'].sum()):,}")
    print(f"Validation S1 count : {df['s1_id'].nunique():,}")

    return df


def load_ground_truth_counts(
    con,
    target_source,
):
    gt_path = sql_path(GROUND_TRUTH)

    query = f"""
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1_id,
            COUNT(*) AS gt_count
        FROM read_csv(
            '{gt_path}',
            delim='\\t',
            header=true,
            all_varchar=true
        ) AS g
        CROSS JOIN UNNEST(
            string_split(
                COALESCE(g.matched_entity_ids, ''),
                ','
            )
        ) AS u(target_id)
        WHERE TRIM(u.target_id) LIKE '{target_source}-%'
        GROUP BY 1
    """

    return con.execute(query).fetchdf()


def macro_f05_for_threshold(
    validation_df,
    gt_counts,
    threshold,
):
    scores = validation_df["score"].to_numpy()
    labels = validation_df["label"].to_numpy().astype(np.int8)
    s1_ids = validation_df["s1_id"].to_numpy()

    predicted = (scores >= threshold).astype(np.int8)

    # Per-candidate:
    # predicted = 1 if model selects the pair
    # tp = 1 only when selected pair is actually a GT pair
    temp = pd.DataFrame(
        {
            "s1_id": s1_ids,
            "predicted": predicted,
            "tp": predicted * labels,
        }
    )

    grouped = temp.groupby(
        "s1_id",
        sort=False,
    ).sum()

    all_s1 = validation_df["s1_id"].drop_duplicates()

    grouped = grouped.reindex(
        all_s1,
        fill_value=0,
    )

    gt_map = gt_counts.set_index("s1_id")["gt_count"]

    total_gt = (
        gt_map.reindex(all_s1)
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )

    pred_count = grouped["predicted"].to_numpy(dtype=np.int64)
    tp = grouped["tp"].to_numpy(dtype=np.int64)

    fp = pred_count - tp
    fn = total_gt - tp

    # F0.5:
    #
    # (1 + beta^2) TP
    # -------------------------
    # (1 + beta^2) TP + beta^2 FN + FP
    #
    # beta = 0.5 => beta^2 = 0.25
    beta2 = 0.25

    numerator = (1.0 + beta2) * tp
    denominator = numerator + beta2 * fn + fp

    scores_per_entity = np.divide(
    numerator,
    denominator,
    out=np.ones_like(numerator, dtype=float),
    where=(denominator != 0)
)

    return float(np.mean(scores_per_entity))


def tune_macro_f05(
    validation_df,
    gt_counts,
):
    print()
    print("=" * 70)
    print("TUNING THRESHOLD ON MACRO F0.5")
    print("=" * 70)

    # Coarse pass.
    coarse = np.arange(
        0.05,
        0.991,
        0.01,
        dtype=np.float64,
    )

    rows = []

    best_threshold = None
    best_score = -1.0

    for threshold in coarse:
        score = macro_f05_for_threshold(
            validation_df,
            gt_counts,
            float(threshold),
        )

        rows.append(
            {
                "threshold": float(threshold),
                "macro_f05": score,
                "stage": "coarse",
            }
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    # Fine pass around coarse optimum.
    fine_start = max(0.05, best_threshold - 0.02)
    fine_end = min(0.99, best_threshold + 0.02)

    fine = np.arange(
        fine_start,
        fine_end + 0.0005,
        0.001,
        dtype=np.float64,
    )

    for threshold in fine:
        score = macro_f05_for_threshold(
            validation_df,
            gt_counts,
            float(threshold),
        )

        rows.append(
            {
                "threshold": float(threshold),
                "macro_f05": score,
                "stage": "fine",
            }
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    threshold_df = pd.DataFrame(rows)

    # Remove duplicate thresholds from coarse/fine overlap.
    threshold_df = (
        threshold_df
        .sort_values(
            ["macro_f05", "threshold"],
            ascending=[False, True],
        )
        .drop_duplicates(
            subset=["threshold"],
            keep="first",
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )

    print(f"Best threshold : {best_threshold:.4f}")
    print(f"Best Macro F0.5 : {best_score:.6f}")

    return (
        float(best_threshold),
        float(best_score),
        threshold_df,
    )


def train_one_source(
    target_source,
    feature_file,
):
    start = time.time()

    print()
    print("#" * 70)
    print(f"TRAINING MODEL: S1 -> {target_source}")
    print("#" * 70)

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{feature_file}"
        )

    con = make_connection()

    try:
        info = inspect_feature_file(
            con,
            feature_file,
            target_source,
        )

        s1_col = info["s1_col"]
        target_col = info["target_col"]
        label_col = info["label_col"]
        feature_cols = info["feature_cols"]

        print()
        print("Feature file:")
        print(f"  {feature_file}")

        print()
        print("Detected columns:")
        print(f"  S1 ID      : {s1_col}")
        print(f"  Target ID  : {target_col}")
        print(f"  Label      : {label_col}")
        print(f"  Features   : {len(feature_cols)}")

        print()
        print("Model features:")
        for col in feature_cols:
            print(f"  {col}")

        hard_expr = build_hard_negative_expression(
            feature_cols
        )

        print()
        print("Hard-negative rule:")
        print(hard_expr)

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        training_df, train_split = load_training_data(
            con,
            feature_file,
            s1_col,
            target_col,
            label_col,
            feature_cols,
            hard_expr,
        )

        # ----------------------------------------------------
        # VALIDATION DATA
        # ----------------------------------------------------

        validation_df = load_validation_data(
            con,
            feature_file,
            s1_col,
            target_col,
            label_col,
            feature_cols,
        )

        # ----------------------------------------------------
        # GROUND-TRUTH COUNTS
        # ----------------------------------------------------

        print()
        print("Loading ground-truth counts...")

        gt_counts = load_ground_truth_counts(
            con,
            target_source,
        )

        validation_s1 = validation_df["s1_id"].nunique()

        gt_validation = validation_df[
            ["s1_id"]
        ].drop_duplicates().merge(
            gt_counts,
            on="s1_id",
            how="left",
        )

        gt_validation["gt_count"] = (
            gt_validation["gt_count"]
            .fillna(0)
            .astype(np.int64)
        )

        print(
            f"Validation S1 entities with GT: "
            f"{len(gt_validation):,}"
        )

        print(
            f"Ground-truth links for validation S1: "
            f"{int(gt_validation['gt_count'].sum()):,}"
        )

        candidate_recall = (
            float(validation_df["label"].sum())
            / float(gt_validation["gt_count"].sum())
            if gt_validation["gt_count"].sum() > 0
            else 0.0
        )

        print(
            f"Candidate recall on validation: "
            f"{candidate_recall:.4%}"
        )

        # ----------------------------------------------------
        # ARRAYS
        # ----------------------------------------------------

        X_train = training_df[feature_cols]
        y_train = training_df["label"].astype(np.int8)

        X_val = validation_df[feature_cols]
        y_val = validation_df["label"].astype(np.int8)

        # Free unnecessary copies before LightGBM.
        del training_df

        print()
        print("Training LightGBM...")

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=DUCKDB_THREADS,
            verbosity=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val),
            ],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=80,
                    verbose=True,
                ),
                lgb.log_evaluation(
                    period=50,
                ),
            ],
        )

        print()
        print(
            f"Best iteration: "
            f"{model.best_iteration_}"
        )

        # ----------------------------------------------------
        # VALIDATION SCORES
        # ----------------------------------------------------

        print()
        print("Scoring validation pairs...")

        validation_df["score"] = model.predict_proba(
            X_val,
        )[:, 1]

        validation_output = OUTPUT_DIR / (
            f"validation_predictions_{target_source.lower()}.parquet"
        )

        validation_df[
            [
                "s1_id",
                "target_id",
                "label",
                "score",
            ]
        ].to_parquet(
            validation_output,
            index=False,
        )

        print(
            f"Saved validation predictions:\n"
            f"  {validation_output}"
        )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        best_threshold, best_macro_f05, threshold_df = (
            tune_macro_f05(
                validation_df,
                gt_validation,
            )
        )

        threshold_output = OUTPUT_DIR / (
            f"threshold_search_{target_source.lower()}.csv"
        )

        threshold_df.to_csv(
            threshold_output,
            index=False,
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_output = MODEL_DIR / (
            f"lgbm_{target_source.lower()}.txt"
        )

        model.booster_.save_model(
            str(model_output)
        )

        print()
        print(f"Saved model:\n  {model_output}")

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        runtime_min = (time.time() - start) / 60.0

        summary = {
            "target_source": target_source,
            "feature_file": str(feature_file),
            "feature_count": len(feature_cols),
            "training_rows": int(len(X_train)),
            "training_positives": int(y_train.sum()),
            "training_negatives": int(len(y_train) - y_train.sum()),
            "validation_rows": int(len(validation_df)),
            "validation_positives": int(y_val.sum()),
            "validation_s1_entities": int(validation_s1),
            "validation_ground_truth_links": int(
                gt_validation["gt_count"].sum()
            ),
            "candidate_recall_validation": candidate_recall,
            "best_iteration": int(model.best_iteration_),
            "best_threshold": best_threshold,
            "validation_macro_f05": best_macro_f05,
            "runtime_minutes": runtime_min,
            "model_file": str(model_output),
            "validation_prediction_file": str(validation_output),
            "threshold_file": str(threshold_output),
        }

        print()
        print("=" * 70)
        print(f"S1 -> {target_source} COMPLETE")
        print("=" * 70)
        print(
            f"Training rows          : "
            f"{summary['training_rows']:,}"
        )
        print(
            f"Training positives     : "
            f"{summary['training_positives']:,}"
        )
        print(
            f"Training negatives     : "
            f"{summary['training_negatives']:,}"
        )
        print(
            f"Validation rows        : "
            f"{summary['validation_rows']:,}"
        )
        print(
            f"Validation GT links    : "
            f"{summary['validation_ground_truth_links']:,}"
        )
        print(
            f"Candidate recall       : "
            f"{candidate_recall:.4%}"
        )
        print(
            f"Best iteration         : "
            f"{summary['best_iteration']}"
        )
        print(
            f"Best threshold         : "
            f"{best_threshold:.4f}"
        )
        print(
            f"Validation Macro F0.5  : "
            f"{best_macro_f05:.6f}"
        )
        print(
            f"Runtime                : "
            f"{runtime_min:.2f} minutes"
        )

        return summary

    finally:
        con.close()


def main():
    overall_start = time.time()

    print()
    print("=" * 70)
    print("STEP 15 - LIGHTGBM ENTITY RESOLUTION TRAINING")
    print("=" * 70)

    print()
    print(f"Project root: {ROOT}")
    print(f"S2 features : {S2_FEATURE_FILE}")
    print(f"S3 features : {S3_FEATURE_FILE}")
    print(f"Ground truth: {GROUND_TRUTH}")

    required = [
        S2_FEATURE_FILE,
        S3_FEATURE_FILE,
        GROUND_TRUTH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    s2_summary = train_one_source(
        "S2",
        S2_FEATURE_FILE,
    )

    s3_summary = train_one_source(
        "S3",
        S3_FEATURE_FILE,
    )

    summaries = pd.DataFrame(
        [
            s2_summary,
            s3_summary,
        ]
    )

    summary_file = MODEL_DIR / "training_summary.csv"

    summaries.to_csv(
        summary_file,
        index=False,
    )

    thresholds = {
        "S2": s2_summary["best_threshold"],
        "S3": s3_summary["best_threshold"],
    }

    threshold_json = MODEL_DIR / "thresholds.json"

    with open(
        threshold_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            thresholds,
            f,
            indent=2,
        )

    runtime_min = (
        time.time() - overall_start
    ) / 60.0

    print()
    print("=" * 70)
    print("STEP 15 COMPLETE")
    print("=" * 70)

    print()
    print(f"Training summary:")
    print(f"  {summary_file}")

    print()
    print(f"Thresholds:")
    print(f"  {threshold_json}")

    print()
    print("Models:")
    print(f"  {MODEL_DIR / 'lgbm_s2.txt'}")
    print(f"  {MODEL_DIR / 'lgbm_s3.txt'}")

    print()
    print(f"Total runtime: {runtime_min:.2f} minutes")


if __name__ == "__main__":
    main()
