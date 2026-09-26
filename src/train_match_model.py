from pathlib import Path
import sys
import joblib

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    BASE_DIR
    / "experiments"
    / "005_ml_training"
    / "training_pair_features.tsv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "005_ml_baseline"
)

MODEL_PATH = (
    MODEL_DIR
    / "hist_gradient_boosting_match_model.joblib"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

THRESHOLD = 0.50


# =============================================================================
# FEATURE COLUMNS
# =============================================================================

FEATURE_COLUMNS = [

    # -------------------------------------------------------------------------
    # NAME
    # -------------------------------------------------------------------------

    "name_exact",
    "name_char_ratio",
    "name_jaro",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap",
    "name_length_diff",
    "name_token_count_diff",
    "name_no_suffix_ratio",
    "name_compact_ratio",
    "name_sorted_ratio",

    # -------------------------------------------------------------------------
    # ADDRESS
    # -------------------------------------------------------------------------

    "address_exact",
    "address_char_ratio",
    "address_jaro",
    "address_jaro_winkler",
    "address_token_jaccard",
    "address_token_overlap",
    "address_length_diff",
    "address_token_count_diff",
    "address_number_overlap",
    "address_number_match",
    "address_number_conflict",
    "address_postal_match",
    "address_postal_conflict",

    # -------------------------------------------------------------------------
    # COUNTRY
    # -------------------------------------------------------------------------

    "country_exact",
    "country_missing",
    "country_conflict",

    # -------------------------------------------------------------------------
    # CONTRADICTION
    # -------------------------------------------------------------------------

    "strong_name_strong_address_conflict",
]


# =============================================================================
# VALIDATE FEATURE FILE
# =============================================================================

def validate_feature_file():

    print("=" * 80)
    print("VALIDATING FEATURE FILE")
    print("=" * 80)

    if not FEATURE_PATH.exists():

        print()
        print(
            "ERROR: Feature file not found:"
        )

        print(
            FEATURE_PATH
        )

        sys.exit(1)

    print()
    print(
        "Feature file:"
    )

    print(
        FEATURE_PATH
    )

    print()

    # Read header only.
    header = pd.read_csv(
        FEATURE_PATH,
        sep="\t",
        nrows=0,
    )

    columns = list(header.columns)

    required_columns = (
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label",
        ]
        + FEATURE_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in columns
    ]

    if missing_columns:

        print(
            "ERROR: Missing required columns:"
        )

        for column in missing_columns:
            print(
                f"  - {column}"
            )

        sys.exit(1)

    print(
        f"Total columns: {len(columns)}"
    )

    print(
        f"ML feature columns: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "Feature schema: PASS"
    )


# =============================================================================
# LOAD FEATURES
# =============================================================================

def load_features():

    print()
    print("=" * 80)
    print("LOADING TRAINING FEATURES")
    print("=" * 80)

    usecols = (
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label",
        ]
        + FEATURE_COLUMNS
    )

    print()
    print(
        "Loading 2,000,000 labeled pairs..."
    )

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t",
        usecols=usecols,
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "label": "int8",
        },
    )

    print(
        f"Rows loaded: "
        f"{len(df):,}"
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Label distribution:"
    )

    for label, count in label_counts.items():

        print(
            f"  Label {label}: "
            f"{count:,}"
        )

    # -------------------------------------------------------------------------
    # Missing-value check
    # -------------------------------------------------------------------------

    missing_features = (
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print()
    print(
        f"Missing feature values: "
        f"{missing_features:,}"
    )

    if missing_features > 0:

        print(
            "Filling missing feature values with 0."
        )

        df[FEATURE_COLUMNS] = (
            df[FEATURE_COLUMNS]
            .fillna(0)
        )

    return df


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(
    X_train,
    y_train,
):

    print()
    print("=" * 80)
    print("TRAINING HISTOGRAM GRADIENT BOOSTING MODEL")
    print("=" * 80)

    print()
    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    model = HistGradientBoostingClassifier(

        learning_rate=0.08,

        max_iter=250,

        max_leaf_nodes=31,

        min_samples_leaf=30,

        l2_regularization=1.0,

        random_state=RANDOM_STATE,
    )

    print()
    print(
        "Model configuration:"
    )

    print(
        "  learning_rate      = 0.08"
    )

    print(
        "  max_iter           = 250"
    )

    print(
        "  max_leaf_nodes     = 31"
    )

    print(
        "  min_samples_leaf   = 30"
    )

    print(
        "  l2_regularization  = 1.0"
    )

    print()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    print()
    print("=" * 80)
    print("MODEL DIAGNOSTIC EVALUATION")
    print("=" * 80)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype("int8")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_test,
        predictions,
        beta=0.5,
        zero_division=0,
    )

    print()
    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    print()
    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall:    {recall:.6f}"
    )

    print(
        f"F0.5:      {f05:.6f}"
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        f"  TN: {tn:,}"
    )

    print(
        f"  FP: {fp:,}"
    )

    print(
        f"  FN: {fn:,}"
    )

    print(
        f"  TP: {tp:,}"
    )

    # -------------------------------------------------------------------------
    # Probability diagnostics
    # -------------------------------------------------------------------------

    probability_series = pd.Series(
        probabilities
    )

    print()
    print(
        "Probability distribution:"
    )

    quantiles = (
        probability_series
        .quantile(
            [
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )

    for quantile, value in quantiles.items():

        print(
            f"  {quantile:.2f}: "
            f"{value:.6f}"
        )

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("MEMBER B - MATCHING MODEL TRAINING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_feature_file()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_features()

    # -------------------------------------------------------------------------
    # Separate features and labels
    # -------------------------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "label"
    ]

    # -------------------------------------------------------------------------
    # Train/test split
    #
    # IMPORTANT:
    # This is only a MODEL DIAGNOSTIC split.
    #
    # It is NOT the final competition validation.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING MODEL DIAGNOSTIC SPLIT")
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This random pair split is ONLY for "
        "training diagnostics."
    )

    print(
        "Final evaluation will use the fixed "
        "S1-level validation split."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print()
    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Diagnostic rows: "
        f"{len(X_test):,}"
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print("=" * 80)
    print("MODEL SAVED")
    print("=" * 80)

    print()
    print(
        MODEL_PATH
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print(
        f"Training pairs: "
        f"{len(df):,}"
    )

    print(
        f"Features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F0.5: "
        f"{metrics['f05']:.6f}"
    )

    print()
    print(
        "STATUS: MODEL TRAINING COMPLETE"
    )

    print()
    print(
        "Next stage: fixed S1-level validation."
    )


if __name__ == "__main__":
    main()