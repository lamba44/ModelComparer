# utils/data_preprocessing.py
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


def _detect_and_drop_leaky_columns(
    X_train,
    X_test,
    y_train,
    uniq_thresh=0.8,
    missing_diff_thresh=0.9,
    corr_thresh=0.95,
    zero_frac_diff_thresh=0.9,
):
    """
    Heuristic-based detector to remove columns that strongly indicate leakage:
      - ID-like columns (>= uniq_thresh fraction unique values)
      - Columns whose presence/non-null rate differs by > missing_diff_thresh between classes
      - Numeric columns with abs(corr) >= corr_thresh with the target
      - Numeric columns where the fraction of zeros (or a sentinel value) differs by > zero_frac_diff_thresh between classes
      - Numeric columns that are constant within any class (nunique per class == 1)
    Returns: X_train_clean, X_test_clean, list_of_dropped_columns
    """
    dropped = {}
    n_rows = len(X_train)
    classes = np.unique(y_train)

    # 1) ID-like columns (many uniques)
    for c in X_train.columns:
        try:
            uniq_frac = X_train[c].nunique(dropna=False) / float(n_rows)
            if uniq_frac >= uniq_thresh:
                dropped[c] = "id_like (high_unique_frac=%.3f)" % uniq_frac
        except Exception:
            continue

    # 2) Class-dependent presence (missingness)
    if len(classes) > 1:
        for c in X_train.columns:
            if c in dropped:
                continue
            try:
                frac_nonnull = X_train.groupby(y_train)[c].apply(
                    lambda s: s.notna().mean()
                )
                if (frac_nonnull.max() - frac_nonnull.min()) > missing_diff_thresh:
                    dropped[c] = "presence_diff (nonnull_frac_diff=%.3f)" % (
                        frac_nonnull.max() - frac_nonnull.min()
                    )
            except Exception:
                continue

    # 3) Numeric correlation with target
    try:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        numeric_cols = []
    if len(numeric_cols) > 0:
        # try to coerce y to numeric for correlation calculation
        ynum = pd.to_numeric(y_train, errors="coerce")
        if not np.all(np.isnan(ynum)):
            for c in numeric_cols:
                if c in dropped:
                    continue
                try:
                    if X_train[c].nunique() <= 1:
                        # constant column overall — treat as leaky/useless
                        dropped[c] = "constant_overall"
                        continue
                    corr = abs(X_train[c].corr(ynum))
                    if pd.notna(corr) and corr >= corr_thresh:
                        dropped[c] = "high_corr (corr=%.3f)" % corr
                except Exception:
                    continue

    # 4) Numeric sentinel/zero-fraction differences and per-class constant detection
    #    (This catches columns like salary_lpa where unplaced rows have salary==0)
    for c in numeric_cols:
        if c in dropped:
            continue
        try:
            per_class_nunique = X_train.groupby(y_train)[c].nunique(dropna=False)
            # If any class has exactly 1 unique value for that column and others differ -> suspicious
            if (
                per_class_nunique == 1
            ).any() and per_class_nunique.max() != per_class_nunique.min():
                dropped[c] = (
                    "constant_within_class (per_class_nunique=%s)"
                    % per_class_nunique.to_dict()
                )
                continue

            # Fraction of zeros (or sentinel equal to the column's min)
            # Using zero as common sentinel; also test fraction equal to the class-min
            frac_zero_by_class = X_train.groupby(y_train)[c].apply(
                lambda s: (s == 0).mean()
            )
            if (
                frac_zero_by_class.max() - frac_zero_by_class.min()
            ) >= zero_frac_diff_thresh:
                dropped[c] = "zero_fraction_diff (zero_frac_diff=%.3f)" % (
                    frac_zero_by_class.max() - frac_zero_by_class.min()
                )
                continue

            # Another sentinel: if one class has values all equal to min (e.g., 0.0) and other class not
            min_eq_by_class = X_train.groupby(y_train)[c].apply(
                lambda s: (s == s.min()).mean()
            )
            if (min_eq_by_class.max() - min_eq_by_class.min()) >= zero_frac_diff_thresh:
                dropped[c] = "min_value_presence_diff (min_fraction_diff=%.3f)" % (
                    min_eq_by_class.max() - min_eq_by_class.min()
                )
                continue
        except Exception:
            continue

    # 5) Categorical columns perfectly mapping to target (already strong leak)
    try:
        cat_cols = X_train.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()
    except Exception:
        cat_cols = []
    for c in cat_cols:
        if c in dropped:
            continue
        try:
            grp = (
                X_train.groupby(c)[y_train.name].nunique(dropna=False)
                if y_train.name in X_train.columns
                else X_train.groupby(c)[y_train.index].apply(
                    lambda idx: y_train.loc[idx].nunique()
                )
            )
            # The above is fragile if y_train.name not aligned; safer approach:
            grp = X_train.groupby(c)[c].apply(
                lambda s: 1
            )  # dummy to use structure below
        except Exception:
            grp = None
        try:
            # Simpler: check mapping by pivoting: for each category value, how many unique target labels appear?
            mapping = (
                X_train[[c]]
                .join(pd.Series(y_train.values, index=X_train.index, name="_target"))
                .groupby(c)["_target"]
                .nunique(dropna=False)
            )
            if mapping.max() == 1:
                dropped[c] = "cat_value_maps_to_single_target"
        except Exception:
            continue

    # finalize list
    dropped_list = sorted(dropped.keys())
    if dropped_list:
        print("Auto-dropping suspected leaky/ID columns and reasons:")
        for col in dropped_list:
            print("  - %s: %s" % (col, dropped[col]))
        X_train = X_train.drop(columns=dropped_list, errors="ignore")
        X_test = X_test.drop(columns=dropped_list, errors="ignore")
    else:
        print("No obvious leaky/ID columns detected automatically.")

    return X_train, X_test, dropped_list


def load_and_preprocess(
    csv_path: str,
    target_column: str,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
    # new knobs (optional)
    high_cardinality_threshold: int = 1000,
    svd_components: int = 200,
    # leakage-detection knobs
    uniq_thresh: float = 0.8,
    missing_diff_thresh: float = 0.9,
    corr_thresh: float = 0.95,
    zero_frac_diff_thresh: float = 0.9,
):
    """
    Improved preprocessing that avoids exploding one-hot encodings for high-cardinality categorical features
    and detects/drops obvious leakage columns (IDs, post-outcome features like salary).

    Strategy:
     - Read CSV and split train/test early (to avoid leakage when computing encodings).
     - Auto-detect and drop ID-like / leaky columns using heuristics (improved to detect sentinel patterns).
     - For remaining categorical columns:
         * If n_unique > high_cardinality_threshold  -> apply frequency (count) encoding (treated as numeric).
         * Else -> apply OneHotEncoder(sparse) followed by TruncatedSVD to compress to `svd_components` dimensions.
     - Numeric columns are StandardScaled.
    """

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split early to compute encodings on training only (avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=(y if task == "classification" else None),
    )

    # Detect and drop obvious leaky / ID columns (this modifies X_train / X_test)
    X_train, X_test, dropped_cols = _detect_and_drop_leaky_columns(
        X_train,
        X_test,
        y_train,
        uniq_thresh=uniq_thresh,
        missing_diff_thresh=missing_diff_thresh,
        corr_thresh=corr_thresh,
        zero_frac_diff_thresh=zero_frac_diff_thresh,
    )

    # Recompute column type lists after dropping
    numeric_cols_all = X_train.select_dtypes(
        include=["int64", "float64", "number"]
    ).columns.tolist()
    categorical_cols_all = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Decide which categorical columns are high-cardinality
    high_card_cols = []
    normal_cat_cols = []
    for c in categorical_cols_all:
        try:
            nuniq = X_train[c].nunique(dropna=False)
        except Exception:
            nuniq = 0
        if nuniq > high_cardinality_threshold or nuniq > (0.05 * X_train.shape[0]):
            high_card_cols.append(c)
        else:
            normal_cat_cols.append(c)

    # Frequency / count encode high-cardinality categorical columns
    for c in high_card_cols:
        counts = X_train[c].value_counts().to_dict()
        X_train[c] = X_train[c].map(counts).fillna(0).astype(float)
        X_test[c] = X_test[c].map(counts).fillna(0).astype(float)
        numeric_cols_all.append(c)

    # Build pipelines
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    transformers = []
    if numeric_cols_all:
        transformers.append(("num", numeric_pipeline, numeric_cols_all))

    if normal_cat_cols:
        total_unique = sum(X_train[c].nunique() for c in normal_cat_cols)
        n_comps = min(svd_components, max(2, total_unique - 1))
        if total_unique <= n_comps:
            cat_pipeline = Pipeline(
                [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]
            )
        else:
            cat_pipeline = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                    (
                        "svd",
                        TruncatedSVD(n_components=n_comps, random_state=random_state),
                    ),
                ]
            )
        transformers.append(("cat", cat_pipeline, normal_cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0
    )

    # Fit/transform on training set only
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # If result is sparse, convert to dense only when safe (small feature count).
    if sparse.issparse(X_train_trans):
        n_features = X_train_trans.shape[1]
        # Heuristic: only convert to dense if feature dimension is reasonably small to avoid MemoryError.
        if n_features <= 20000:
            try:
                X_train_trans = X_train_trans.toarray()
                X_test_trans = X_test_trans.toarray()
            except MemoryError:
                # leave sparse if conversion fails
                pass

    if dropped_cols:
        print(
            "Columns dropped during preprocessing for being leaky/ID-like:",
            dropped_cols,
        )

    return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor
