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
    task="classification",
    uniq_thresh=0.8,
    missing_diff_thresh=0.9,
    corr_thresh=0.95,
    zero_frac_diff_thresh=0.9,
    min_unique_values_for_classification=50,
):

    dropped = {}
    n_rows = len(X_train)

    unique_y = np.unique(y_train)
    consider_as_classification = False
    if (
        task == "classification"
        and len(unique_y) <= min_unique_values_for_classification
    ):
        consider_as_classification = True

    for c in X_train.columns:
        try:
            uniq_frac = X_train[c].nunique(dropna=False) / float(n_rows)
            if uniq_frac >= uniq_thresh:
                dropped[c] = "id_like (high_unique_frac=%.3f)" % uniq_frac
        except Exception:
            continue

    if consider_as_classification:
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

    try:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        numeric_cols = []
    if len(numeric_cols) > 0:
        ynum = pd.to_numeric(y_train, errors="coerce")
        if not np.all(np.isnan(ynum)):
            for c in numeric_cols:
                if c in dropped:
                    continue
                try:
                    if X_train[c].nunique() <= 1:
                        dropped[c] = "constant_overall"
                        continue
                    corr = abs(X_train[c].corr(ynum))
                    if pd.notna(corr) and corr >= corr_thresh:
                        dropped[c] = "high_corr (corr=%.3f)" % corr
                except Exception:
                    continue

    if consider_as_classification:
        for c in numeric_cols:
            if c in dropped:
                continue
            try:
                per_class_nunique = X_train.groupby(y_train)[c].nunique(dropna=False)
                if (
                    per_class_nunique == 1
                ).any() and per_class_nunique.max() != per_class_nunique.min():
                    dropped[c] = (
                        "constant_within_class (per_class_nunique=%s)"
                        % per_class_nunique.to_dict()
                    )
                    continue

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

                min_eq_by_class = X_train.groupby(y_train)[c].apply(
                    lambda s: (s == s.min()).mean()
                )
                if (
                    min_eq_by_class.max() - min_eq_by_class.min()
                ) >= zero_frac_diff_thresh:
                    dropped[c] = "min_value_presence_diff (min_fraction_diff=%.3f)" % (
                        min_eq_by_class.max() - min_eq_by_class.min()
                    )
                    continue
            except Exception:
                continue

    try:
        cat_cols = X_train.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()
    except Exception:
        cat_cols = []
    if consider_as_classification:
        for c in cat_cols:
            if c in dropped:
                continue
            try:
                mapping = (
                    X_train[[c]]
                    .join(
                        pd.Series(y_train.values, index=X_train.index, name="_target")
                    )
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
    min_unique_values_for_classification: int = 50,
):

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=(y if task == "classification" else None),
    )

    X_train, X_test, dropped_cols = _detect_and_drop_leaky_columns(
        X_train,
        X_test,
        y_train,
        task=task,
        uniq_thresh=uniq_thresh,
        missing_diff_thresh=missing_diff_thresh,
        corr_thresh=corr_thresh,
        zero_frac_diff_thresh=zero_frac_diff_thresh,
        min_unique_values_for_classification=min_unique_values_for_classification,
    )

    numeric_cols_all = X_train.select_dtypes(
        include=["int64", "float64", "number"]
    ).columns.tolist()
    categorical_cols_all = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

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

    for c in high_card_cols:
        counts = X_train[c].value_counts().to_dict()
        X_train[c] = X_train[c].map(counts).fillna(0).astype(float)
        X_test[c] = X_test[c].map(counts).fillna(0).astype(float)
        numeric_cols_all.append(c)

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

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    if sparse.issparse(X_train_trans):
        n_features = X_train_trans.shape[1]
        if n_features <= 20000:
            try:
                X_train_trans = X_train_trans.toarray()
                X_test_trans = X_test_trans.toarray()
            except MemoryError:
                pass

    if dropped_cols:
        print(
            "Columns dropped during preprocessing for being leaky/ID-like:",
            dropped_cols,
        )

    return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor
