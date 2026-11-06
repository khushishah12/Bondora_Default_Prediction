# %% [markdown]
# # "ML ASSIGNMENT FINAL" approach, re-run with 2019 as the TEST year
#
# Source of the approach: Downloads/ML ASSIGNMENT FINAL.zip (ml_assignment.ipynb + PROJECT_REPORT.txt).
# This file is SEPARATE from bondora_pipeline.py and writes only to zip_approach_2019/.
#
# What is kept from the zip's approach (unchanged):
#   * target = default within 12 months of LoanDate (DefaultDate in [LoanDate, LoanDate+12m)),
#     eligibility rule, snapshot cutoff = max(ReportAsOfEOD)
#   * leakage list LEAKY_OR_ID, all remaining columns are predictors (incl. Bids*, income, credit scores)
#   * 5 engineered ratios (dti, amt_ratio, inc_month, loan_inc, freecash_neg)
#   * numeric: median impute; categorical: 'missing' + one-hot (max_categories=200); CatBoost native cats
#   * candidates: Dummy, LogisticRegression, LightGBM, XGBoost, CatBoost with the zip's hyper-parameters
#   * class-imbalance comparison, small tuning grid, per-model accuracy-optimal threshold chosen on VALIDATION
#   * selection = highest VALIDATION ROC-AUC, then refit on all pre-test data and test ONCE
#
# The ONE structural change (forced by the request): every split moves back one year, because
# 2019 can no longer be the validation set if it is the test set.
#     zip:   train 2013-2018 | validation 2019 | test 2020
#     here:  train 2013-2017 | validation 2018 | test 2019   (refit on 2013-2018 before testing)
#
# Small deviations, all flagged where they occur:
#   * the final refit uses the SAME hyper-parameters that were validated (the zip refit CatBoost with
#     1200 iterations although 1000 were validated);
#   * EDA plots and permutation importance are left out (they do not affect any number).

# %% Imports and config
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
import os
_cands = [os.environ.get("BONDORA_DATA"), BASE.parent / "data" / "LoanData_Bondora.csv.zip", BASE.parent / "data" / "LoanData_Bondora.csv",
          BASE / "LoanData_Bondora.csv.zip", BASE / "LoanData_Bondora.csv"]
_found = [Path(p) for p in _cands if p and Path(p).exists()]
if not _found:
    raise SystemExit("Raw data not found. Put LoanData_Bondora.csv (or .zip) in ../data/ or set BONDORA_DATA.")
DATA_ZIP = _found[0]
OUT = BASE.parent / "results" / "test2019_n12_full_run"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END, VAL_YEAR, TEST_YEAR = 2017, 2018, 2019      # train 2013..2017 | validation 2018 | test 2019
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# %% 1. Load
df = pd.read_csv(DATA_ZIP, low_memory=False)
tick(f"loaded {df.shape}")

# %% 2. Target (identical to the zip)
for c in ["LoanDate", "DefaultDate", "ListedOnUTC", "BiddingStartedOn", "LoanApplicationStartedDate",
          "ContractEndDate", "FirstPaymentDate", "MaturityDate_Original", "MaturityDate_Last",
          "LastPaymentOn", "StageActiveSince"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

CUTOFF = pd.to_datetime(df["ReportAsOfEOD"]).max()
df["window_end"] = df["LoanDate"] + pd.DateOffset(months=12)
valid_default_date = df["DefaultDate"].notna() & (df["DefaultDate"] >= df["LoanDate"])
default_in_window = valid_default_date & (df["DefaultDate"] < df["window_end"])
df["default_within_12m"] = default_in_window.astype(int)
window_complete = df["window_end"] <= CUTOFF
df["eligible"] = default_in_window | (window_complete & ~default_in_window)
print("cutoff:", CUTOFF.date(), "| eligible:", int(df["eligible"].sum()), "of", len(df))

# %% 3. Leakage audit and feature list (identical to the zip)
LEAKY_OR_ID = {
    "default_within_12m", "window_end", "eligible", "DefaultDate", "ReportAsOfEOD", "LoanId", "LoanNumber", "UserName",
    "LastPaymentOn", "DebtOccuredOn", "DebtOccuredOnForSecondary", "CurrentDebtDaysPrimary", "CurrentDebtDaysSecondary",
    "PrincipalOverdueBySchedule", "PlannedPrincipalPostDefault", "PlannedInterestPostDefault", "EAD1", "EAD2",
    "PrincipalRecovery", "InterestRecovery", "RecoveryStage", "StageActiveSince", "ModelVersion", "Status",
    "Restructured", "ActiveLateCategory", "WorseLateCategory", "PrincipalPaymentsMade",
    "InterestAndPenaltyPaymentsMade", "PrincipalWriteOffs", "InterestAndPenaltyWriteOffs", "PrincipalBalance",
    "InterestAndPenaltyBalance", "ActiveLateLastPaymentCategory", "GracePeriodStart", "GracePeriodEnd",
    "NextPaymentDate", "NextPaymentNr", "NrOfScheduledPayments", "ReScheduledOn", "PrincipalDebtServicingCost",
    "InterestAndPenaltyDebtServicingCost", "PlannedPrincipalTillDate", "PlannedInterestTillDate",
    "ActiveScheduleFirstPaymentReached",
    "ProbabilityOfDefault", "LossGivenDefault", "ExpectedLoss", "ExpectedReturn", "Rating", "EL_V0", "Rating_V0",
    "EL_V1", "Rating_V1", "Rating_V2",
    "LoanDate", "LoanApplicationStartedDate", "ListedOnUTC", "BiddingStartedOn", "ContractEndDate", "FirstPaymentDate",
    "MaturityDate_Original", "MaturityDate_Last", "DateOfBirth", "County", "City", "ApplicationSignedHour",
    "ApplicationSignedWeekday",
}
data = df[df["eligible"]].copy()
candidate_features = [c for c in data.columns if c not in LEAKY_OR_ID]
print("eligible loans:", len(data), "| candidate features:", len(candidate_features))

# %% 4. Chronological split (shifted back one year; see header)
data = data.sort_values("LoanDate").reset_index(drop=True)
yr = data["LoanDate"].dt.year
train = data[(yr >= 2013) & (yr <= TRAIN_END)].copy()
val = data[yr == VAL_YEAR].copy()
test = data[yr == TEST_YEAR].copy()
for nm, s in [("train", train), ("val", val), ("test", test)]:
    assert (s["window_end"] <= CUTOFF).all(), f"{nm} has an immature window"       # rule is a no-op before 2020
rows = [{"split": n, "start": s["LoanDate"].min().date(), "end": s["LoanDate"].max().date(), "n": len(s),
         "defaults": int(s["default_within_12m"].sum()), "default_rate": round(s["default_within_12m"].mean(), 4)}
        for n, s in [("Train 2013-2017", train), ("Validation 2018", val), ("TEST 2019", test)]]
split_table = pd.DataFrame(rows)
print(split_table.to_string(index=False))
split_table.to_csv(OUT / "split_summary.csv", index=False)

# %% 5. Cleaning, matrices, feature engineering (identical to the zip)
FORCE_NUMERIC = ["CreditScoreEsMicroL", "CreditScoreEsEquifaxRisk", "CreditScoreFiAsiakasTietoRiskGrade",
                 "CreditScoreEeMini", "NrOfDependants"]
for d in (train, val, test):
    for c in FORCE_NUMERIC:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
numeric_features = [c for c in candidate_features if pd.api.types.is_numeric_dtype(train[c])]
categorical_features = [c for c in candidate_features if c not in numeric_features]
print("numeric:", len(numeric_features), "| categorical:", categorical_features)


def make_matrix(d):
    X = pd.DataFrame(index=d.index)
    for c in numeric_features:
        X[c] = pd.to_numeric(d[c], errors="coerce")
    for c in categorical_features:
        X[c] = d[c].where(d[c].notna(), "missing").astype(str)
    return X


def add_features(d):
    d["dti"] = (d["MonthlyPayment"] / (d["IncomeTotal"] / 12.0)).replace([np.inf, -np.inf], np.nan)
    d["amt_ratio"] = (d["Amount"] / d["AppliedAmount"]).replace([np.inf, -np.inf], np.nan)
    d["inc_month"] = d["IncomeTotal"] / 12.0
    d["loan_inc"] = (d["Amount"] / d["IncomeTotal"]).replace([np.inf, -np.inf], np.nan)
    d["freecash_neg"] = (d["FreeCash"] < 0).astype(int)
    return d


Xtr, Xva, Xte = [add_features(make_matrix(d)) for d in (train, val, test)]
ytr, yva, yte = (d["default_within_12m"].values for d in (train, val, test))
numeric_features = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
categorical_features = [c for c in Xtr.columns if c not in numeric_features]
print("model matrices:", Xtr.shape, Xva.shape, Xte.shape, "| predictors:", len(numeric_features) + len(categorical_features))


def make_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=200, sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


# %% 6. Class imbalance check on validation (as in the zip)
print("\ntrain class balance:", pd.Series(ytr).value_counts(normalize=True).round(4).to_dict())
imb = []
for label, w in [("no_balance", None), ("balanced", "balanced")]:
    pipe = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)),
                     ("m", LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.03, num_leaves=31,
                                          class_weight=w, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1))])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xva)[:, 1]
    imb.append({"scheme": label, "val_accuracy": accuracy_score(yva, p >= .5), "val_recall": recall_score(yva, p >= .5),
                "val_f1": f1_score(yva, p >= .5), "val_roc_auc": roc_auc_score(yva, p)})
print(pd.DataFrame(imb).round(4).to_string(index=False))
tick("imbalance check done")

# %% 7. Baselines and the three boosted models (the zip's hyper-parameters), evaluated on VALIDATION (2018)
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
logreg = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)),
                   ("m", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE))]).fit(Xtr, ytr)
LGBM_KW = dict(n_estimators=1500, max_depth=8, learning_rate=0.008, num_leaves=63, reg_alpha=0.5, reg_lambda=15,
               min_child_samples=40, subsample=0.8, colsample_bytree=0.7, class_weight="balanced", n_jobs=-1,
               random_state=RANDOM_STATE, verbose=-1)
XGB_KW = dict(n_estimators=1500, max_depth=6, learning_rate=0.008, subsample=0.8, colsample_bytree=0.7, reg_lambda=15,
              reg_alpha=0.5, min_child_weight=10, scale_pos_weight=1.5, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
CB_KW = dict(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=8, random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)

lgbm = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)), ("m", LGBMClassifier(**LGBM_KW))]).fit(Xtr, ytr)
tick("lightgbm fitted")
xgb = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)), ("m", XGBClassifier(**XGB_KW))]).fit(Xtr, ytr)
tick("xgboost fitted")
cat_idx = [Xtr.columns.get_loc(c) for c in categorical_features]
catboost = CatBoostClassifier(cat_features=cat_idx, **CB_KW).fit(Xtr, ytr)
tick("catboost fitted")

# %% 8. Tuning grid on VALIDATION only (as in the zip; results are reported, the validated main configs are kept)
tuning = []
for params in [dict(n_estimators=800, max_depth=4, learning_rate=0.02, num_leaves=20, reg_lambda=4),
               dict(n_estimators=1500, max_depth=8, learning_rate=0.008, num_leaves=63, reg_lambda=15),
               dict(n_estimators=2000, max_depth=10, learning_rate=0.006, num_leaves=127, reg_lambda=25)]:
    kw = {**LGBM_KW, **params}
    pipe = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)), ("m", LGBMClassifier(**kw))]).fit(Xtr, ytr)
    p = pipe.predict_proba(Xva)[:, 1]
    tuning.append({"model": "LightGBM", **params, "val_accuracy": accuracy_score(yva, p >= .5), "val_roc_auc": roc_auc_score(yva, p)})
    tick(f"tuning lightgbm {params['max_depth']}")
for depth, lr, iters in [(6, 0.03, 1000), (8, 0.02, 1200)]:
    cb = CatBoostClassifier(iterations=iters, depth=depth, learning_rate=lr, l2_leaf_reg=8, random_seed=RANDOM_STATE,
                            verbose=False, allow_writing_files=False, cat_features=cat_idx).fit(Xtr, ytr)
    p = cb.predict_proba(Xva)[:, 1]
    tuning.append({"model": "CatBoost", "n_estimators": iters, "max_depth": depth, "learning_rate": lr,
                   "val_accuracy": accuracy_score(yva, p >= .5), "val_roc_auc": roc_auc_score(yva, p)})
    tick(f"tuning catboost depth {depth}")
tuning_df = pd.DataFrame(tuning).sort_values("val_roc_auc", ascending=False)
print("\nTuning grid (validation 2018):\n", tuning_df.round(4).to_string(index=False))
tuning_df.to_csv(OUT / "tuning_grid_validation.csv", index=False)


# %% 9. Validation comparison with accuracy-optimal thresholds (chosen on VALIDATION 2018)
def best_acc_threshold(prob, y):
    best = (-1, 0.5)
    for t in np.arange(0.05, 0.95, 0.005):
        a = accuracy_score(y, (prob >= t).astype(int))
        if a > best[0]:
            best = (a, t)
    return best[1]


cands = {"LogisticRegression": logreg.predict_proba(Xva)[:, 1], "LightGBM": lgbm.predict_proba(Xva)[:, 1],
         "XGBoost": xgb.predict_proba(Xva)[:, 1], "CatBoost": catboost.predict_proba(Xva)[:, 1]}
comp = [{"model": "Dummy (majority)", "threshold": np.nan, "val_accuracy": accuracy_score(yva, dummy.predict(Xva)),
         "val_precision": 0.0, "val_recall": 0.0, "val_f1": 0.0, "val_roc_auc": 0.5, "val_pr_auc": np.nan,
         "val_balanced_acc": 0.5}]
for name, prob in cands.items():
    t = best_acc_threshold(prob, yva)
    pred = (prob >= t).astype(int)
    comp.append({"model": name, "threshold": round(t, 3), "val_accuracy": accuracy_score(yva, pred),
                 "val_precision": precision_score(yva, pred, zero_division=0), "val_recall": recall_score(yva, pred),
                 "val_f1": f1_score(yva, pred), "val_roc_auc": roc_auc_score(yva, prob),
                 "val_pr_auc": average_precision_score(yva, prob), "val_balanced_acc": balanced_accuracy_score(yva, pred)})
comparison_df = pd.DataFrame(comp).sort_values("val_roc_auc", ascending=False)
print("\nVALIDATION (2018) model comparison:\n", comparison_df.round(4).to_string(index=False))
comparison_df.to_csv(OUT / "validation_model_comparison_2018.csv", index=False)

# %% 10. Final selection by validation ROC-AUC, refit on 2013-2018, threshold fixed on 2018
pool = comparison_df[comparison_df["model"] != "Dummy (majority)"]
best_name = pool.sort_values(["val_roc_auc", "val_accuracy"], ascending=False).iloc[0]["model"]
sel_threshold = float(comparison_df.loc[comparison_df["model"] == best_name, "threshold"].iloc[0])
print("\nSelected on validation ROC-AUC:", best_name, "| validation-tuned threshold:", sel_threshold)

X_final = pd.concat([Xtr, Xva], ignore_index=True)
y_final = np.concatenate([ytr, yva])
if best_name == "CatBoost":
    final_model = CatBoostClassifier(cat_features=[X_final.columns.get_loc(c) for c in categorical_features], **CB_KW)
elif best_name == "LightGBM":
    final_model = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)), ("m", LGBMClassifier(**LGBM_KW))])
elif best_name == "XGBoost":
    final_model = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)), ("m", XGBClassifier(**XGB_KW))])
else:
    final_model = Pipeline([("pre", make_preprocessor(numeric_features, categorical_features)),
                            ("m", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE))])
final_model.fit(X_final, y_final)
tick(f"final {best_name} refit on {len(X_final)} loans (2013-2018)")

# %% 11. FINAL TEST on 2019 (touched once)
test_prob = final_model.predict_proba(Xte)[:, 1]


def full_metrics(y, p, pred):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "brier": brier_score_loss(y, p)}


m_thr = full_metrics(yte, test_prob, (test_prob >= sel_threshold).astype(int))
m_half = full_metrics(yte, test_prob, (test_prob >= 0.5).astype(int))
majority = (yte == 0).mean()
print(f"\nFINAL TEST 2019  (model={best_name}, threshold={sel_threshold:.3f}, n={len(yte)}, base rate={yte.mean():.4f})")
print(pd.DataFrame({"at_val_threshold": m_thr, "at_0.5": m_half}).round(4).to_string())
print(f"majority-class accuracy on test: {majority:.4f}")
print("confusion matrix at validation threshold (rows=true, cols=pred):\n",
      confusion_matrix(yte, (test_prob >= sel_threshold).astype(int)))

# per-country view, to compare with bondora_pipeline.py (same 2019 loans)
rows = []
for k in ["ALL"] + sorted(test["Country"].unique().tolist()):
    msk = np.ones(len(test), bool) if k == "ALL" else (test["Country"].values == k)
    y_k, p_k = yte[msk], test_prob[msk]
    if len(np.unique(y_k)) < 2:
        continue
    rows.append({"group": k, "n": int(msk.sum()), "base_rate": round(y_k.mean(), 4), "roc_auc": round(roc_auc_score(y_k, p_k), 4),
                 "pr_auc": round(average_precision_score(y_k, p_k), 4),
                 "lift": round(average_precision_score(y_k, p_k) / y_k.mean(), 3), "brier": round(brier_score_loss(y_k, p_k), 4),
                 "mean_pred": round(p_k.mean(), 4)})
by_country = pd.DataFrame(rows)
print("\nTEST 2019 by country:\n", by_country.to_string(index=False))

pd.DataFrame([{"model": best_name, "threshold": sel_threshold, **m_thr, "majority_baseline_accuracy": majority,
               "test_period": "2019", "train_period": "2013-2018 (refit)", "validation_period": "2018"}]
             ).to_csv(OUT / "final_test_results_2019.csv", index=False)
by_country.to_csv(OUT / "final_test_by_country_2019.csv", index=False)
pd.DataFrame({"y_true": yte, "prob_default": test_prob, "pred_default": (test_prob >= sel_threshold).astype(int),
              "country": test["Country"].values}).to_csv(OUT / "final_test_predictions_2019.csv", index=False)

# %% 12. Native importance of the final model (legitimacy check of the top drivers)
if best_name == "CatBoost":
    imp = pd.Series(final_model.get_feature_importance(), index=X_final.columns)
else:
    imp = pd.Series(final_model.named_steps["m"].feature_importances_, index=final_model.named_steps["pre"].get_feature_names_out())
print("\nTop 15 native importances:\n", imp.sort_values(ascending=False).head(15).round(2).to_string())
imp.sort_values(ascending=False).to_csv(OUT / "feature_importance.csv", header=["importance"])
tick("done")
print("\nZip-approach 2019 test done.")
