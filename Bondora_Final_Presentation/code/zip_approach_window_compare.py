# %% [markdown]
# # "ML ASSIGNMENT FINAL" approach across WINDOW WIDTHS, for two test years
#
# Usage (from the code/ folder):  python zip_approach_window_compare.py --test-year 2019|2020 [--windows 6,12,18,24]
#
# Separate from bondora_pipeline.py and zip_approach_test2019.py; writes only to zip_window_compare/.
#
# Protocol (the zip's, shifted so the test year is the one asked for):
#     test year T:  train 2013..T-2 | validation T-1 | test T
#     select the model by validation ROC-AUC, take the accuracy-optimal threshold from validation,
#     refit on 2013..T-1 with the same hyper-parameters, score the test year ONCE.
#     (T=2020 is the zip's own protocol, T=2019 is the one-year-earlier version.)
#
# Target for window N months (the zip's definition): default if LoanDate <= DefaultDate < LoanDate + N months.
#
# Deviations from the zip, all deliberate:
#   1. ELIGIBILITY: the zip also keeps loans whose window is NOT finished if they already defaulted (it drops
#      their non-defaulting peers), which biases the default rate upward and gets worse as N grows. Here a loan
#      is used only if its full N-month window ended by the snapshot, for every split and every N.
#   2. The tuning grid and the imbalance comparison are skipped (neither feeds the final model in the zip).
#   3. Extra reference columns: Bondora's own ProbabilityOfDefault on the SAME test loans, and the
#      majority-class accuracy. Metrics are NOT comparable across N (different base rates and test loans), so
#      judge each N against its own references.

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

ap = argparse.ArgumentParser()
ap.add_argument("--test-year", type=int, required=True)
ap.add_argument("--windows", default="6,12,18,24")
args = ap.parse_args()
T = args.test_year
WINDOWS = [int(x) for x in args.windows.split(",")]
MIN_TEST, MIN_VAL, MIN_POS = 500, 1000, 50          # below this a run is reported as not feasible / underpowered

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
import os
_cands = [os.environ.get("BONDORA_DATA"), BASE.parent / "data" / "LoanData_Bondora.csv.zip", BASE.parent / "data" / "LoanData_Bondora.csv",
          BASE / "LoanData_Bondora.csv.zip", BASE / "LoanData_Bondora.csv"]
_found = [Path(p) for p in _cands if p and Path(p).exists()]
if not _found:
    raise SystemExit("Raw data not found. Put LoanData_Bondora.csv (or .zip) in ../data/ or set BONDORA_DATA.")
DATA_ZIP = _found[0]
OUT = BASE.parent / "results"          # packaged layout: code/ and results/ are siblings
OUT.mkdir(exist_ok=True)
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# %% Load once
df = pd.read_csv(DATA_ZIP, low_memory=False)
for c in ["LoanDate", "DefaultDate"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
CUTOFF = pd.to_datetime(df["ReportAsOfEOD"]).max()
tick(f"loaded {df.shape}; snapshot {CUTOFF.date()}; test year {T}; windows {WINDOWS}")

# %% Leakage list and features: identical to the zip
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
FORCE_NUMERIC = ["CreditScoreEsMicroL", "CreditScoreEsEquifaxRisk", "CreditScoreFiAsiakasTietoRiskGrade",
                 "CreditScoreEeMini", "NrOfDependants"]
candidate_features = [c for c in df.columns if c not in LEAKY_OR_ID and c not in
                      ("window_end", "eligible", "default_within_12m")]
for c in FORCE_NUMERIC:
    df[c] = pd.to_numeric(df[c], errors="coerce")
numeric_features0 = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]
categorical_features0 = [c for c in candidate_features if c not in numeric_features0]


def make_matrix(d):
    X = pd.DataFrame(index=d.index)
    for c in numeric_features0:
        X[c] = pd.to_numeric(d[c], errors="coerce")
    for c in categorical_features0:
        X[c] = d[c].where(d[c].notna(), "missing").astype(str)
    X["dti"] = (X["MonthlyPayment"] / (X["IncomeTotal"] / 12.0)).replace([np.inf, -np.inf], np.nan)
    X["amt_ratio"] = (X["Amount"] / X["AppliedAmount"]).replace([np.inf, -np.inf], np.nan)
    X["inc_month"] = X["IncomeTotal"] / 12.0
    X["loan_inc"] = (X["Amount"] / X["IncomeTotal"]).replace([np.inf, -np.inf], np.nan)
    X["freecash_neg"] = (X["FreeCash"] < 0).astype(int)
    return X


XALL = make_matrix(df)
cat_cols = categorical_features0
num_cols = [c for c in XALL.columns if c not in cat_cols]
tick(f"feature matrix {XALL.shape}: {len(num_cols)} numeric + {len(cat_cols)} categorical {cat_cols}")


def make_preprocessor():
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=200, sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def build(name):
    if name == "LogisticRegression":
        return Pipeline([("pre", make_preprocessor()), ("m", LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE))])
    if name == "LightGBM":
        return Pipeline([("pre", make_preprocessor()), ("m", LGBMClassifier(
            n_estimators=1500, max_depth=8, learning_rate=0.008, num_leaves=63, reg_alpha=0.5, reg_lambda=15,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.7, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1))])
    if name == "XGBoost":
        return Pipeline([("pre", make_preprocessor()), ("m", XGBClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.008, subsample=0.8, colsample_bytree=0.7, reg_lambda=15,
            reg_alpha=0.5, min_child_weight=10, scale_pos_weight=1.5, n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0))])
    if name == "CatBoost":
        return CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=8,
                                  random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
                                  cat_features=[XALL.columns.get_loc(c) for c in cat_cols])
    raise ValueError(name)


def best_acc_threshold(prob, y):
    best = (-1, 0.5)
    for t in np.arange(0.05, 0.95, 0.005):
        a = accuracy_score(y, (prob >= t).astype(int))
        if a > best[0]:
            best = (a, t)
    return best[1]


def metrics(y, p, thr):
    pr = average_precision_score(y, p)
    pred, half = (p >= thr).astype(int), (p >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": pr, "lift": pr / y.mean(), "brier": brier_score_loss(y, p),
            "mean_pred": p.mean(), "acc_at_val_thr": accuracy_score(y, pred),
            "precision_at_val_thr": precision_score(y, pred, zero_division=0), "recall_at_val_thr": recall_score(y, pred),
            "f1_at_val_thr": f1_score(y, pred), "bal_acc_at_val_thr": balanced_accuracy_score(y, pred),
            "acc_at_0.5": accuracy_score(y, half), "recall_at_0.5": recall_score(y, half), "f1_at_0.5": f1_score(y, half)}


# %% One run = one window width N
def run_one(N):
    end = df["LoanDate"] + pd.DateOffset(months=N)
    valid_def = df["DefaultDate"].notna() & (df["DefaultDate"] >= df["LoanDate"])
    y_all = (valid_def & (df["DefaultDate"] < end)).astype(int)
    mature = end <= CUTOFF                                             # deviation 1: full window observed, no exceptions
    yr = df["LoanDate"].dt.year
    sets = {"train": mature & (yr >= 2013) & (yr <= T - 2), "val": mature & (yr == T - 1), "test": mature & (yr == T)}
    info = {"test_year": T, "window_months": N, "last_mature_issue_date": df.loc[mature, "LoanDate"].max().date()}
    for k, m in sets.items():
        info[f"n_{k}"] = int(m.sum())
        info[f"rate_{k}"] = round(float(y_all[m].mean()), 4) if m.sum() else np.nan
    t_lo = df.loc[sets["test"], "LoanDate"]
    info["test_span"] = f"{t_lo.min().date()}..{t_lo.max().date()}" if len(t_lo) else "-"
    if (info["n_test"] < MIN_TEST or info["n_val"] < MIN_VAL or info["n_train"] < MIN_VAL
            or y_all[sets["test"]].sum() < MIN_POS or y_all[sets["val"]].sum() < MIN_POS):
        info["status"] = "NOT FEASIBLE (too few mature loans in validation/test)"
        return info
    info["status"] = "ok" + (" (test set is a partial year)" if info["n_test"] < 0.6 * (yr == T).sum() else "")

    Xtr, Xva, Xte = (XALL[sets[k]] for k in ("train", "val", "test"))
    ytr, yva, yte = (y_all[sets[k]].values for k in ("train", "val", "test"))
    # model selection on validation
    val_prob, val_thr, fitted = {}, {}, {}
    for name in ["LogisticRegression", "LightGBM", "XGBoost", "CatBoost"]:
        m = build(name).fit(Xtr, ytr)
        val_prob[name] = m.predict_proba(Xva)[:, 1]
        val_thr[name] = best_acc_threshold(val_prob[name], yva)
        info[f"val_roc_{name}"] = round(roc_auc_score(yva, val_prob[name]), 4)
    pool = sorted(val_prob, key=lambda n: (info[f"val_roc_{n}"], accuracy_score(yva, val_prob[n] >= val_thr[n])), reverse=True)
    best = pool[0]
    info["selected"], info["val_threshold"] = best, round(val_thr[best], 3)
    info["val_acc_at_thr"] = round(accuracy_score(yva, val_prob[best] >= val_thr[best]), 4)
    info["val_majority_acc"] = round(1 - yva.mean(), 4)
    # refit on train+validation with the same hyper-parameters, test once
    final = build(best).fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
    p = final.predict_proba(Xte)[:, 1]
    info.update({k: round(float(v), 4) for k, v in metrics(yte, p, val_thr[best]).items()})
    info["test_majority_acc"] = round(1 - yte.mean(), 4)
    # reference: Bondora's own score on the SAME test loans (never used as a feature)
    pd_ = df.loc[sets["test"], "ProbabilityOfDefault"].values
    ok = ~np.isnan(pd_)
    info["bondora_pd_roc_auc"] = round(roc_auc_score(yte[ok], pd_[ok]), 4)
    info["bondora_pd_pr_auc"] = round(average_precision_score(yte[ok], pd_[ok]), 4)
    cty = df.loc[sets["test"], "Country"].values
    for k in ["EE", "ES", "FI"]:
        mk = cty == k
        if mk.sum() >= 300 and 0 < yte[mk].mean() < 1:
            info[f"roc_{k}"] = round(roc_auc_score(yte[mk], p[mk]), 4)
            okk = ok & mk
            info[f"bondora_roc_{k}"] = round(roc_auc_score(yte[okk], pd_[okk]), 4)
    return info


rows = []
for N in WINDOWS:
    tick(f"=== window N={N} months, test year {T} ===")
    r = run_one(N)
    rows.append(r)
    print({k: r[k] for k in r if k in ("window_months", "status", "n_train", "n_val", "n_test", "rate_test", "selected",
                                       "roc_auc", "pr_auc", "acc_at_val_thr", "test_majority_acc", "bondora_pd_roc_auc")}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"window_compare_test{T}.csv", index=False)
tick("done")
print(f"\nWindow comparison for test year {T} done.")
