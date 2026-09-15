# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# ===== 1) 输入文件 =====
INPUT_CSV = "merged_calibrated.csv"

# ===== 2) 输出文件名 =====
OUT_HIGH_NARROW = "cand_high_narrow.csv"
OUT_HIGH_WIDE = "cand_high_wide.csv"
OUT_HIGH_UNCERTAIN = "cand_high_uncertain.csv"
OUT_BALANCED = "cand_balanced_for_review.csv"
OUT_STATS = "cand_summary_stats.txt"

# ===== 3) 读取 =====
df = pd.read_csv(INPUT_CSV, low_memory=False)

# ===== 4) 统一列名检查 =====
required_cols = [
    "standard_inchi_key",
    "uniprot_id",
    "affinity_type",
    "y_true",
    "y_pred_calibrated",
    "err_bound",
    "interval_lo_calibrated",
    "interval_hi_calibrated",
    "abs_error_calibrated",
    "prob_within_tol_0p2_calibrated",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少必要列: {missing}")

# ===== 5) 只保留论文表可能用到的列 =====
keep_cols = [
    "standard_inchi_key",
    "chembl_molecule_chembl_id",
    "uniprot_id",
    "chembl_target_chembl_id",
    "affinity_type",
    "y_true",
    "y_pred_calibrated",
    "err_bound",
    "interval_lo_calibrated",
    "interval_hi_calibrated",
    "abs_error_calibrated",
    "prob_within_tol_0p2_calibrated",
    "accepted",
    "prob_strong_binder",
    "smiles",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()

# ===== 6) 简单清洗 =====
for c in ["y_true", "y_pred_calibrated", "err_bound",
          "interval_lo_calibrated", "interval_hi_calibrated",
          "abs_error_calibrated", "prob_within_tol_0p2_calibrated"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[
    "standard_inchi_key",
    "uniprot_id",
    "affinity_type",
    "y_true",
    "y_pred_calibrated",
    "err_bound",
    "interval_lo_calibrated",
    "interval_hi_calibrated",
    "abs_error_calibrated"
]).copy()

# UniProt 如果有 "P04049|L7RRS6" 这种，只取第一个，便于后续人工查文献
df["uniprot_main"] = df["uniprot_id"].astype(str).str.split("|").str[0]

# 去重：同一个 InChIKey + UniProt + endpoint，只保留误差最小的一条
df = df.sort_values(["abs_error_calibrated", "err_bound"], ascending=[True, True])
df = df.drop_duplicates(subset=["standard_inchi_key", "uniprot_main", "affinity_type"], keep="first").copy()

# ===== 7) 自动阈值 =====
# 当前这份文件大概率只有 pIC50，但代码写成通用的
score_q90 = df["y_pred_calibrated"].quantile(0.90)   # 高分阈值
score_q95 = df["y_pred_calibrated"].quantile(0.95)   # 更高分阈值
narrow_q25 = df["err_bound"].quantile(0.25)          # 窄区间阈值
wide_q75 = df["err_bound"].quantile(0.75)            # 宽区间阈值

# 你可以按需要改
HIGH_SCORE = max(8.0, score_q90)      # 至少 8.0 pX
VERY_HIGH_SCORE = max(8.5, score_q95)
GOOD_ERROR = 0.30                     # 预测比较接近真实
MODERATE_ERROR = 0.60
BAD_ERROR = 1.00

# ===== 8) 打标签 =====
df["score_level"] = np.where(df["y_pred_calibrated"] >= VERY_HIGH_SCORE, "very_high",
                      np.where(df["y_pred_calibrated"] >= HIGH_SCORE, "high", "other"))

df["interval_level"] = np.where(df["err_bound"] <= narrow_q25, "narrow",
                         np.where(df["err_bound"] >= wide_q75, "wide", "mid"))

df["error_level"] = np.where(df["abs_error_calibrated"] <= GOOD_ERROR, "good",
                      np.where(df["abs_error_calibrated"] <= MODERATE_ERROR, "moderate",
                      np.where(df["abs_error_calibrated"] >= BAD_ERROR, "bad", "mid")))

# ===== 9) 三类论文最有用的候选 =====
# A. 高分 + 区间窄 + 误差小
cand_high_narrow = df[
    (df["y_pred_calibrated"] >= HIGH_SCORE) &
    (df["err_bound"] <= narrow_q25) &
    (df["abs_error_calibrated"] <= GOOD_ERROR)
].copy()

# B. 高分 + 区间宽 + 误差仍小（最能体现“高分不等于高置信”）
cand_high_wide = df[
    (df["y_pred_calibrated"] >= HIGH_SCORE) &
    (df["err_bound"] >= wide_q75) &
    (df["abs_error_calibrated"] <= GOOD_ERROR)
].copy()

# C. 高分 + 区间宽 + 误差较大（反例，补充里可用，不建议正文主表用太多）
cand_high_uncertain = df[
    (df["y_pred_calibrated"] >= HIGH_SCORE) &
    (df["err_bound"] >= wide_q75) &
    (df["abs_error_calibrated"] >= MODERATE_ERROR)
].copy()

# 排序规则
cand_high_narrow = cand_high_narrow.sort_values(
    ["y_pred_calibrated", "abs_error_calibrated", "err_bound"],
    ascending=[False, True, True]
)
cand_high_wide = cand_high_wide.sort_values(
    ["y_pred_calibrated", "abs_error_calibrated", "err_bound"],
    ascending=[False, True, False]
)
cand_high_uncertain = cand_high_uncertain.sort_values(
    ["y_pred_calibrated", "abs_error_calibrated", "err_bound"],
    ascending=[False, False, False]
)

# ===== 10) 每类先截取一些，供你发我继续筛 =====
TOP_N_NARROW = 20
TOP_N_WIDE = 20
TOP_N_UNCERTAIN = 20

cand_high_narrow = cand_high_narrow.head(TOP_N_NARROW).copy()
cand_high_wide = cand_high_wide.head(TOP_N_WIDE).copy()
cand_high_uncertain = cand_high_uncertain.head(TOP_N_UNCERTAIN).copy()

# ===== 11) 合并成总候选表 =====
cand_high_narrow["candidate_group"] = "high_score_narrow_interval_good_error"
cand_high_wide["candidate_group"] = "high_score_wide_interval_good_error"
cand_high_uncertain["candidate_group"] = "high_score_wide_interval_large_error"

cand_balanced = pd.concat(
    [cand_high_narrow, cand_high_wide, cand_high_uncertain],
    axis=0,
    ignore_index=True
)

# 再加一个更适合看论文表的简洁列顺序
final_cols = [
    "candidate_group",
    "standard_inchi_key",
    "chembl_molecule_chembl_id",
    "uniprot_main",
    "uniprot_id",
    "chembl_target_chembl_id",
    "affinity_type",
    "y_true",
    "y_pred_calibrated",
    "err_bound",
    "interval_lo_calibrated",
    "interval_hi_calibrated",
    "abs_error_calibrated",
    "prob_within_tol_0p2_calibrated",
    "accepted",
    "prob_strong_binder",
]
final_cols = [c for c in final_cols if c in cand_balanced.columns]
cand_balanced = cand_balanced[final_cols].copy()

# ===== 12) 导出 =====
cand_high_narrow.to_csv(OUT_HIGH_NARROW, index=False, encoding="utf-8-sig")
cand_high_wide.to_csv(OUT_HIGH_WIDE, index=False, encoding="utf-8-sig")
cand_high_uncertain.to_csv(OUT_HIGH_UNCERTAIN, index=False, encoding="utf-8-sig")
cand_balanced.to_csv(OUT_BALANCED, index=False, encoding="utf-8-sig")

# ===== 13) 输出统计说明 =====
affinity_counts = df["affinity_type"].value_counts(dropna=False).to_dict()

with open(OUT_STATS, "w", encoding="utf-8") as f:
    f.write("=== Candidate selection summary ===\n")
    f.write(f"Total rows after cleaning/dedup: {len(df)}\n")
    f.write(f"Affinity type counts: {affinity_counts}\n\n")
    f.write(f"HIGH_SCORE threshold: {HIGH_SCORE:.3f}\n")
    f.write(f"VERY_HIGH_SCORE threshold: {VERY_HIGH_SCORE:.3f}\n")
    f.write(f"NARROW err_bound threshold (Q25): {narrow_q25:.3f}\n")
    f.write(f"WIDE err_bound threshold (Q75): {wide_q75:.3f}\n")
    f.write(f"GOOD_ERROR threshold: {GOOD_ERROR:.3f}\n")
    f.write(f"MODERATE_ERROR threshold: {MODERATE_ERROR:.3f}\n")
    f.write(f"BAD_ERROR threshold: {BAD_ERROR:.3f}\n\n")
    f.write(f"cand_high_narrow: {len(cand_high_narrow)}\n")
    f.write(f"cand_high_wide: {len(cand_high_wide)}\n")
    f.write(f"cand_high_uncertain: {len(cand_high_uncertain)}\n")
    f.write(f"cand_balanced: {len(cand_balanced)}\n")

print("完成。输出文件：")
print(OUT_HIGH_NARROW)
print(OUT_HIGH_WIDE)
print(OUT_HIGH_UNCERTAIN)
print(OUT_BALANCED)
print(OUT_STATS)