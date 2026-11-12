import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def compute_icc_two_way_random(ratings):
    n, k = ratings.shape
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1, keepdims=True)
    col_means = np.mean(ratings, axis=0, keepdims=True)

    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    ms_rows = ss_rows / df1
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / df2

    icc_2_1 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error))
    icc_2_k = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    f_stat = ms_rows / ms_error
    return icc_2_1, icc_2_k, f_stat, df1, df2

def evaluate_regression(y_true, y_pred):
    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    plcc = pearsonr(y_true, y_pred)[0]
    srcc = spearmanr(y_true, y_pred)[0]
    return r2, rmse, mae, plcc, srcc

def main():
    df_ann = pd.read_csv("CB_FQA_annotations_v2.csv")
    df_pred = pd.read_csv("test_set_predictions.csv")

    print("=================================================================")
    print("1. Dataset Split & Action Distribution (Tables 5, 7, 8, and 9)")
    print("=================================================================")
    print("Split counts:", df_ann["Split"].value_counts().to_dict())
    print("Action counts:", df_ann["Action_Category"].value_counts().to_dict())
    print("\nTest Set Distribution by Participant (Table 8):")
    print(pd.crosstab(df_pred["Defender_ID"], df_pred["Action_Category"], margins=True))

    print("\n=================================================================")
    print("2. Inter-Rater Reliability Analysis (Table 6)")
    print("=================================================================")
    rater_cols = ["Rater_1", "Rater_2", "Rater_3", "Rater_4", "Rater_5"]
    for cat in ["Lateral Sliding", "Crossover Transition", "Close-out & Contesting", "Backpedal & Recovery"]:
        sub = df_ann[df_ann["Action_Category"] == cat][rater_cols].values
        i1, i5, f_val, df1, df2 = compute_icc_two_way_random(sub)
        print(f"{cat:<24} | N={len(sub):<4} | ICC(2,1)={i1:.3f} | ICC(2,5)={i5:.3f} | F({df1},{df2})={f_val:.2f}")

    all_ratings = df_ann[rater_cols].values
    i1, i5, f_val, df1, df2 = compute_icc_two_way_random(all_ratings)
    print(f"{'Overall Dataset':<24} | N={len(all_ratings):<4} | ICC(2,1)={i1:.3f} | ICC(2,5)={i5:.3f} | F({df1},{df2})={f_val:.2f}")

    print("\n=================================================================")
    print("3. Test-Set Benchmark & Ablation Evaluation (Tables 10 & 13)")
    print("=================================================================")
    y_true = df_pred["Mean_Score_Norm"].values
    model_cols = [
        "Pred_Unregistered", "Pred_ICP", "Pred_ST_GCN", "Pred_USDL",
        "Pred_TwoStage_AQA", "Pred_CTR_GCN", "Pred_Backbone",
        "Pred_wo_DIF", "Pred_wo_PA", "Pred_wo_KCL", "Pred_Kin_DualGAT_ConvLSTM"
    ]
    for col in model_cols:
        r2, rmse, mae, plcc, srcc = evaluate_regression(y_true, df_pred[col].values)
        print(f"{col:<28} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f} | PLCC={plcc:.3f} | SRCC={srcc:.3f}")

    print("\n=================================================================")
    print("4. Category-Wise Performance of Kin-DualGAT-ConvLSTM (Table 11)")
    print("=================================================================")
    for cat in ["Lateral Sliding", "Crossover Transition", "Close-out & Contesting", "Backpedal & Recovery"]:
        sub = df_pred[df_pred["Action_Category"] == cat]
        r2, rmse, mae, plcc, srcc = evaluate_regression(sub["Mean_Score_Norm"].values, sub["Pred_Kin_DualGAT_ConvLSTM"].values)
        print(f"{cat:<24} | N={len(sub):<3} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

if __name__ == "__main__":
    main()