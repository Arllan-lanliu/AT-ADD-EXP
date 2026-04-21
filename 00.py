

import pandas as pd
from sklearn.metrics import classification_report, f1_score


score_path = "/data/liulan/workspace/released_models/AT-ADD-Baseline/ckpt_t2/ft-beatsaasist_aug_speech0.3_sound0.8_algo4/analysis_dev/result/atadd-track2_logits_dev.csv"
df_pred = pd.read_csv(score_path)

y_true = df_pred["label"].values
y_pred = df_pred["predict"].values

print("\n===== Overall Performance =====")
report = classification_report(y_true, y_pred, target_names=["real", "fake"])
print(report)

print("\n===== Performance by Audio Type =====")
types = df_pred["type"].unique()

type_macro_f1 = {}

for t in types:
    subset = df_pred[df_pred["type"] == t]

    y_true_t = subset["label"]
    y_pred_t = subset["predict"]

    report_t = classification_report(
        y_true_t, y_pred_t, target_names=["real", "fake"], output_dict=True
    )

    f1_real = report_t["real"]["f1-score"]
    f1_fake = report_t["fake"]["f1-score"]
    macro_f1 = (f1_real + f1_fake) / 2

    type_macro_f1[t] = macro_f1

    print(f"\n--- Type: {t} ---")
    print(classification_report(y_true_t, y_pred_t, target_names=["real", "fake"]))
    print(f"Macro-F1 ({t}): {macro_f1:.4f}")

output_order = ["speech", "sound", "music", "singing"]
type_macro_f1 = {t: type_macro_f1[t] for t in output_order if t in type_macro_f1}
track2_score = sum(type_macro_f1.values()) / len(type_macro_f1)
print("\nMacro-F1\tSpeech\tSound\tSinging\tMusic")
print(
    "{:.4f}\t\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        track2_score,
        type_macro_f1.get("speech", float("nan")),
        type_macro_f1.get("sound",  float("nan")),
        type_macro_f1.get("singing",float("nan")),
        type_macro_f1.get("music",  float("nan")),
    )
)