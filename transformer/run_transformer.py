import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import load_sessions
from model_transformer import PlateauTransformer
from train import leave_one_session_out, predict_session

sessions = load_sessions(
    'OX002386SA3P16S3538_cleaned.csv', cell_id='cell0',
    slope_thresh=0.001, use_placeholder=True, placeholder_bounds=(20, 90),
)
print(f"{len(sessions)} sessions loaded, feature dim = {sessions[0].features.shape[1]}")

results = leave_one_session_out(
    sessions,
    model_cls=PlateauTransformer,
    hidden_size=32,
    model_kwargs={"input_size": 4, "num_heads": 4, "num_layers": 2, "dropout": 0.1},
    lr=3e-3,
    epochs=300,
    batch_size=4,
    weight_decay=1e-3,
    patience=30,
    verbose=True,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
per_sample_rows = []
summary_rows = []

for i, (res, session) in enumerate(zip(results, sessions)):
    pred = predict_session(res["model"], res["scaler"], session)
    abs_err = np.abs(pred - session.soc_true)
    mae = abs_err.mean()

    ax = axes[i]
    ax.plot(session.soc_true, label="true SoC", color="tab:blue")
    ax.plot(pred, label="Transformer predicted SoC", color="tab:orange", linestyle="--")
    ax.axhline(session.anchor_soc, color="gray", linestyle=":", label="anchor")
    ax.set_title(f"Session {i} (held out)\nMAE={mae:.2f} SoC%")
    ax.set_xlabel("sample index")
    if i == 0:
        ax.set_ylabel("SoC (%)")
    ax.legend()

    for t in range(len(session.soc_true)):
        per_sample_rows.append({
            "session": i,
            "cell_id": session.cell_id,
            "sample_index": t,
            "anchor_soc": session.anchor_soc,
            "true_soc": session.soc_true[t],
            "predicted_soc": pred[t],
            "abs_error": abs_err[t],
        })

    summary_rows.append({
        "session": i,
        "cell_id": session.cell_id,
        "n_samples": len(session.soc_true),
        "anchor_soc": session.anchor_soc,
        "soc_min": session.soc_true.min(),
        "soc_max": session.soc_true.max(),
        "mae_soc_pct": mae,
        "best_val_loss": res["best_val_loss"],
    })

plt.tight_layout()
plt.savefig("transformer_sessions.png", dpi=150)
print("saved transformer_sessions.png")

per_sample_df = pd.DataFrame(per_sample_rows)
summary_df = pd.DataFrame(summary_rows)
per_sample_df.to_csv("transformer_predictions_per_sample.csv", index=False)
summary_df.to_csv("transformer_results_summary.csv", index=False)
print("saved transformer_predictions_per_sample.csv")
print("saved transformer_results_summary.csv")