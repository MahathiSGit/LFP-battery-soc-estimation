from data import load_sessions
from model_gru import PlateauGRU
from train import leave_one_session_out, predict_session

sessions = load_sessions(
    "OX002386SA3P16S3538_cleaned.csv",
    cell_id="pack",
    slope_thresh=0.001,
    use_placeholder=True,
    placeholder_bounds=(20, 90),
)
print(f"{len(sessions)} sessions loaded")

results = leave_one_session_out(
    sessions,
    model_cls=PlateauGRU,
    hidden_size=32,
    model_kwargs={"input_size": 4, "num_layers": 1, "dropout": 0.1},
    lr=3e-3,
    epochs=300,
    batch_size=4,
    weight_decay=1e-3,
    patience=30,
    verbose=True,
)

for i, (res, session) in enumerate(zip(results, sessions)):
    pred = predict_session(res["model"], res["scaler"], session)
    mae = abs(pred - session.soc_true).mean()
    print(f"session {i}: MAE = {mae:.2f} SoC%")