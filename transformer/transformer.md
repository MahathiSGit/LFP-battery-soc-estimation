# Plateau-Region SoC Estimation — Transformer

Estimates state of charge (SoC) in the flat OCV plateau region, using the
same anchor + delta framing as the GRU branch: rather than predicting
absolute SoC directly (ill-posed on a near-flat voltage signal), the
model predicts **delta-SoC since a known anchor point** (the OCV-table
value at plateau entry).

This branch swaps the GRU for a causal Transformer encoder, evaluated as
an alternative sequence architecture for the same task.

## Files in this branch

- **`data.py`** — same as the `gru` branch: loads the raw pack log,
  detects plateau sessions, and turns each into a `(features, target)`
  sequence. Shared, unchanged.
- **`model_transformer.py`** — `PlateauTransformer`, a causal Transformer
  encoder that predicts delta-SoC at every timestep.
- **`train.py`** — same model-agnostic training loop as the `gru` branch.
  Works with any model implementing `forward(x, lengths) -> (B, T)`
  delta-SoC predictions, so no changes needed here between branches.
- **`run_transformer.py`** — loads sessions, trains the Transformer,
  evaluates per session, plots results, and exports CSVs.

## Why a causal mask

A vanilla Transformer encoder attends **bidirectionally** across the
whole input sequence by default — it would happily use *future*
timesteps to help predict the current one. The GRU can't do that; it's
recurrent, so it only ever sees the past. Since this model is meant for
real-time SoC estimation (a BMS doesn't have future voltage samples at
inference time), `PlateauTransformer` applies a causal mask so each
timestep can only attend to itself and earlier timesteps — matching the
GRU's past-only behavior rather than giving the Transformer an unfair
advantage that wouldn't hold up in deployment.

## Why positional encoding

Unlike the GRU, a Transformer has no inherent notion of sequence order —
attention alone is permutation-invariant. A standard sinusoidal
positional encoding is added to the input embeddings so the model can
tell *which* timestep it's looking at, not just *what* the values are.

## Model design

```
SoC(t) = anchor_SoC + delta_SoC_predicted(t)
```

Same as the GRU branch — only the architecture producing
`delta_SoC_predicted(t)` differs.

Input features per timestep: **voltage, current, temperature, elapsed
time since session start** (4 features), same as the GRU branch, for a
fair comparison.

## Evaluation

Leave-one-session-out, same as the GRU branch: train on all other
sessions, evaluate on the held-out one, report MAE in SoC% per session.

## Requirements

```bash
conda env create -f environment.yml
conda activate battery-soc
```

## Running it

Make sure these are all in the same folder:
- `data.py`
- `model_transformer.py`
- `train.py`
- `run_transformer.py`
- your cleaned pack log CSV

Then:
```bash
conda activate battery-soc
python run_transformer.py
```

## Output

- `transformer_sessions.png` — 3-panel plot (true SoC, predicted SoC,
  anchor) per session.
- `transformer_predictions_per_sample.csv` — every timestep's true SoC,
  predicted SoC, and absolute error.
- `transformer_results_summary.csv` — one row per session: MAE, sample
  count, anchor SoC, SoC range.

## Known limitations

- **Data-hungry by design.** Attention-based models generally need more
  training examples than a recurrent model to learn stable patterns.
  With only a couple of sessions available from a single-cell log, don't
  expect the Transformer to outperform the GRU here — the GRU's stronger
  recurrent inductive bias is simply a better fit at this data scale.
  This isn't evidence the Transformer architecture is worse; it's
  evidence this dataset is too small to fairly judge either model.
- **Same session-coverage caveat as the GRU branch.** A session whose
  anchor SoC sits far from the training sessions' anchors will predict
  poorly, regardless of architecture — that's a data-coverage gap, not
  something either model's design can fix on its own.
- **Same anchor-accuracy dependency as the GRU branch** — SoC is
  reconstructed as anchor + delta, so anchor errors propagate directly
  into every prediction.

## Note on comparing against the GRU branch

`data.py` and `train.py` are identical between the `gru` and
`transformer` branches by design, so results are directly comparable —
same sessions, same features, same evaluation protocol, only the model
differs. If you want a single script that trains both models on
*guaranteed*-identical session objects in one run (rather than trusting
two separate scripts stay in sync), that's the `run_compare.py` pattern
from earlier in this project — worth pulling into whichever branch ends
up merged last, or into `main` once both models are merged.