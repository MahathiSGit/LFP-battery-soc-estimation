# 🔋 Battery SOC/SOH Estimation

Estimating State of Charge (SOC) and State of Health (SOH) for a Li-ion
battery pack. SOC estimation uses a hybrid pipeline: an **OCV-SoC lookup
table** for normal (non-plateau) regions, and a sequence model (**GRU** or
**Transformer**) to bridge voltage plateau regions, where the OCV curve
goes nearly flat and voltage alone can't resolve SoC reliably. A separate
analysis compares the OCV table itself against real pack log data to
check how well it matches this specific hardware.

---

## 🚨 Important Rules

- **Never commit directly to `main`.**
- **Always work on your own branch.**
- **Pull the latest changes before starting work.**
- **Push only your own branch.**
- **Do NOT merge your own Pull Request — discuss with the team first.**

---

## Step 1: Clone the repository (Only once)

```bash
git clone <repository-url>
cd <repository-name>
```

---

## Step 2: Set up the environment (Only once)

```bash
conda env create -f environment.yml
conda activate battery-soc
```

---

## Step 3: Switch to the branch you're working on

This repo's branches, one per workstream:

```bash
git checkout data_cleaning
git checkout gru
git checkout transformer
git checkout compare_OCV_analysis
```

If you're starting a new workstream not listed above:
```bash
git checkout -b <your-branch-name>
git push -u origin <your-branch-name>
```

---

## Step 4: Before you start coding (Every session)

Always sync with the latest `main` before touching any code.

```bash
git checkout main
git pull origin main

git checkout <your-branch-name>
git merge main
```

---

## Step 5: Save your work

```bash
git add .
git commit -m "Short description of what you changed"
```

Examples for this project:
```bash
git commit -m "Added slope-based checkpoint detector"
git commit -m "Fixed anchor SoC calculation in data_prep"
git commit -m "Added masked MAE loss for GRU training"
git commit -m "Added causal mask to Transformer model"
git commit -m "Added charge/discharge-aware OCV interpolation"
```

---

## Step 6: Push your changes

```bash
git push origin <your-branch-name>
```

---

## Step 7: Open a Pull Request

1. Go to GitHub.
2. Open your branch.
3. Click **Compare & Pull Request**.
4. Write what you changed and why.
5. Tag the team for review.
6. **Wait for discussion before merging.**

---

## 🔄 Daily Workflow

```
Pull latest main
        ↓
Switch to your branch
        ↓
Merge main into your branch
        ↓
Code
        ↓
Commit
        ↓
Push
        ↓
Create Pull Request
```

---

## ❌ Things NOT to Do

- Don't push directly to `main`.
- Don't work on someone else's branch.
- Don't force push (`git push --force`).
- Don't delete someone else's branch.
- Don't commit broken or half-finished code.
- Don't commit large data files or model checkpoints (`.pt`, `.csv`).



---

## 🧠 Architecture Overview (SoC pipeline)

**How it works:**

1. **Battery pack sensors** stream voltage, current, and temperature.
2. A **checkpoint detector** decides, at every point in time, whether the
   battery is in a normal region or a voltage plateau.
3. **Non-plateau periods** → SoC comes straight from the **OCV-SoC lookup
   table**.
4. **Plateau periods** → the pipeline takes an **anchor SoC** (the last
   trustworthy table value at plateau entry) and hands off to the **GRU or
   Transformer model**, which reads voltage, current, temperature, and
   elapsed time and predicts how far SoC has drifted from that anchor
   (`delta-SoC`).
5. **Final SoC estimate** = anchor + predicted delta on a plateau, or the
   direct table lookup otherwise.
6. The anchor **resets every time a new plateau begins**, so each plateau
   session is estimated independently from a fresh, trustworthy starting
   point.

### Key Design Decisions

- **Delta prediction, not absolute SoC**: both models predict SoC *drift*
  from an anchor, not absolute SoC — this keeps the job well-posed (a
  bounded correction) instead of ill-posed (reconstructing SoC from a flat
  curve).
- **Checkpoint detector routing**: a slope-based `|dOCV/dSoC|` detector
  flags plateau vs. non-plateau; a placeholder SoC-bound fallback exists
  for when low-current data coverage is too sparse to trust the slope
  estimate.
- **Causal by construction**: the GRU is recurrent (past-only by nature);
  the Transformer uses an explicit causal mask so it can't use future
  timesteps either — both stay valid for real-time inference.
- **Session-based training**: each plateau episode (entry to exit) is one
  training sequence ("session"); models are evaluated with
  leave-one-session-out (LOSO) cross-validation.

### Current Status

The pipeline is validated end-to-end on both models, but training data is
currently limited to **3 plateau sessions from 1 cell** — enough to
confirm the pipeline works, not enough for either model to learn a
relationship that generalizes. The session with an anchor SoC far from
the other two consistently shows the worst error in LOSO evaluation,
regardless of model architecture — a data-coverage gap, not a modeling
failure. More sessions, ideally pooled across many cells/logs, are needed
before either model's accuracy should be trusted for deployment.

The OCV table comparison (`compare_OCV_analysis` branch) separately found
a systematic ~0.4–0.6V offset between measured and table-predicted pack
voltage on charge-classified rest points — see that branch's
documentation for the full breakdown and open questions.

---

## 💬 If You Get Stuck

Don't randomly try Git commands.
Ask in the team group before doing anything that could affect `main` or
someone else's branch.