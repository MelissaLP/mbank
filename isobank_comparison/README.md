# mbank on the isobank eccentric 5D space

mbank run on the space of isobank's `configs/eccentric_5d.py`: m1 in [1, 7], m2 in [1, 3], s1z in [-0.9, 0.9],
s2z in [-0.05, 0.05], e in [1e-5, 0.15], `TaylorF2Ecc`, f_min = 15 Hz, `aligo_O3actual_H1` ASD, symphony metric.

| File | What |
|---|---|
| `eccentric_5d.ini` | settings of the three mbank steps |
| `aligo_O3actual_H1.txt` | the ASD (a copy of isobank's `data/psds/`) |
| `environment.yml` | the `mbank` conda env |
| `timing.py` | times a small piece of each step and scales it up to the ini |

## Setup

```bash
conda env create -f isobank_comparison/environment.yml
conda activate mbank
pip install --no-deps -e .          # this fork, from the repo root
```

The env sets `OMP_NUM_THREADS=1`. With more threads, lalsimulation 7.0.0's `TaylorF2Ecc` returns a different
waveform at every call for e > 0 (up to 20 % apart), and the metric volumes become noise (log volume off by
up to ~50). With one thread it is reproducible and identical to lalsimulation 6.2.1. Keep it on the cluster too.

## Run

From this folder:

```bash
mbank_generate_flow_dataset eccentric_5d.ini
mbank_train_flow eccentric_5d.ini
mbank_place_templates eccentric_5d.ini
```

Outputs go to `eccentric_5d/`. The bank, `eccentric_5d/bank.npy`, is in the isobank coordinates
(m1, m2, s1z, s2z, e), so it is scored as the isobank bank is:

```bash
python scripts/evaluate_bank.py --config configs/eccentric_5d.py \
    --bank .../eccentric_5d/bank.npy --save-ff runs/eccentric_5d/ff_mbank.npy     # in isobank
```

The shortlist uses isobank's flow: check that the miss rate is flat in k for the mbank bank too.

## Time

Measured on an Apple M5 laptop, one thread, with the ini as it is:

| Step | Time |
|---|---|
| dataset: 20000 metrics (0.011 s each) | 3.7 min |
| training: early stop at epoch 5500 | 11 s |
| placement: 2000 livepoints, 22168 iterations | 3.7 min |
| total | 7.6 min |

The result: 2.85M templates at covering fraction 0.9. `python timing.py eccentric_5d.ini --flow eccentric_5d/flow.zip`
estimates the times for other settings; its placement time is an upper bound (it assumes no livepoint is covered).

## Known issue: probability outside the box

The flow's range is the box plus a 1 % margin per side (`epsilon_` in `mbank/flow/flowmodel.py`). The loss only
sees the training points, none of which are in the margin, and the trained flow puts 95 % of its samples there.
The fit inside the box is good (median error in log volume 0.036), so the bank is not affected, but:

- placement discards ~95 % of its proposals (only ~130 of each 3000 are kept);
- the volume mbank reports (`C`) is too large by the inverse of the fraction inside (~22x).
