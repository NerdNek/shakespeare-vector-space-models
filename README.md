# Assignment 3 — Shakespeare Vector Space Models

CECS 429/529 Search Engine Technology. Group assignment code for term-document / term-context matrices, TF-IDF, PPMI, and similarity ranking.

## Setup

```bash
cd assignment3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required dependency: **NumPy**.  
Optional dependency: **SciPy** (recommended for the full corpus). A dense `|V|×|V|` float32 matrix with `|V|≈22,602` needs about **2 GB** of RAM; the sparse path avoids that allocation.

Place these files next to `main.py` (already in this folder):

- `will_play_text.csv`
- `vocab.txt`
- `play_names.txt`

## Run the required assignment path

```bash
python main.py
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--seed 42` | Reproducible random target play (default 42) |
| `--window 2` | Term-context window (default 2, matches starter `__main__`) |
| `--dense-context` | Use starter-style dense NumPy `|V|×|V|` matrices (~2 GB+) |
| `--skip-slow-set-sims` | Word rankings with cosine only |

Default full-corpus run builds **sparse** term-context / PPMI matrices, then ranks with the same formulas as the dense helpers. The required functions `create_term_context_matrix` and `create_PPMI_matrix` still return **dense NumPy arrays** when you call them directly (starter contract).

## Small verification tests

```bash
python test_small_examples.py
```

## Optional experiments (not required)

```bash
python optional_experiments.py
python optional_experiments.py --all
python optional_experiments.py --word throne --characters --ablation
```

These scripts compare representations and preprocessing variants. They report overlaps and neighbor lists; they do **not** claim a ranking is more accurate without external evidence.

## Sparse vs dense tradeoff

| | Dense NumPy (starter) | SciPy sparse (optional) |
|--|-----------------------|-------------------------|
| Return type | `numpy.ndarray` | `scipy.sparse.csr_matrix` |
| Matches starter docstring | Yes | No (different type) |
| Memory for `|V|≈22k` | ~2 GB per matrix | Only nonzero cells |
| Dependency | NumPy only | NumPy + SciPy |

Use dense when you need strict starter compatibility on a small vocabulary or a machine with enough RAM. Use sparse for the full Shakespeare run.
