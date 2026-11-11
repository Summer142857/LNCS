# LNCS: Language-based Neural COP Solver

Code for *Bridging Large Language Models and Optimization: A Unified Framework
for Text-attributed Combinatorial Optimization* (ACML 2026).

LNCS combines text-attributed instances, a frozen text encoder, a shared
attention generator and CGERL multi-task reinforcement learning. The release
supports TSP, CVRP, KP, MVC, SMTWTP, VRPB and MIS at sizes 20, 50 and 100.

This is release candidate **0.1.0-rc3**. It includes training, synthetic data
generation and raw-objective evaluation. Pretrained solver checkpoints, the
paper's original test splits and certified reference labels are not included.
Read [RELEASE_NOTES.md](RELEASE_NOTES.md) for implementation changes and
paper/code differences before comparing experimental results.

## Installation

Use Python 3.10 or newer, with a PyTorch build appropriate for your CPU or CUDA
system. The locally verified environment uses Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The core environments and mock tests require PyTorch, NumPy, NetworkX and SciPy.
The full requirements also install the text encoders and their integration
tests. Exact observed versions are in `requirements-local-tested.txt` (macOS
CPU/MPS) and `requirements-tested.txt` (earlier cluster environment).

Llama weights require the provider's access approval. Authenticate using your
own Hugging Face login or `HF_TOKEN`. Use `--cache-dir /path/to/cache` to reuse
models and `--local-files-only` to prevent downloads. Model weights and their
licenses are obtained separately. No solver or external API is needed for RL
training or raw-objective evaluation.

## Quick software check

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --output outputs/smoke
python generate_data.py --problem-size 20 --episodes 5 --output data/smoke
python evaluate.py --episodes 5 --checkpoint outputs/smoke/checkpoint-1.pt \
  --data-dir data/smoke --output outputs/smoke-evaluation.json
```

`--smoke-test` uses deterministic mock text embeddings and a small generator.
It verifies execution and must not be used to report LNCS performance.
Evaluation inherits the checkpoint's architecture, encoder, device, task list,
problem size and POMO size; explicit CLI options override them.

For a real pretrained encoder with the full six-layer, 512-dimensional generator:

```bash
python train.py --encoder ST --device cpu --problem-size 20 --pomo-size 4 \
  --epochs 1 --episodes 3 --batch-size 2 --output outputs/st-check
```

The first run downloads `sentence-transformers/multi-qa-distilbert-cos-v1`.
On Apple Silicon with MPS available, use `--device mps`. These short checks
exercise training, not solution quality. See [VALIDATION.md](VALIDATION.md).

## Training and checkpoints

`configs/base.json` uses Llama-2-7B, five base tasks, size 50, 200 epochs,
2,000 instances per task per epoch, batch size 16, Adam (learning rate 1e-4,
weight decay 1e-6), and CGERL projection with summed task gradients.

```bash
python train.py --problem-size 20 --epochs 100 --output outputs/lncs20
python train.py --problem-size 50 --epochs 200 --output outputs/lncs50
```

Other encoders are `ST`, `e5`, `llama3_8b` and `qwen3`. Causal encoders default
to float32 and `legacy_mean` pooling; lower precision and `masked_mean` are
explicit options that change the representation. `--no-pcgrad` disables CGERL.

Warm-start a new run or transfer task using weights without restoring optimizer
state or the epoch counter:

```bash
python train.py --problem-size 100 --epochs 20 \
  --checkpoint outputs/lncs50/checkpoint-200.pt --output outputs/lncs100
python train.py --tasks VRPB --problem-size 50 --epochs 16 \
  --checkpoint outputs/lncs50/checkpoint-200.pt --output outputs/vrpb50
python train.py --tasks MIS --problem-size 100 --epochs 48 \
  --checkpoint outputs/lncs50/checkpoint-200.pt --output outputs/mis100
```

At 2,000 episodes and batch size 16, each epoch has 125 updates. Thus 16 and
48 epochs correspond to 2,000 and 6,000 updates, respectively.

Resume an RC3 run with its saved settings and a new final epoch number:

```bash
python train.py --resume --epochs 200 \
  --checkpoint outputs/lncs50/checkpoint-100.pt --output outputs/lncs50
```

Resume requires the same device backend and training settings. Use a warm-start
when changing backends or loading earlier releases. Earlier compatible generator
weights load after discarding only the unused `llm_emb` copy; no other unexpected
parameters are ignored. New training runs refuse to reuse an existing run
unless `--overwrite` is explicit. Keep a separate output directory for each run.

## Data and evaluation

```bash
python generate_data.py --tasks TSP CVRP KP MVC SMTWTP VRPB MIS \
  --problem-size 50 --episodes 1000 --seed 1234 --output data/test50
python evaluate.py --tasks TSP CVRP KP MVC SMTWTP VRPB MIS \
  --episodes 1000 --batch-size 16 --checkpoint outputs/lncs50/checkpoint-200.pt \
  --data-dir data/test50 --output outputs/test50.json
```

These are newly generated instances. For a fixed task ordering, each dataset
uses `seed + task_index`, recorded in its metadata. Files are tensor-only `.pt`
dictionaries loaded with `weights_only=True`; shape, value and graph constraints
are validated. Generation refuses duplicate tasks and existing files unless
`--overwrite` is specified.

Distributions use uniform coordinates/features, CVRP/VRPB capacities 30/40/50,
KP capacities 5/12.5/25, Barabási–Albert graphs (attachment parameter 4), and
SMTWTP due times in [0,n] with uniform weights and processing times. VRPB has
20% pickup customers and mixed delivery/pickup service without precedence.
Each route initially carries its own total delivery demand; payload must remain
in [0,1]. POMO starts prefer delivery nodes and are capped by the number of
available starts in the batch.

Reports include mean objective, standard error and instance count. KP and MIS
maximize; the other tasks minimize. Evaluation selects the best POMO rollout
per instance and uses eight geometric augmentations for routing by default
(`--augmentation 1` disables them). Optimality gaps require independently
validated reference solutions. This release includes TAI/heuristic text
construction, but does not include an exact-solver labeling pipeline.

## Citation and license

```bibtex
@inproceedings{jiang2026lncs,
  title={Bridging Large Language Models and Optimization: A Unified Framework for Text-attributed Combinatorial Optimization},
  author={Jiang, Xia and Wu, Yaoxin and Wang, Yuan and Zhang, Yingqian},
  booktitle={Asian Conference on Machine Learning},
  year={2026}
}
```

Original contributions are under MIT. Upstream portions retain their licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Funding: This work is supported by National Natural Science Foundation of China
under grant No. 52472461 from The Second Research Institute of Civil Aviation
Administration of China.
