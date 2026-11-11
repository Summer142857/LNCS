# Validation record

Revision 0.1.0-rc3. Date: 2026-11-10.

Local environment: macOS 26.6.2 arm64, Python 3.12.14, PyTorch 2.15.0,
Transformers 4.57.6 and Sentence Transformers 5.7.0. Exact installed versions
are recorded in `requirements-local-tested.txt`.

| Check | Verified scope |
| --- | --- |
| Functional/integration suite | 24 tests; seven tasks and sizes 20/50/100; independently recomputed feasibility/objectives; projection; caching; prompts; datasets and CLI/checkpoint guards |
| VRPB constraints | Overload counterexample, valid mixed pickup case, and 64 signed-demand combinations checked against independently simulated payloads |
| Graph boundaries | Variable edge counts, empty and complete graphs, finished-rollout padding, zero-loss empty vertex-cover training |
| CPU/mock resume | Resumed parameters match uninterrupted training exactly |
| Apple MPS/mock resume | RNG preserved; parameters agree within rtol=1e-4 and atol=1e-6 |
| Generator cleanup | Seven-task RC2/RC3 generator comparison using the same environments, active weights and mock embeddings: matching greedy actions; node encodings agree within rtol=1e-5 and atol=2e-6 |
| Offline encoder integration | Tiny random Llama, Qwen3, BERT and Sentence Transformer fixtures; pooling, frozen weights, bfloat16 conversion and connector backward |
| Real pretrained ST | CPU and Apple MPS: seven-task training, save/load, resume and independent feasibility/objective checks with the full six-layer, 512-dimensional generator |
| Packaged workflow | ZIP/manifest integrity, syntax/content scan, test suite, generated data, CLI training/evaluation and incomplete final batches |

Repeat the suite with:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The MPS-specific test skips when that backend is unavailable. Tiny encoder
fixtures require no model download and only test library integration. Separate
real ST checks use actual pretrained `multi-qa-distilbert-cos-v1` weights,
size 20, four POMO starts, three training instances/task/epoch and two epochs;
independent solution checks cover two instances/task with eight augmentations
for routing. These short runs do not establish trained-solver quality.

Full pretrained Llama2/Llama3/Qwen3 execution on CUDA and reproduction of the
paper's tables, curves or optimality gaps remain unverified. Historical weight
provenance and original evaluation splits remain unconfirmed. The implementation
differences in [RELEASE_NOTES.md](RELEASE_NOTES.md) also limit exact reproduction.
