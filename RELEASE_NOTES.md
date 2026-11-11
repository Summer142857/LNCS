# Implementation and release notes

Revision 0.1.0-rc3, 2026-11-10. This candidate provides a maintained execution
pipeline; the paper's numerical results have not been reproduced with it.

## RC3 changes

- Removed the unused `llm_emb` parameter copy, unreachable numeric-input
  branches, obsolete pickle/testset loaders and unused VRPB variant metadata.
  Active generator parameter names and shapes remain unchanged. Compatible
  earlier weights support warm-start/evaluation; optimizer resume requires
  RC3 checkpoint format 2.
- Corrected VRPB capacity masks to the mixed-service definition in the paper:
  a route starts with the sum of its positive demands, and payload changes by
  subtracting each signed demand. Both initial and intermediate payloads must
  be in [0,1]. The previous full-load reset could accept excess initial delivery
  loads and reject valid pickups. VRPB results must be reevaluated under this
  correction. Decoder `load` is remaining delivery budget, `1-D-M`, where D is
  accumulated delivery demand and M is maximum prefix net pickup in the route.
- Cache task embeddings by exact task text, including graph edge counts.
  Encode task text separately from node text so cold and warm cache paths use
  the same batching convention. This changes causal embeddings relative to
  earlier releases when using padding-sensitive `legacy_mean`.
- MVC supports variable edge counts and empty graphs. MVC/MIS explicitly track
  finished rollouts and exclude padding from solution prefixes and policy loss.
- Added dataset/configuration validation, checkpoint-setting inheritance,
  output overwrite protection and regression checks. SMTWTP cumulative
  completion times now use a cumulative sum with the same objective definition.

## Earlier correctness changes retained

- CGERL conflict projection is enabled, using fixed original reference
  gradients and summing projected task gradients. The retrieved source's
  conflict branch was a no-op; this alone does not establish which historical
  experiments used that source version.
- The last SMTWTP job retains its own attributes. Transfer prompts and shared
  model wiring support VRPB and MIS. Historical batch-sized task caches are
  recomputed when loading weights.
- Text encoders remain frozen in evaluation mode. Sentence Transformer outputs
  permit connector backward; bfloat16 outputs can be converted to NumPy.
  Checkpoints include CPU, Python, NumPy and applicable CUDA/MPS random states.

## Paper/code differences and reproduction scope

| Item | Released behavior |
| --- | --- |
| Normalization | InstanceNorm1d from the retrieved implementation; paper prose describes batch normalization. Retained for weight compatibility. |
| Final pointer | Combined attention output scores encoded nodes directly. No extra final query/key projection layers are added. |
| Causal pooling | Default `legacy_mean` averages all positions, including right padding. Node embeddings can depend on text batching. Explicit `masked_mean` changes this; E5 uses masked pooling. |
| Prompts | Five base-task wording/heuristics retained except the corrupted final SMTWTP job; explicit transfer instructions added. |
| Historical checkpoints | Matching tensor shapes establishes load compatibility, not training provenance or correspondence to a paper table. |
| Data and reference labels | Seeded synthetic generators and TAI/heuristic text included; original test splits, certified reference solutions, exact-solver annotation pipeline and pretrained solver weights not distributed. |

Changes to constraints, prompts, pooling, gradient projection or data batching
can change experimental results. Reevaluate or retrain as appropriate before
claiming paper-level reproduction. [VALIDATION.md](VALIDATION.md) records
software checks and their limits.
