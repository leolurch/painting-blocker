# Museum-image deduplication

Standalone, resumable pipeline for consolidating museum downloads, proposing visual duplicates,
recording human decisions, and exporting a standard experiment dataset. Raw source files are never
changed. No database is used until `export-dataset` creates the final `dataset.db`.

## Setup

Python 3.11+ is required. The DINOv3-7B and LightGlue stages require the configured GPU runtime.

```bash
cd experiments/datasets/deduplication
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install --force-reinstall --no-cache-dir opencv-python-headless==4.11.0.86
pip install --no-deps git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8
cp config/deduplication.example.toml config/deduplication.toml
```

The DINO model is gated on Hugging Face; authenticate in the GPU environment before embedding.
The model revision, preprocessing, cosine threshold, classifier hash, and LightGlue revision are
pinned. Do not change them without recalibration/parity validation.
The verifier classifier is `image_matching/matching/classifier.pkl` in https://github.com/HPI-Information-Systems/smARTmatch.
Save that file as `deduplication/classifier.pkl` before verification.

## One-command set matching

`match-sets` runs ingestion, an exact-SHA precheck, DINO embedding/retrieval, LightGlue verification,
and review preparation in one resumable invocation:

```bash
python -m deduplication match-sets \
  --set-a-path /data/queries \
  --set-b-path /data/references \
  --work-dir /cluster/work/query-reference-overlap \
  --config config/deduplication.pc995.toml \
  --workers 10 \
  --reuse-dino-work-dir /cluster/work/previous-museum-dedup \
  --reuse-superpoint-work-dir /cluster/work/previous-museum-dedup \
  --node-local-cache-dir "$TMPDIR"
```

Canonical filesystem identity selects the topology. When both paths identify the same directory,
matching is symmetric and self-pairs are excluded. Different, non-nested paths use bipartite mode:
only set A queries are compared with set B references. Retrieval retains every calibrated-threshold
match **or** each query's nearest `min_retrieval_window` images (200 in the supplied configs).
Exact byte matches are candidates regardless of DINO score. Perceptual hashing is intentionally
deferred until a Hamming threshold is calibrated.

Before pair verification, the command precomputes one SuperPoint artifact per unique image SHA.
Bipartite candidates are sharded and grouped by the set-B reference: every worker keeps all set-A
query features resident on its GPU and loads each assigned set-B feature once. Worker schedule files
under `verification/schedules/` preserve that reference affinity and avoid repeatedly scanning the
full candidate CSV. `--node-local-cache-dir` places the content-addressed feature cache on local
scratch; omit it when several independently launched nodes must share one cache.

The reuse options import artifacts only after validating their image SHA and model/extractor
fingerprints. They accept the original museum deduplication work directory even though the new run
uses different `set_b/...` image IDs. DINO retrieval automatically uses exact float32 CUDA matrix
multiplication when a GPU is visible (TF32 is disabled), with a NumPy fallback for CPU tests.

The Wikidata query corpus is exported separately with
`experiments/datasets/wikimedia/export_wikidata_dedup_queries.py`; the generic matcher has no
Wikidata/database-specific logic.

### Full Wikidata → museum B200 job

`match_wikidata_full_museum_h100.sbatch` runs all 2,208 supported image files from the full
2,209-entry Wikidata image directory against the 12,933 retained museum originals on one B200 with
160GB host RAM. It uses 16 reference-affine LightGlue workers, prior museum artifact reuse, and
node-local SuperPoint storage. The job sources `HF_TOKEN` from the repository-local `.env` and is
pinned to the `lgm` Conda environment. Submit it from `data/work` as
`sbatch experiments/datasets/deduplication/match_wikidata_full_museum_h100.sbatch`;
bootstrap and runtime logs are written under the deduplication directory's `slurm-logs/`.

## Staged workflow

Run commands from this directory so the local package is importable. By default outputs use the
project's `work/` directory. On a cluster, select a run directory on high-capacity storage and
pass the same option to every command:

```bash
WORK_ARGS=(--work-dir "data/datasets/museum_deduplication_smoke")
```

The directory is created when the run starts, while manifests retain portable `work/...` image
paths.

```bash
python -m deduplication ingest "${WORK_ARGS[@]}" \
  --source-dir metmuseum=/data/metmuseum \
  --source-dir artic=/data/artic \
  --config config/deduplication.toml
python -m deduplication embed "${WORK_ARGS[@]}" --config config/deduplication.toml
python -m deduplication retrieve "${WORK_ARGS[@]}" --config config/deduplication.toml
python -m deduplication precompute-features "${WORK_ARGS[@]}" \
  --config config/deduplication.toml --workers 10
python -m deduplication verify "${WORK_ARGS[@]}" --config config/deduplication.toml
python -m deduplication prepare-review "${WORK_ARGS[@]}"
python -m deduplication review "${WORK_ARGS[@]}" --reviewer "$USER" --port 8080
python -m deduplication finalize-review "${WORK_ARGS[@]}" --require-complete

# Either retain one keeper per accepted connected identity component:
python -m deduplication build-clusters "${WORK_ARGS[@]}" --config config/deduplication.toml
python -m deduplication prepare-dataset-export "${WORK_ARGS[@]}"

# Or remove a certified minimum set touching every accepted pair:
python -m deduplication select-removals "${WORK_ARGS[@]}" --workers 16
python -m deduplication prepare-dataset-export "${WORK_ARGS[@]}" \
  --selection minimum-vertex-cover

python -m deduplication export-dataset "${WORK_ARGS[@]}"
python -m deduplication status "${WORK_ARGS[@]}"
```

`--source-dir` is repeatable and accepts either `PATH` (directory basename becomes the source name)
or `NAME=PATH`. Ingestion flattens supported images into `work/images/` as
`<source>_<original-basename>`. It reports all basename collisions before copying anything.

Review binds to `127.0.0.1` by default. Use SSH port forwarding for remote review rather than
exposing the unauthenticated server. The UI presents connected components of the immutable review
pair graph as thumbnail galleries, ordered by decreasing image count. Accepting a gallery accepts
all of its proposal edges, so clustering keeps one image and removes the rest; rejecting it rejects
all proposal edges and keeps the images separate. Every gallery image also has controls to accept or
reject all proposal edges in that cluster which touch that specific image. A cluster-wide reset
button returns every edge in the displayed cluster to pending. Bulk actions append one ordinary
decision event per affected edge to `work/review/review_decisions.csv`, overriding earlier decisions without deleting their
audit history. Gallery thumbnails are bounded, lazy-loaded, and disposable; click one to open the
original image. Cluster review is the default `/` route; the original one-pair-at-a-time interface
remains available at `/pairs`, and both interfaces update the same append-only decision log.
`resolve-pending --decision accepted` can explicitly finish a review by appending accepted events
for every pending pair before finalization.

`select-removals` solves an exact minimum vertex cover over finalized accepted edges. It stores
resumable, hash-keyed component solutions under `optimization/minimum_vertex_cover_parts/` and
writes a certified selection CSV plus JSON sidecar. The corresponding export mode retains every
image outside the cover as its own class and records a direct retained witness for each exclusion.

If `cluster_conflicts.csv` is non-empty, resolve the contradictory pair decisions, finalize review
again, and rebuild clusters. Keeper overrides can be supplied with
`--overrides work/keeper_overrides.csv` using `cluster_id,keeper_image_id` columns.

### Distributed verification

Verification supports several model workers per visible GPU and deterministic shards across
nodes. Every node must see the same work directory and candidate CSV. For two nodes/GPUs with ten
workers each, run concurrently:

```bash
# node/GPU 0
python -m deduplication verify "${WORK_ARGS[@]}" --config "$CONFIG" \
  --workers 10 --num-shards 2 --shard-index 0

# node/GPU 1
python -m deduplication verify "${WORK_ARGS[@]}" --config "$CONFIG" \
  --workers 10 --num-shards 2 --shard-index 1
```

After both shard summaries report complete, merge on a CPU node:

```bash
python -m deduplication verify "${WORK_ARGS[@]}" --config "$CONFIG" \
  --num-shards 2 --merge-only
```

Pair IDs are hash-partitioned, so shards are disjoint and balanced. Parts are independently
resumable. Workers share atomic content-keyed feature files while retaining only a bounded CPU LRU
(`--feature-cache-size`, default 32) to avoid multiplying the full feature corpus by worker count.
Do not launch two commands with the same shard index.

## Resumability and tracked artifacts

`match-sets` records stage state in `pipeline_state.json` and uses a work-directory lock. A failed
invocation preserves finalized artifacts; rerunning the same command validates and reuses them.
If review decisions exist, it refuses to replace a changed review queue.

Embedding, retrieval, and verification use finalized parts. Their heavy recomputable part/cache
directories and the review thumbnail cache are the only ignored work artifacts. Shared image
copies, merged results, decisions, clusters, manifests, and final datasets remain visible to Git.
Failed verification rows are retried
on the next `verify` invocation.

Named external work directories are isolated from one another, allowing smoke and full runs to
coexist. External output is not automatically tracked by the repository, so back up merged CSVs
and especially `review/review_decisions.csv` separately.

Never delete or modify `work/images/` after review starts. Content hashes bind embeddings,
candidates, decisions, clusters, and export manifests to the exact copied bytes.

## Tests

CPU-only deterministic stages and mocked embedding behavior run locally:

```bash
PYTHONPATH=. python -m pytest -q tests
```

Real DINO/LightGlue parity and the full corpus run must be performed on the B200/A100 environment.
