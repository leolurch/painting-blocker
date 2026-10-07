"""Command-line interface for the standalone deduplication stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import file_io
from .config import load_config, write_resolved_config, write_run_metadata

DEFAULT_CONFIG = file_io.PROJECT_DIR / "config" / "deduplication.toml"


def _add_common(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    if config:
        parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--work-dir", type=Path, default=file_io.PROJECT_DIR / "work",
        help="Work/output directory for this run (default: deduplication project/work)",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone museum-image deduplication pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    _add_common(ingest)
    ingest.add_argument("--source-dir", action="append", required=True, help="PATH or NAME=PATH; repeatable")

    embed_command = sub.add_parser("embed")
    _add_common(embed_command)
    embed_command.add_argument("--reuse-dino-work-dir", type=Path, action="append", default=[])
    embed_command.add_argument("--num-shards", type=int, default=1)
    embed_command.add_argument("--shard-index", type=int)
    embed_command.add_argument("--merge-only", action="store_true")
    retrieve_command = sub.add_parser("retrieve")
    _add_common(retrieve_command)
    retrieve_command.add_argument("--mode", choices=("symmetric", "bipartite"), default="symmetric")

    match_sets = sub.add_parser("match-sets")
    _add_common(match_sets)
    match_sets.add_argument("--set-a-path", type=Path, required=True)
    match_sets.add_argument("--set-b-path", type=Path, required=True)
    match_sets.add_argument("--workers", type=int, default=1, help="LightGlue workers on this node/GPU")
    match_sets.add_argument("--part-size", type=int, default=250)
    match_sets.add_argument("--feature-cache-size", type=int, default=32)
    match_sets.add_argument("--reuse-dino-work-dir", type=Path, action="append", default=[])
    match_sets.add_argument("--reuse-superpoint-work-dir", type=Path, action="append", default=[])
    match_sets.add_argument("--node-local-cache-dir", type=Path)

    precompute = sub.add_parser("precompute-features")
    _add_common(precompute)
    precompute.add_argument("--workers", type=int, default=1)
    precompute.add_argument("--feature-dir", type=Path)
    precompute.add_argument("--reuse-superpoint-work-dir", type=Path, action="append", default=[])
    precompute.add_argument("--num-shards", type=int, default=1)
    precompute.add_argument("--shard-index", type=int)

    verify = sub.add_parser("verify")
    _add_common(verify)
    verify.add_argument("--workers", type=int, default=1, help="Model worker processes on this node/GPU")
    verify.add_argument("--num-shards", type=int, default=1, help="Total deterministic shards across nodes")
    verify.add_argument("--shard-index", type=int, help="Zero-based shard assigned to this invocation")
    verify.add_argument("--merge-only", action="store_true", help="Merge completed shard outputs without loading models")
    verify.add_argument("--part-size", type=int, default=250)
    verify.add_argument("--feature-cache-size", type=int, default=32, help="CPU feature LRU entries per worker")
    verify.add_argument("--feature-dir", type=Path, help="Content-addressed SuperPoint cache (node-local is supported)")

    prepare = sub.add_parser("prepare-review")
    _add_common(prepare, config=False)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--include-negatives", action="store_true")

    server = sub.add_parser("review")
    _add_common(server, config=False)
    server.add_argument("--pairs", type=Path)
    server.add_argument("--decisions", type=Path)
    server.add_argument("--bind", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--reviewer", default="local")

    resolve = sub.add_parser("resolve-pending")
    _add_common(resolve, config=False)
    resolve.add_argument("--pairs", type=Path)
    resolve.add_argument("--decisions", type=Path)
    resolve.add_argument("--decision", required=True, choices=("accepted", "rejected"))
    resolve.add_argument("--reviewer", default="bulk-resolve-pending")

    finalize = sub.add_parser("finalize-review")
    _add_common(finalize, config=False)
    finalize.add_argument("--pairs", type=Path)
    finalize.add_argument("--decisions", type=Path)
    finalize.add_argument("--output", type=Path)
    finalize.add_argument("--require-complete", action="store_true")

    select = sub.add_parser("select-removals")
    _add_common(select, config=False)
    select.add_argument("--reviewed-pairs", type=Path)
    select.add_argument("--output", type=Path)
    select.add_argument("--workers", type=int, default=16)
    select.add_argument("--time-limit", type=float, default=900.0, help="Maximum seconds per component")

    clusters = sub.add_parser("build-clusters")
    _add_common(clusters)
    clusters.add_argument("--images", type=Path)
    clusters.add_argument("--reviewed-pairs", type=Path)
    clusters.add_argument("--output", type=Path)
    clusters.add_argument("--overrides", type=Path)

    prepare_export = sub.add_parser("prepare-dataset-export")
    _add_common(prepare_export, config=False)
    prepare_export.add_argument(
        "--selection", choices=("clusters", "minimum-vertex-cover"), default="clusters"
    )
    prepare_export.add_argument("--clusters", type=Path)
    prepare_export.add_argument("--cover", type=Path)
    prepare_export.add_argument("--manifest", type=Path)
    prepare_export.add_argument("--excluded", type=Path)

    export = sub.add_parser("export-dataset")
    _add_common(export, config=False)
    export.add_argument("--manifest", type=Path)
    export.add_argument("--excluded", type=Path)
    export.add_argument("--output", type=Path)

    status = sub.add_parser("status")
    _add_common(status, config=False)
    return parser


def _work_path(value: Path | None, relative: str) -> Path:
    return value.expanduser().resolve() if value is not None else file_io.WORK_DIR / relative


def pipeline_status() -> dict[str, Any]:
    work = file_io.WORK_DIR

    def csv_count(path: Path) -> int | None:
        return sum(1 for _ in file_io.iter_csv_rows(path)) if path.is_file() else None

    cluster_path = work / "clusters.csv"
    cluster_rows = file_io.read_csv_rows(cluster_path) if cluster_path.is_file() else None
    result: dict[str, Any] = {
        "valid_images": csv_count(work / "images.csv"),
        "invalid_images": csv_count(work / "invalid_images.csv"),
        "exact_hash_candidates": csv_count(work / "hashing" / "exact_hash_candidates.csv"),
        "embeddings_merged": (work / "embeddings" / "embeddings.npy").is_file(),
        "superpoint_precomputed": (work / "verification" / "feature_precompute_summary.json").is_file(),
        "embedding_parts": len(list((work / "embeddings" / "parts").glob("part_*.npy"))),
        "candidates": csv_count(work / "candidates" / "candidates.csv"),
        "verification": csv_count(work / "verification" / "verification.csv"),
        "review_pairs": csv_count(work / "review" / "review_pairs.csv"),
        "reviewed_pairs": csv_count(work / "review" / "reviewed_pairs.csv"),
        "clusters": len({row["cluster_id"] for row in cluster_rows}) if cluster_rows is not None else None,
        "cluster_conflicts": csv_count(work / "cluster_conflicts.csv"),
        "keepers": csv_count(work / "dataset_manifest.csv"),
        "excluded": csv_count(work / "excluded_images.csv"),
        "dataset_exported": (work / "dataset" / "dataset.db").is_file(),
    }
    pipeline_state = work / "pipeline_state.json"
    if pipeline_state.is_file():
        result["pipeline_state"] = file_io.read_json(pipeline_state)
    candidate_summary = work / "candidates" / "summary.json"
    if candidate_summary.is_file():
        result["candidate_summary"] = file_io.read_json(candidate_summary)
    feature_summary = work / "verification" / "feature_precompute_summary.json"
    if feature_summary.is_file():
        result["feature_precompute_summary"] = file_io.read_json(feature_summary)
    verification_summary = work / "verification" / "summary.json"
    if verification_summary.is_file():
        result["verification_summary"] = file_io.read_json(verification_summary)
    decisions = work / "review" / "review_decisions.csv"
    pairs = work / "review" / "review_pairs.csv"
    if pairs.is_file():
        from .review import replay_decisions

        pair_ids = {row["pair_id"] for row in file_io.read_csv_rows(pairs)}
        states = replay_decisions(decisions, pair_ids)
        result["review_progress"] = {
            "accepted": sum(row["decision"] == "accepted" for row in states.values()),
            "rejected": sum(row["decision"] == "rejected" for row in states.values()),
            "pending": len(pair_ids) - sum(row["decision"] in {"accepted", "rejected"} for row in states.values()),
            "total": len(pair_ids),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    file_io.configure_work_dir(args.work_dir)
    command = args.command
    summary: Any
    if command == "status":
        summary = pipeline_status()
    elif command == "prepare-review":
        from .review import prepare_review

        summary = prepare_review(
            _work_path(args.output, "review/review_pairs.csv"),
            include_negatives=args.include_negatives,
        )
    elif command == "review":
        from .review import create_app

        create_app(
            _work_path(args.pairs, "review/review_pairs.csv"),
            _work_path(args.decisions, "review/review_decisions.csv"),
            args.reviewer,
        ).run(
            host=args.bind, port=args.port, debug=False, threaded=True
        )
        return 0
    elif command == "resolve-pending":
        from .review import resolve_pending

        summary = resolve_pending(
            _work_path(args.pairs, "review/review_pairs.csv"),
            _work_path(args.decisions, "review/review_decisions.csv"),
            args.decision,
            args.reviewer,
        )
    elif command == "finalize-review":
        from .review import finalize_review

        summary = finalize_review(
            _work_path(args.pairs, "review/review_pairs.csv"),
            _work_path(args.decisions, "review/review_decisions.csv"),
            _work_path(args.output, "review/reviewed_pairs.csv"),
            require_complete=args.require_complete,
        )
    elif command == "select-removals":
        from .selection import select_minimum_vertex_cover

        summary = select_minimum_vertex_cover(
            file_io.WORK_DIR / "images.csv",
            _work_path(args.reviewed_pairs, "review/reviewed_pairs.csv"),
            _work_path(args.output, "optimization/minimum_vertex_cover.csv"),
            workers=args.workers,
            time_limit=args.time_limit,
        )
    elif command == "prepare-dataset-export":
        manifest_path = _work_path(args.manifest, "dataset_manifest.csv")
        excluded_path = _work_path(args.excluded, "excluded_images.csv")
        if args.selection == "minimum-vertex-cover":
            from .selection import prepare_vertex_cover_export

            summary = prepare_vertex_cover_export(
                _work_path(args.cover, "optimization/minimum_vertex_cover.csv"),
                manifest_path,
                excluded_path,
            )
        else:
            if args.cover is not None:
                raise ValueError("--cover requires --selection minimum-vertex-cover")
            from .clustering import prepare_dataset_export

            summary = prepare_dataset_export(
                _work_path(args.clusters, "clusters.csv"), manifest_path, excluded_path
            )
    elif command == "export-dataset":
        # Lazy import preserves the invariant that only this command imports sqlite3.
        from .export_dataset import export_dataset

        summary = export_dataset(
            _work_path(args.manifest, "dataset_manifest.csv"),
            _work_path(args.excluded, "excluded_images.csv"),
            _work_path(args.output, "dataset"),
        )
    else:
        config = load_config(args.config)
        file_io.WORK_DIR.mkdir(parents=True, exist_ok=True)
        sharded_command = (
            command in {"verify", "embed", "precompute-features"}
            and getattr(args, "num_shards", 1) > 1
        )
        concurrent_shard = sharded_command and not getattr(args, "merge_only", False)
        if not concurrent_shard:
            write_resolved_config(config, file_io.WORK_DIR)
        if command == "match-sets":
            from .pipeline import match_sets

            summary = match_sets(
                config,
                args.set_a_path,
                args.set_b_path,
                workers=args.workers,
                part_size=args.part_size,
                feature_cache_size=args.feature_cache_size,
                reuse_dino_work_dirs=args.reuse_dino_work_dir,
                reuse_superpoint_work_dirs=args.reuse_superpoint_work_dir,
                node_local_cache_dir=args.node_local_cache_dir,
            )
        elif command == "precompute-features":
            from .verification import precompute_superpoint_features

            if args.num_shards <= 0:
                raise ValueError("precompute --num-shards must be positive")
            if args.num_shards > 1 and args.shard_index is None:
                raise ValueError("precompute --shard-index is required when --num-shards > 1")
            summary = precompute_superpoint_features(
                config,
                workers=args.workers,
                feature_dir=args.feature_dir,
                reuse_work_dirs=args.reuse_superpoint_work_dir,
                num_shards=args.num_shards,
                shard_index=0 if args.shard_index is None else args.shard_index,
            )
            if args.num_shards == 1:
                write_run_metadata(config, file_io.WORK_DIR, {"feature_precompute_summary": summary})
        elif command == "ingest":
            from .ingest import ingest, parse_source_arg

            specs = [parse_source_arg(value) for value in args.source_dir]
            summary = ingest(config, specs)
            write_run_metadata(config, file_io.WORK_DIR, summary)
        elif command == "embed":
            from .embedding import embed

            if args.num_shards <= 0:
                raise ValueError("embed --num-shards must be positive")
            if args.merge_only and args.shard_index is not None:
                raise ValueError("embed --shard-index is not valid with --merge-only")
            if args.num_shards > 1 and args.shard_index is None and not args.merge_only:
                raise ValueError("embed --shard-index is required when --num-shards > 1")
            summary = embed(
                config,
                reuse_work_dirs=args.reuse_dino_work_dir,
                num_shards=args.num_shards,
                shard_index=0 if args.shard_index is None else args.shard_index,
                merge_only=args.merge_only,
            )
            if args.num_shards == 1 or args.merge_only:
                write_run_metadata(config, file_io.WORK_DIR, {"embedding_metadata": summary})
        elif command == "retrieve":
            from .retrieval import retrieve

            summary = retrieve(config, mode=args.mode)
            write_run_metadata(config, file_io.WORK_DIR, {"retrieval_summary": summary})
        elif command == "verify":
            from .verification import merge_verification, verify

            if args.num_shards <= 0 or args.workers <= 0 or args.part_size <= 0:
                raise ValueError("verify workers, num-shards, and part-size must be positive")
            if args.merge_only:
                if args.shard_index is not None:
                    raise ValueError("--shard-index is not valid with --merge-only")
                summary = merge_verification(config, num_shards=args.num_shards)
                write_run_metadata(config, file_io.WORK_DIR, {"verification_summary": summary})
            else:
                if args.shard_index is None:
                    if args.num_shards != 1:
                        raise ValueError("--shard-index is required when --num-shards > 1")
                    shard_index = 0
                else:
                    shard_index = args.shard_index
                summary = verify(
                    config,
                    workers=args.workers,
                    num_shards=args.num_shards,
                    shard_index=shard_index,
                    part_size=args.part_size,
                    feature_cache_size=args.feature_cache_size,
                    feature_dir=args.feature_dir,
                    merge=args.num_shards == 1,
                )
                if args.num_shards == 1:
                    write_run_metadata(config, file_io.WORK_DIR, {"verification_summary": summary})
        elif command == "build-clusters":
            from .clustering import build_clusters

            summary = build_clusters(
                config,
                _work_path(args.images, "images.csv"),
                _work_path(args.reviewed_pairs, "review/reviewed_pairs.csv"),
                _work_path(args.output, "clusters.csv"),
                overrides_path=args.overrides.expanduser().resolve() if args.overrides else None,
            )
        else:  # pragma: no cover
            raise AssertionError(command)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
