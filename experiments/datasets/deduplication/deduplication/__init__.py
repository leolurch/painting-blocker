"""Standalone, resumable museum-image deduplication pipeline.

Stages: ingest -> embed -> retrieve -> verify -> prepare-review -> review ->
finalize-review -> build-clusters -> prepare-dataset-export -> export-dataset.

No SQLite database exists before the final ``export-dataset`` stage.
"""

__version__ = "0.1.0"
