"""SQLite dataset accessors used by experiment configs and splits."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

LOCAL_STATUSES = {"downloaded", "generated"}


@dataclass(frozen=True)
class ImageRecord:
    file_id: str
    class_id: int
    local_rel_path: str | None
    download_status: str
    source_role: str | None
    tags: tuple[str, ...]

    def absolute_path(self, dataset_db: Path, image_root: Path | None = None) -> Path | None:
        if not self.local_rel_path or self.download_status not in LOCAL_STATUSES:
            return None
        db_relative = dataset_db.resolve().parent / self.local_rel_path
        if db_relative.exists() or image_root is None:
            return db_relative
        return image_root / Path(self.local_rel_path).name


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    required = {"classes", "image_files", "image_file_classes", "splits"}
    found = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = sorted(required - found)
    if missing:
        raise ValueError(f"Dataset DB missing required tables: {missing}")


def dataset_counts(db_path: Path | str) -> dict[str, int]:
    with connect(db_path) as conn:
        ensure_schema(conn)
        return {
            "num_classes": int(conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]),
            "num_images": int(conn.execute("SELECT COUNT(*) FROM image_files").fetchone()[0]),
        }


def split_exists(db_path: Path | str, split_name: str) -> bool:
    with connect(db_path) as conn:
        ensure_schema(conn)
        row = conn.execute("SELECT 1 FROM splits WHERE split_name=? LIMIT 1", (split_name,)).fetchone()
        return row is not None


def _tags(value: str | None) -> tuple[str, ...]:
    return tuple(sorted({part.strip() for part in str(value or "").split(",") if part.strip()}))


def fetch_same_painting_records(
    db_path: Path | str,
    file_ids: list[str] | tuple[str, ...] | None = None,
    split_name: str | None = None,
    statuses: set[str] | None = LOCAL_STATUSES,
) -> dict[str, ImageRecord]:
    clauses = ["ifc.primary_label='same_painting'"]
    params: list[object] = []
    if file_ids is not None:
        if not file_ids:
            return {}
        clauses.append("f.file_id IN ({})".format(",".join("?" for _ in file_ids)))
        params.extend(file_ids)
    if split_name is not None:
        clauses.append("EXISTS(SELECT 1 FROM splits s WHERE s.file_id=f.file_id AND s.split_name=?)")
        params.append(split_name)
    if statuses is not None:
        clauses.append("f.download_status IN ({})".format(",".join("?" for _ in statuses)))
        params.extend(sorted(statuses))
    where = " AND ".join(clauses)
    sql = f"""
        SELECT f.file_id, f.local_rel_path, f.download_status, f.source_role,
               ifc.class_id, COALESCE(tags.tags_csv, '') AS tags_csv
        FROM image_files f
        JOIN image_file_classes ifc ON ifc.file_id=f.file_id
        LEFT JOIN (
          SELECT image_file_class_id, GROUP_CONCAT(tag, ',') AS tags_csv
          FROM image_file_class_tags GROUP BY image_file_class_id
        ) tags ON tags.image_file_class_id=ifc.image_file_class_id
        WHERE {where}
        ORDER BY f.file_id
    """
    with connect(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql, params).fetchall()
    return {
        str(row["file_id"]): ImageRecord(
            file_id=str(row["file_id"]),
            class_id=int(row["class_id"]),
            local_rel_path=row["local_rel_path"],
            download_status=str(row["download_status"] or ""),
            source_role=row["source_role"],
            tags=_tags(row["tags_csv"]),
        )
        for row in rows
    }


def ids_from_records(records: dict[str, ImageRecord]) -> list[str]:
    return sorted(records)


def ids_for_split(db_path: Path | str, split_name: str) -> list[str]:
    return ids_from_records(fetch_same_painting_records(db_path, split_name=split_name))


def all_same_painting_ids(db_path: Path | str) -> list[str]:
    return ids_from_records(fetch_same_painting_records(db_path))


def ids_for_role_or_tag(
    db_path: Path | str,
    role_token: str,
    tag: str,
    role_aliases: tuple[str, ...] = (),
) -> list[str]:
    tokens = {role_token.strip().lower(), *(alias.strip().lower() for alias in role_aliases)}
    records = fetch_same_painting_records(db_path)
    tag_matches = sorted(file_id for file_id, record in records.items() if tag in record.tags)
    if tag_matches:
        return tag_matches
    return sorted(
        file_id
        for file_id, record in records.items()
        if bool(tokens & {part.strip() for part in str(record.source_role or "").lower().split(",") if part.strip()})
    )


def class_to_image_ids(db_path: Path | str) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for file_id, record in fetch_same_painting_records(db_path).items():
        grouped.setdefault(record.class_id, []).append(file_id)
    return {class_id: sorted(ids) for class_id, ids in sorted(grouped.items())}
