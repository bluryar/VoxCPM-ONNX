import io
import sqlite3
import time
from typing import Optional, Tuple

import numpy as np


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ref_features (
                id TEXT PRIMARY KEY,
                prompt_text TEXT,
                patch_size INTEGER,
                dtype TEXT,
                created_at INTEGER,
                data BLOB
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_ref_features(db_path: str, feat_id: str, prompt_text: Optional[str], patch_size: int, dtype_str: str, patches: np.ndarray) -> None:
    # 确保表存在
    init_db(db_path)
    buf = io.BytesIO()
    # Store in npy format for portability (contains shape and dtype)
    np.save(buf, patches)
    blob = buf.getvalue()
    ts = int(time.time() * 1000)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO ref_features (id, prompt_text, patch_size, dtype, created_at, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (feat_id, prompt_text or "", int(patch_size), dtype_str, ts, sqlite3.Binary(blob)),
        )
        conn.commit()
    finally:
        conn.close()


def load_ref_features(db_path: str, feat_id: str) -> Tuple[np.ndarray, int, str, str]:
    # 确保表存在
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT prompt_text, patch_size, dtype, data FROM ref_features WHERE id = ?",
            (feat_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"feat_id not found: {feat_id}")
        prompt_text, patch_size, dtype_str, blob = row
        buf = io.BytesIO(blob)
        patches = np.load(buf)
        return patches, int(patch_size), str(prompt_text or ""), str(dtype_str)
    finally:
        conn.close()

__all__ = ["init_db", "save_ref_features", "load_ref_features"]