"""FastAPI router for the Drop-your-data feature."""

import os
import time
import threading
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException

from src.dropbox.pipeline import analyze
from src.dropbox.errors import (
    DropboxError,
    EmptyFileError,
    InvalidFileTypeError,
    FileSizeLimitError,
    NoTextColumnError,
    ColumnNotFoundError,
    SheetLimitError,
)

router = APIRouter()

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
_MODELS_LOAD_LOCK = threading.Lock()
_MODELS_INITIALIZED = False


def _ensure_models_loaded():
    """Ensure sentiment models and theme centroids are loaded once, with thread safety."""
    global _MODELS_INITIALIZED
    if not _MODELS_INITIALIZED:
        with _MODELS_LOAD_LOCK:
            if not _MODELS_INITIALIZED:
                print("\n[Dropbox Router] Initializing models on first request...")
                t0 = time.time()
                from src.dropbox.scorer import _get_models
                _get_models()
                t_scorer = time.time() - t0
                print(f"[Dropbox Router] Loaded TF-IDF vectorizer and Logistic Regression in {t_scorer:.4f}s")

                t1 = time.time()
                from src.dropbox.themes import _init_theme_centroids
                _init_theme_centroids()
                t_themes = time.time() - t1
                print(f"[Dropbox Router] Loaded 29 theme centroids and SentenceTransformer in {t_themes:.4f}s")

                _MODELS_INITIALIZED = True


@router.get("/limits")
def get_limits() -> Dict[str, Any]:
    """Return upload limits and file constraints for the dropbox feature."""
    return {
        "max_file_size_bytes": MAX_FILE_BYTES,
        "max_file_size_mb": 5.0,
        "max_rows": 3000,
        "max_columns": 50,
        "allowed_extensions": [".csv", ".xlsx"]
    }


@router.post("/analyze")
def analyze_dropbox(
    request: Request,
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None),
    include_themes: Optional[str] = Form("true")
) -> Dict[str, Any]:
    """Analyze an uploaded CSV or XLSX file.

    Synchronous def to ensure FastAPI executes in worker threadpool.
    Enforces 5 MB read cap, format checks, and safe error responses.
    """
    # 1. Reject early if Content-Length header exceeds 5 MB
    content_length_header = request.headers.get("content-length")
    if content_length_header:
        try:
            if int(content_length_header) > MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum allowed limit of 5.00 MB."
                )
        except ValueError:
            pass

    # 2. Check file extension
    filename = file.filename or "uploaded_data.csv"
    _, ext = os.path.splitext(filename.lower())
    if ext not in [".csv", ".xlsx"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{ext}'. Only .csv and .xlsx files are supported."
        )

    # 3. Read at most 5 MB + 1 byte
    chunk = file.file.read(MAX_FILE_BYTES + 1)
    if len(chunk) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds maximum allowed limit of 5.00 MB."
        )
    if len(chunk) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty (0 bytes)."
        )

    # 4. Lazy thread-safe model initialization
    _ensure_models_loaded()

    # Determine include_themes boolean flag (default True)
    inc_themes = True
    if include_themes is not None:
        inc_themes = str(include_themes).strip().lower() not in ["false", "0", "no"]

    # 5. Execute pipeline
    try:
        result = analyze(
            file_bytes=chunk,
            filename=filename,
            text_column=text_column,
            include_themes=inc_themes
        )
        return result
    except (EmptyFileError, NoTextColumnError, ColumnNotFoundError, SheetLimitError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeLimitError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except InvalidFileTypeError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        # Unexpected error: Return 500 with friendly message, NEVER a stack trace
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while analyzing the file."
        )
