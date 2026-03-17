# Repository Analysis

## Scope
This analysis reviews the current codebase for structure, correctness, maintainability, and operational readiness.

## High-level architecture
- `face_attendance.py` contains the full computer-vision pipeline, persistence, and CLI modes (`register`/`recognize`).
- `api.py` exposes attendance data and export endpoints using FastAPI.
- Persistence is file-based (`attendance_v2.csv`, `encodings_v2.pickle`) with no locking or transactional guarantees.

## What is working well
- Attendance toggle model (`state`: 1/0) is straightforward and easy to consume via API.
- Legacy CSV compatibility is handled in `_detect_schema_and_read()`.
- Liveness + confidence buffering reduce accidental false positives.
- API includes practical export capability (`/export/excel`).

## Key risks and issues

### 1) API robustness for empty/missing CSV
`api.py` directly reads `attendance_v2.csv` in `/attendance` and `/live-status` without checking file existence or parser errors. Fresh deployments can fail with 500s.

### 2) Concurrency/data corruption risk
Both the vision process and API can read/write the same CSV and export file concurrently without file locks. This can produce partial reads or race conditions under load.

### 3) Inconsistent ID normalization
`face_attendance.py` normalizes IDs to strings while existing CSV content may include mixed dtypes. Although partially handled, mixed historical data still risks mismatch edge-cases.

### 4) Large monolithic script
`face_attendance.py` combines storage, liveness, CV inference, and UI rendering in one file. This limits testability and makes future changes higher-risk.

### 5) Documentation formatting issue
`README.md` uses `'''bash` blocks instead of fenced markdown code blocks (```` ```bash ````), reducing readability in markdown renderers.

## Suggested improvement plan (priority order)
1. Harden API reads with safe file handling (`exists`, `EmptyDataError`, default empty DataFrame).
2. Add simple file locking for CSV write/export operations.
3. Split `face_attendance.py` into modules (`storage.py`, `recognition.py`, `liveness.py`, `cli.py`).
4. Add unit tests for attendance state transitions and schema migration behavior.
5. Fix README markdown fences and add troubleshooting notes.

## Validation commands run
- `python -m py_compile api.py face_attendance.py`

