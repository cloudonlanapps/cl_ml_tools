# cl_ml_tools

A Python library for building master-worker media processing / ML systems. Provides task plugins, worker runtime, and protocols for job persistence and file storage.

## Features

- **Plugin-based architecture** - Add new media processing / ML tasks without modifying core code
- **Protocol-driven design** - Implement `JobRepository` and `FileStorage` to integrate with any database/storage
- **Dynamic discovery** - Plugins auto-registered via Python entry points
- **Race-condition safe** - Atomic job claiming prevents duplicate processing
- **FastAPI integration** - Auto-generates REST endpoints for job creation

## Installation

```bash
pip install cl_ml_tools

# With FastAPI support (for master/API server)
pip install cl_ml_tools[master]

# With compute plugins (image processing, etc.)
pip install cl_ml_tools[compute]

# All extras
pip install cl_ml_tools[master,compute]
```

## Quick Start

### 1. Implement the Protocols

You need to implement two protocols to integrate with your system:

#### JobRepository Protocol

```python
from typing import Optional, List
from cl_ml_tools.common.schemas import Job

class SQLiteJobRepository:
    """SQLite implementation of JobRepository protocol."""
    
    def __init__(self, db_path: str):
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                task_output TEXT,
                error_message TEXT,
                created_by TEXT,
                priority INTEGER DEFAULT 5,
                created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000)
            )
        """)
        self.conn.commit()
    
    def add_job(
        self, 
        job: Job, 
        created_by: Optional[str] = None, 
        priority: Optional[int] = None
    ) -> str:
        """Save job to database."""
        import json
        self.conn.execute(
            """INSERT INTO jobs (job_id, task_type, params, status, created_by, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (job.job_id, job.task_type, json.dumps(job.params), 
             job.status, created_by, priority or 5)
        )
        self.conn.commit()
        return job.job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        import json
        cursor = self.conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        )
        row = cursor.fetchone()
        if row:
            return Job(
                job_id=row['job_id'],
                task_type=row['task_type'],
                params=json.loads(row['params']),
                status=row['status'],
                progress=row['progress'],
                task_output=json.loads(row['task_output']) if row['task_output'] else None,
                error_message=row['error_message']
            )
        return None
    
    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job fields."""
        import json
        if 'task_output' in kwargs and kwargs['task_output'] is not None:
            kwargs['task_output'] = json.dumps(kwargs['task_output'])
        
        set_clause = ', '.join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [job_id]
        
        cursor = self.conn.execute(
            f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values
        )
        self.conn.commit()
        return cursor.rowcount > 0
    
    def fetch_next_job(self, task_types: List[str]) -> Optional[Job]:
        """
        Atomically claim next queued job.
        
        Uses BEGIN IMMEDIATE to acquire write lock, preventing race conditions
        when multiple workers are running.
        """
        import json
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        
        try:
            placeholders = ','.join('?' * len(task_types))
            cursor.execute(f"""
                SELECT * FROM jobs 
                WHERE status = 'queued' AND task_type IN ({placeholders})
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """, task_types)
            
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE jobs SET status = 'processing' WHERE job_id = ?",
                    (row['job_id'],)
                )
                self.conn.commit()
                return Job(
                    job_id=row['job_id'],
                    task_type=row['task_type'],
                    params=json.loads(row['params']),
                    status='processing',  # Return with updated status
                    progress=row['progress'],
                    task_output=json.loads(row['task_output']) if row['task_output'] else None,
                    error_message=row['error_message']
                )
            
            self.conn.rollback()
            return None
        except Exception:
            self.conn.rollback()
            raise
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job from database."""
        cursor = self.conn.execute(
            "DELETE FROM jobs WHERE job_id = ?", (job_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0
```

#### FileStorage Protocol

```python
from pathlib import Path
import hashlib
import shutil

class LocalFileStorage:
    """Local filesystem implementation of FileStorage protocol."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_job_directory(self, job_id: str) -> Path:
        """Create job directory with input/output subdirs."""
        job_dir = self.base_dir / "jobs" / job_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        (job_dir / "output").mkdir(parents=True, exist_ok=True)
        return job_dir
    
    def get_input_path(self, job_id: str) -> Path:
        """Get absolute path to job's input directory."""
        return self.base_dir / "jobs" / job_id / "input"
    
    def get_output_path(self, job_id: str) -> Path:
        """Get absolute path to job's output directory."""
        return self.base_dir / "jobs" / job_id / "output"
    
    async def save_input_file(self, job_id: str, filename: str, file) -> dict:
        """Save uploaded file to job's input directory."""
        input_dir = self.get_input_path(job_id)
        file_path = input_dir / filename
        
        # Read and save file
        content = await file.read()
        file_path.write_bytes(content)
        
        # Calculate hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        return {
            "filename": filename,
            "path": str(file_path),  # Absolute path
            "size": len(content),
            "hash": file_hash
        }
    
    def cleanup_job(self, job_id: str) -> bool:
        """Delete job directory and all files."""
        job_dir = self.base_dir / "jobs" / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
            return True
        return False
```

### 2. Setup Master (API Server)

```python
from fastapi import FastAPI
from cl_ml_tools.master import create_master_router

# Your protocol implementations
from my_app.repository import SQLiteJobRepository
from my_app.storage import LocalFileStorage

app = FastAPI(title="ML Tools")

# Initialize implementations
repository = SQLiteJobRepository("./jobs.db")
file_storage = LocalFileStorage("./media_storage")

# Auth dependency (implement your own)
async def get_current_user():
    return None  # Or return user object from JWT/session

# Mount all plugin routes (auto-discovered from entry points)
app.include_router(
    create_master_router(repository, file_storage, get_current_user),
    prefix="/api"
)

# Optional: Add job status endpoint
@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.model_dump()
```

Run the server:
```bash
uvicorn my_app.main:app --reload
```

> **Note**: Available API endpoints are auto-generated from registered plugins. See `pyproject.toml` entry points and `src/cl_ml_tools/plugins/` for available task types.

### 3. Setup Worker

```python
import asyncio
from cl_ml_tools.worker import Worker

# Same repository as master (shared database)
from my_app.repository import SQLiteJobRepository

repository = SQLiteJobRepository("./jobs.db")
worker = Worker(repository)

async def run_worker():
    """Process jobs forever."""
    print(f"Worker started. Supported tasks: {worker.get_supported_task_types()}")
    
    while True:
        # Process one job (or all supported types if None)
        processed = await worker.run_once()
        
        if not processed:
            # No jobs available, wait before polling again
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(run_worker())
```

Run the worker:
```bash
python -m my_app.worker
```

## Creating Custom Plugins

### 1. Create Plugin Structure

```
my_app/plugins/watermark/
├── __init__.py
├── schema.py
├── task.py
└── routes.py
```

### 2. Define Parameters Schema

```python
# schema.py
from cl_ml_tools.common.schemas import BaseJobParams

class WatermarkParams(BaseJobParams):
    watermark_text: str
    position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right
    opacity: float = 0.5
```

### 3. Implement Task

```python
# task.py
from cl_ml_tools.common.compute_module import ComputeModule
from cl_ml_tools.common.schemas import Job
from .schema import WatermarkParams

class WatermarkTask(ComputeModule):
    @property
    def task_type(self) -> str:
        return "watermark"
    
    def get_schema(self):
        return WatermarkParams
    
    async def execute(self, job: Job, params: WatermarkParams, progress_callback=None):
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            for i, (input_path, output_path) in enumerate(
                zip(params.input_paths, params.output_paths)
            ):
                # Load image
                img = Image.open(input_path)
                
                # Add watermark
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), params.watermark_text, fill=(255, 255, 255, 128))
                
                # Save
                img.save(output_path)
                
                # Report progress
                if progress_callback:
                    progress_callback(int((i + 1) / len(params.input_paths) * 100))
            
            return {
                "status": "ok",
                "task_output": {"processed_files": params.output_paths}
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

### 4. Create Route Factory

```python
# routes.py
from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Callable
from uuid import uuid4

from cl_ml_tools.common.job_repository import JobRepository
from cl_ml_tools.common.file_storage import FileStorage
from cl_ml_tools.common.schemas import Job

def create_router(
    repository: JobRepository,
    file_storage: FileStorage,
    get_current_user: Callable
) -> APIRouter:
    router = APIRouter()

    @router.post("/jobs/watermark")
    async def create_watermark_job(
        file: UploadFile = File(...),
        watermark_text: str = Form(...),
        position: str = Form("bottom-right"),
        opacity: float = Form(0.5),
        priority: int = Form(5),
        user = Depends(get_current_user)
    ):
        job_id = str(uuid4())
        
        file_storage.create_job_directory(job_id)
        file_info = await file_storage.save_input_file(job_id, file.filename, file)
        
        input_path = file_info["path"]
        output_path = str(file_storage.get_output_path(job_id) / f"watermarked_{file.filename}")
        
        job = Job(
            job_id=job_id,
            task_type="watermark",
            params={
                "input_paths": [input_path],
                "output_paths": [output_path],
                "watermark_text": watermark_text,
                "position": position,
                "opacity": opacity
            }
        )
        repository.add_job(job, created_by=user.id if user else None, priority=priority)
        
        return {"job_id": job_id, "status": "queued"}

    return router
```

### 5. Register Plugin

Add to your `pyproject.toml`:

```toml
[project.entry-points."cl_ml_tools.tasks"]
watermark = "my_app.plugins.watermark.task:WatermarkTask"

[project.entry-points."cl_ml_tools.routes"]
watermark = "my_app.plugins.watermark.routes:create_router"
```

The plugin is now auto-discovered by both master and worker!

## Database-Specific Implementation Notes

### PostgreSQL / MySQL

Use `FOR UPDATE SKIP LOCKED` for efficient concurrent job claiming:

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

def fetch_next_job(self, task_types: List[str], session: Session) -> Optional[Job]:
    stmt = (
        select(JobModel)
        .where(JobModel.status == 'queued')
        .where(JobModel.task_type.in_(task_types))
        .order_by(JobModel.priority.desc(), JobModel.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)  # Skip locked rows
    )
    
    job = session.execute(stmt).scalar_one_or_none()
    if job:
        job.status = 'processing'
        session.commit()
        return Job.model_validate(job)
    return None
```

### SQLite

Use `BEGIN IMMEDIATE` for write lock (shown in Quick Start example above).

> **Note**: SQLite locks the entire database, not individual rows. For high concurrency, use PostgreSQL or MySQL in production.

## Available Plugins

See [`src/cl_ml_tools/plugins/`](src/cl_ml_tools/plugins/) for built-in plugins and their documentation.

Registered plugins are listed in [`pyproject.toml`](pyproject.toml) under `[project.entry-points."cl_ml_tools.tasks"]`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                        │
├─────────────────────────────────────────────────────────────┤
│  SQLiteJobRepository implements JobRepository               │
│  LocalFileStorage implements FileStorage                    │
│  get_current_user() for authentication                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      cl_ml_tools                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Protocols:          Base Class:                            │
│  ┌──────────────┐    ┌────────────────────────────────┐    │
│  │JobRepository │    │ ComputeModule (ABC)            │    │
│  │FileStorage   │    │ - task_type                    │    │
│  └──────────────┘    │ - get_schema()                 │    │
│                      │ - execute(job, params, cb)     │    │
│                      └────────────────────────────────┘    │
│                                   ▲                         │
│  Plugins:                         │ extends                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ See src/cl_ml_tools/plugins/ for available tasks│    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Runtime:                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Worker - orchestrates job execution                 │   │
│  │ create_master_router() - aggregates plugin routes   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your plugin to `src/cl_ml_tools/plugins/`
4. Register entry points in `pyproject.toml`
5. Add a README.md in your plugin directory documenting parameters
6. Submit a pull request
