"""cl_ml_tools - Tools for master-worker media processing / ML ."""

from .common.compute_module import ComputeModule
from .common.job_repository import JobRepository
from .common.job_storage import AsyncFileLike, FileLike, JobStorage, SavedJobFile
from .common.schema_job import BaseJobParams, Job
from .common.schema_job_record import (
    JobRecord,
    JobRecordUpdate,
    JobStatus,
    TaskOutputRecord,
    TaskParamsRecord,
)
from .master import create_master_router
from .plugins.face_detection.schema import BBox, FaceLandmarks
from .utils.mqtt import (
    BroadcasterBase,
    MQTTBroadcaster,
    NoOpBroadcaster,
    get_broadcaster,
    shutdown_broadcaster,
)
from .worker import Worker

__version__ = "0.1.0"

__all__ = [
    "Job",
    "BaseJobParams",
    "JobRecord",
    "JobRecordUpdate",
    "TaskOutputRecord",
    "TaskParamsRecord",
    "JobStatus",
    "AsyncFileLike",
    "FileLike",
    "SavedJobFile",
    "ComputeModule",
    "JobRepository",
    "JobStorage",
    "__version__",
    "Worker",
    "BroadcasterBase",
    "MQTTBroadcaster",
    "NoOpBroadcaster",
    "create_master_router",
    "get_broadcaster",
    "shutdown_broadcaster",
    "BBox",
    "FaceLandmarks",
]
