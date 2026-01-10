"""FastAPI routes for HLS streaming conversion plugin."""

import json
from typing import Annotated, Callable, TypedDict, cast

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.job_storage import JobStorage
from ...common.schema_job_record import JobCreatedResponse
from ...common.user import UserLike
from .schema import HLSStreamingOutput, HLSStreamingParams, VariantConfig


class VariantDict(TypedDict):
    resolution: int | None
    bitrate: int | None


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create router with injected dependencies."""
    router = APIRouter()

    @router.post("/jobs/hls_streaming", response_model=JobCreatedResponse)
    async def create_hls_job(
        file: Annotated[UploadFile, File(description="Video file to convert")],
        variants: Annotated[
            str,
            Form(description="JSON array of variants: [{resolution:720,bitrate:3500}]"),
        ] = '[{"resolution":720,"bitrate":3500},{"resolution":480,"bitrate":1500}]',
        include_original: Annotated[
            bool, Form(description="Include original quality")
        ] = False,
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        parsed = json.loads(variants)  # pyright: ignore[reportAny]
        variants_raw = cast(list[VariantDict], parsed)
        variant_models = [VariantConfig(**v) for v in variants_raw]

        return await create_job_from_upload(
            task_type="hls_streaming",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=HLSStreamingOutput,
            params_factory=lambda path: HLSStreamingParams(
                input_path=path,
                output_path="output",
                variants=variant_models,
                include_original=include_original,
            ),
        )

    _ = create_hls_job
    return router
