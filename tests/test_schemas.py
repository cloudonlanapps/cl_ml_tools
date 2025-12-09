"""Tests for common schemas."""

import pytest
from pydantic import ValidationError

from cl_ml_tools.common.schemas import Job, BaseJobParams


class TestBaseJobParams:
    """Tests for BaseJobParams schema."""

    def test_default_values(self):
        """Test default empty lists."""
        params = BaseJobParams()
        assert params.input_paths == []
        assert params.output_paths == []

    def test_with_paths(self):
        """Test with input and output paths."""
        params = BaseJobParams(
            input_paths=["/input/file1.jpg", "/input/file2.jpg"],
            output_paths=["/output/file1.jpg", "/output/file2.jpg"],
        )
        assert len(params.input_paths) == 2
        assert len(params.output_paths) == 2

    def test_output_paths_must_be_unique(self):
        """Test that duplicate output paths are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            BaseJobParams(
                input_paths=["/input/1.jpg", "/input/2.jpg"],
                output_paths=["/output/same.jpg", "/output/same.jpg"],
            )
        assert "Output paths must be unique" in str(exc_info.value)

    def test_path_count_must_match(self):
        """Test that input and output path counts must match."""
        with pytest.raises(ValidationError) as exc_info:
            BaseJobParams(
                input_paths=["/input/1.jpg", "/input/2.jpg"],
                output_paths=["/output/1.jpg"],
            )
        assert "must match" in str(exc_info.value)


class TestJob:
    """Tests for Job schema."""

    def test_required_fields(self):
        """Test required fields."""
        job = Job(
            job_id="test-123",
            task_type="image_resize",
            params={"width": 100, "height": 100},
        )
        assert job.job_id == "test-123"
        assert job.task_type == "image_resize"
        assert job.status == "queued"
        assert job.progress == 0

    def test_default_values(self):
        """Test default values."""
        job = Job(job_id="test-123", task_type="test", params={})
        assert job.status == "queued"
        assert job.progress == 0
        assert job.task_output is None
        assert job.error_message is None

    def test_progress_bounds(self):
        """Test progress must be 0-100."""
        with pytest.raises(ValidationError):
            Job(job_id="test", task_type="test", params={}, progress=101)

        with pytest.raises(ValidationError):
            Job(job_id="test", task_type="test", params={}, progress=-1)

    def test_from_attributes(self):
        """Test model_config from_attributes."""

        class MockDBRow:
            job_id = "db-123"
            task_type = "test"
            params = {"key": "value"}
            status = "completed"
            progress = 100
            task_output = {"result": "success"}
            error_message = None

        job = Job.model_validate(MockDBRow())
        assert job.job_id == "db-123"
        assert job.status == "completed"
