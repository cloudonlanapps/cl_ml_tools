"""Performance benchmark for face embedding plugin."""

import time
from pathlib import Path
import pytest
from loguru import logger
from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

@pytest.mark.requires_models
def test_face_embedding_performance_benchmark(sample_image_path: Path):
    """Benchmark face embedding performance over multiple iterations."""
    logger.info(f"Starting performance benchmark with image: {sample_image_path}")
    
    embedder = FaceEmbedder()
    
    # Warmup
    logger.info("Performing warmup iterations...")
    for _ in range(2):
        _ = embedder.embed(str(sample_image_path), normalize=True, compute_quality=True)
        
    # Benchmark
    iterations = 5
    logger.info(f"Running {iterations} benchmark iterations...")
    
    start_total = time.perf_counter()
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}")
        _ = embedder.embed(str(sample_image_path), normalize=True, compute_quality=True)
    
    end_total = time.perf_counter()
    avg_total = (end_total - start_total) / iterations
    
    logger.info(f"Benchmark completed. Average total time: {avg_total:.3f}s")
    
    # Also test WITHOUT quality score to see if that's the bottleneck
    logger.info("Running benchmark WITHOUT quality score...")
    start_no_quality = time.perf_counter()
    for i in range(iterations):
        _ = embedder.embed(str(sample_image_path), normalize=True, compute_quality=False)
    end_no_quality = time.perf_counter()
    avg_no_quality = (end_no_quality - start_no_quality) / iterations
    
    logger.info(f"Benchmark (no quality) completed. Average time: {avg_no_quality:.3f}s")
    
    assert avg_total < 2.0  # Sanity check, though we expect it to be much lower
