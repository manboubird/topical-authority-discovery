"""This is a pipeline that processes data."""

from kedro.pipeline import Pipeline

from .data_processing.pipeline import create_pipeline as create_data_processing_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = create_data_processing_pipeline()

    return {
        "__default__": data_processing_pipeline,
        "data_processing": data_processing_pipeline,
    }
