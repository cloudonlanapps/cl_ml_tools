from enum import IntEnum
from typing import Any, List, Mapping, Sequence

class GraphOptimizationLevel(IntEnum):
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 99

class SessionOptions:
    intra_op_num_threads: int
    inter_op_num_threads: int
    graph_optimization_level: GraphOptimizationLevel
    enable_profiling: bool
    log_severity_level: int

    def __init__(self) -> None: ...

class NodeArg:
    name: str

class InferenceSession:
    def __init__(
        self,
        path: str,
        sess_options: SessionOptions | None = ...,
        providers: Sequence[str] | None = ...,
    ) -> None: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> List[Any]: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...

def get_available_providers() -> list[str]: ...
