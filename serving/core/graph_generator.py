import os
import sys
from .request import *
from .logger import get_logger
from .run_paths import input_path

logger = get_logger("GraphGenerator")


_llm_converter_type = None


def _load_llm_converter(chakra_root):
    """Load Chakra's Python converter into the serving process once."""
    global _llm_converter_type
    if _llm_converter_type is not None:
        return _llm_converter_type

    search_roots = (
        os.path.join(chakra_root, "build", "lib"),
        chakra_root,
    )
    for root in reversed(search_roots):
        if root not in sys.path:
            sys.path.insert(0, root)

    from chakra.src.converter.llm_converter import LLMConverter

    _llm_converter_type = LLMConverter
    return _llm_converter_type


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, cleanup_trace=True):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    logger.debug(
        "Generating graph in-process: input=%s, output=%s, num_npus=%s, "
        "npu_offset=%s, local_offloading=%s",
        trace_path,
        output_path,
        num_npus,
        npu_offset,
        enable_local_offloading,
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    converter_type = _load_llm_converter(chakra)
    converter = converter_type(
        trace_path,
        output_path,
        num_npus,
        npu_offset,
        enable_local_offloading,
    )
    converter.convert()
    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return
