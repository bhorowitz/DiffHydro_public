import os
import jax

def init_distributed():
    """Initialize JAX multihost if we're running on multiple processes."""
    # If already initialized or single process, do nothing
    if jax.process_count() == 1:
        return

    coordinator = os.environ.get("COORDINATOR_ADDRESS", None)
    if coordinator is None:
        raise RuntimeError("COORDINATOR_ADDRESS env var must be set in multi-host runs")

    num_processes = int(os.environ["JAX_PROCESS_COUNT"])
    process_id = int(os.environ["JAX_PROCESS_INDEX"])

    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=num_processes,
        process_id=process_id,
    )