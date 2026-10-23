"""Extracted from FlashInfer; Copyright FlashInfer team; Apache-2.0."""
import functools

@functools.cache
def _get_cute_dsl_version() -> str:
    """Fingerprint of the CuTe-DSL compiler stack.

    Covers every installed ``nvidia-cutlass-dsl*`` distribution, not just the
    main package: the codegen backend lives in the libs packages, and the
    CUDA-family variant is encoded in the package *name* (e.g.
    ``nvidia-cutlass-dsl-libs-cu13``), so two environments with the same DSL
    version but different libs variants must not share artifacts. The system
    CUDA toolkit/driver deliberately does not participate: ``cute.compile``
    uses this bundled stack, not ``CUDA_HOME``.
    """
    import importlib.metadata

    stack = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("nvidia-cutlass-dsl")
    )
    return ";".join(stack) if stack else "unknown"
