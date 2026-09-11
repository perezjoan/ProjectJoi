"""embodied v3. The CUDA allocator setting must be in the environment before torch is first imported, and the
embedder (sentence-transformers) imports torch long before the brain loads, so it is set here, at package import."""
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
