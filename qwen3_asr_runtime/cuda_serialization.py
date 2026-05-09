# coding=utf-8
from __future__ import annotations

import threading


# CUDA graph capture is process-global enough that unrelated model kernels in
# another thread can invalidate it. Share this lock between ASR graph capture and
# auxiliary GPU generation paths that may run concurrently in the service.
CUDA_GRAPH_CAPTURE_LOCK = threading.Lock()


__all__ = ["CUDA_GRAPH_CAPTURE_LOCK"]
