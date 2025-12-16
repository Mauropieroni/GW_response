# Existing API (backwards compatible)
from .constants import *
from .lisa import *
from .utils import *
from .detector import *
from .single_link import *
from .tdi import *
from .response import *
from .noise import *

# High-level API (v2.0)
from .results import ResponseResult
from .api import compute_response, quick_response
from .presets import (
    list_tdi_options,
    get_tdi_info,
    TDI_GUIDE,
    LISA_BAND,
    LISA_SWEET_SPOT,
    DEFAULT_CONFIG,
)

# Performance and optimization utilities
from .config import (
    configure_for_performance,
    configure_xla_flags,
    configure_jax_memory,
    get_device_info,
    print_device_info,
)
from .parallel import (
    parallel_compute_response,
    parallel_single_link_response,
    get_device_count,
    get_parallel_info,
)
from . import benchmark
