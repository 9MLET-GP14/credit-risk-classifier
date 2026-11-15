"""Pacote principal para sustentação e observabilidade do modelo de risco de crédito."""

import os
import sys
import warnings

# Evita warning/traceback do loky no Windows 11 ao tentar invocar wmic depreciado.
# O loky só pula _count_physical_cores() quando LOKY_MAX_CPU_COUNT < os.cpu_count().
if sys.platform == "win32":
    _logical_cores = os.cpu_count() or 2
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, _logical_cores - 1)))

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)

__version__ = "0.1.0"
