from typing import TypeVar
import numpy as np
from torch import Any, dtype

AdjList = TypeVar("AdjList", bound=dict[int, np.ndarray[Any, dtype]])
