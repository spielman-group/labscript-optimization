"""The parameter space a learner searches.

Holds the enabled parameters, their bounds, and the mapping onto the runmanager
globals that carry them. Scaling to the unit cube lives here rather than in a
learner so that every learner sees bounds the same way; it is a linear map onto
the boundaries, not a fit to the data, so it does not change as observations
arrive.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """One optimised quantity.

    Args:
        name: The key this parameter has in the configuration.
        global_name: The runmanager global that carries its value.
        minimum: Lower bound, in real units.
        maximum: Upper bound, in real units.
        start: Preferred first value, or ``None`` to start from a random draw.
        enable: False leaves the parameter out of the search at its start value.
    """

    name: str
    global_name: str
    minimum: float
    maximum: float
    start: float | None = None
    enable: bool = True

    def __post_init__(self):
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError(
                f"parameter {self.name!r} needs finite bounds, got "
                f"[{self.minimum}, {self.maximum}]"
            )
        if self.minimum >= self.maximum:
            raise ValueError(
                f"parameter {self.name!r} has min >= max: "
                f"[{self.minimum}, {self.maximum}]"
            )
        if self.start is not None and not (self.minimum <= self.start <= self.maximum):
            raise ValueError(
                f"parameter {self.name!r} has start {self.start} outside "
                f"[{self.minimum}, {self.maximum}]"
            )


class ParameterSpace:
    """The enabled parameters, in a fixed order, with their bounds."""

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = tuple(p for p in parameters if p.enable)
        if not self.parameters:
            raise ValueError("no enabled parameters to optimise")
        self.disabled = tuple(p for p in parameters if not p.enable)
        self.minimum = np.array([p.minimum for p in self.parameters], dtype=float)
        self.maximum = np.array([p.maximum for p in self.parameters], dtype=float)
        self.extent = self.maximum - self.minimum

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def num_params(self) -> int:
        return len(self.parameters)

    @property
    def global_names(self) -> tuple[str, ...]:
        return tuple(p.global_name for p in self.parameters)

    @property
    def start(self) -> np.ndarray | None:
        """The configured starting point, or ``None`` if not every parameter has one."""
        if any(p.start is None for p in self.parameters):
            return None
        return np.array([p.start for p in self.parameters], dtype=float)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """``x`` moved inside the bounds."""
        return np.clip(x, self.minimum, self.maximum)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Elementwise-all test of whether each row of ``x`` is within bounds."""
        x = np.atleast_2d(x)
        return np.all((x >= self.minimum) & (x <= self.maximum), axis=-1)

    def scale(self, x: np.ndarray) -> np.ndarray:
        """Map real units onto the unit cube."""
        return (np.asarray(x, dtype=float) - self.minimum) / self.extent

    def unscale(self, u: np.ndarray) -> np.ndarray:
        """Map the unit cube back onto real units."""
        return np.asarray(u, dtype=float) * self.extent + self.minimum

    def uniform(self, rng: np.random.Generator, k: int = 1) -> np.ndarray:
        """``k`` points drawn uniformly from the whole space."""
        return rng.uniform(self.minimum, self.maximum, size=(k, self.num_params))

    def absolute_trust_region(self, trust_region) -> np.ndarray | None:
        """Resolve a trust region onto an absolute per-parameter distance.

        A float in (0, 1) is read as a fraction of each parameter's range; a
        sequence is read as absolute distances already. ``None`` means the
        learner may travel anywhere.
        """
        if trust_region is None:
            return None
        if np.isscalar(trust_region):
            if not 0 < float(trust_region) < 1:
                raise ValueError(
                    f"a scalar trust_region is a fraction of each parameter's "
                    f"range and must be in (0, 1), got {trust_region}"
                )
            region = float(trust_region) * self.extent
        else:
            region = np.array(trust_region, dtype=float)
            if region.shape != (self.num_params,):
                raise ValueError(
                    f"trust_region has shape {region.shape}, expected "
                    f"({self.num_params},)"
                )
        if not np.all(region > 0):
            raise ValueError(f"every trust_region value must be positive: {region}")
        if not np.all(region <= self.extent):
            raise ValueError(
                f"trust_region {region} is wider than the bounds {self.extent}"
            )
        return region
