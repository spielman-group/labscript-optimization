"""The parameter space a learner searches: enabled parameters and their bounds.

Scaling to the unit cube is a fixed linear map onto the bounds, not a fit to
the data, so every learner sees the bounds the same way. Which runmanager
global carries which parameter is the configuration's business, not this
module's.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import knobs


@dataclass(frozen=True)
class Parameter:
    """One optimized quantity.

    Parameters
    ----------
    name : str
        The key this parameter has in the configuration.
    minimum, maximum : float
        Bounds, in real units.
    start : float, optional
        Preferred first value; ``None`` starts from a random draw.
    enable : bool
        ``False`` drops the parameter from the search, start included. Its
        runmanager global is not set, so it keeps the value runmanager holds.
    """

    name: str
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
    """The enabled parameters, in a fixed order, with their bounds.

    Raises ``ValueError`` for an empty space, for a repeated name and for a
    ``start`` on only some of the parameters.
    """

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = tuple(p for p in parameters if p.enable)
        if not self.parameters:
            raise ValueError("no enabled parameters to optimize")
        # A proposal is read back by name, so a repeated name would silently
        # take the first dimension's value.
        names = [p.name for p in self.parameters]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"{', '.join(repr(n) for n in repeated)} names more than one "
                f"enabled parameter; each needs a name of its own. "
                f"The names given are {names}"
            )
        # A start is proposed as one point over every parameter. A partial one
        # is refused, not dropped, which would open the run on a draw unannounced.
        started = [p.name for p in self.parameters if p.start is not None]
        unstarted = [p.name for p in self.parameters if p.start is None]
        if started and unstarted:
            raise ValueError(
                f"a start is one point over every searched parameter, so it is "
                f"written on all of them or on none. "
                f"{', '.join(repr(n) for n in started)} "
                f"{'has' if len(started) == 1 else 'have'} one; "
                f"{', '.join(repr(n) for n in unstarted)} "
                f"{'does' if len(unstarted) == 1 else 'do'} not."
            )
        self.minimum = np.array([p.minimum for p in self.parameters], dtype=float)
        self.maximum = np.array([p.maximum for p in self.parameters], dtype=float)
        self.extent = self.maximum - self.minimum

    @property
    def num_params(self) -> int:
        return len(self.parameters)

    @property
    def start(self) -> np.ndarray | None:
        """The point a run opens at, or ``None`` if no parameter has a start."""
        if any(p.start is None for p in self.parameters):
            return None
        return np.array([p.start for p in self.parameters], dtype=float)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """``x`` moved inside the bounds."""
        return np.clip(x, self.minimum, self.maximum)

    def scale(self, x: np.ndarray) -> np.ndarray:
        """Map real units onto the unit cube."""
        return (np.asarray(x, dtype=float) - self.minimum) / self.extent

    def unscale(self, u: np.ndarray) -> np.ndarray:
        """Map the unit cube back onto real units."""
        return np.asarray(u, dtype=float) * self.extent + self.minimum

    def uniform(
        self,
        rng: np.random.Generator,
        k: int = 1,
        center: np.ndarray | None = None,
        region: np.ndarray | None = None,
    ) -> np.ndarray:
        """``k`` points drawn uniformly from the space.

        With a ``center``, the draw is confined to ``region`` around it, as
        :meth:`bounds_near` clips it. Without one, ``region`` is ignored.
        """
        low, high = self.minimum, self.maximum
        if center is not None:
            low, high = self.bounds_near(center, region)
        return rng.uniform(low, high, size=(k, self.num_params))

    def absolute_trust_region(
        self, trust_region: float | Sequence[float] | np.ndarray | None
    ) -> np.ndarray | None:
        """Resolve a trust region to an absolute distance per parameter.

        A number in (0, 1) is a fraction of each parameter's range; a sequence
        of numbers is absolute distances, one per parameter. ``None`` means no
        limit and is returned as ``None``. Anything else, including a quoted
        ``"0.1"``, raises ``ValueError``.
        """
        if trust_region is None:
            return None
        if knobs.is_number(trust_region):
            if not 0 < trust_region < 1:
                raise ValueError(
                    f"a scalar trust_region is a fraction of each parameter's "
                    f"range and must be in (0, 1), got {trust_region}"
                )
            region = float(trust_region) * self.extent
        elif knobs.is_numbers(trust_region):
            region = np.array(trust_region, dtype=float)
            if region.shape != (self.num_params,):
                raise ValueError(
                    f"trust_region has shape {region.shape}, expected "
                    f"({self.num_params},)"
                )
        else:
            raise ValueError(
                f"trust_region must be a number in (0, 1) or a list of numbers, "
                f"one distance per parameter, not {trust_region!r}"
            )
        if not np.all(region > 0):
            raise ValueError(f"every trust_region value must be positive: {region}")
        if not np.all(region <= self.extent):
            raise ValueError(
                f"trust_region {region} is wider than the bounds {self.extent}"
            )
        return region

    def bounds_near(
        self, center: np.ndarray, region: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The bounds within ``region`` of ``center``, kept inside the space.

        ``region`` is an absolute distance per parameter, as
        :meth:`absolute_trust_region` returns it; ``None`` gives the whole space.
        """
        if region is None:
            return self.minimum, self.maximum
        return (
            np.maximum(self.minimum, center - region),
            np.minimum(self.maximum, center + region),
        )
