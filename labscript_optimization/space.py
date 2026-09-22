"""The parameter space a learner searches.

Holds the enabled parameters and their bounds, and nothing about runmanager:
which global carries which parameter is the configuration's business rather
than this module's. Scaling to the unit cube lives here rather than in a
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
        minimum: Lower bound, in real units.
        maximum: Upper bound, in real units.
        start: Preferred first value, or ``None`` to start from a random draw.
        enable: False leaves the parameter out of the search at its start value.
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

    Each carries a name of its own: the names are how a parameter vector is
    read back out as values, so they identify the dimensions.
    """

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = tuple(p for p in parameters if p.enable)
        if not self.parameters:
            raise ValueError("no enabled parameters to optimise")
        # A name is how a proposal's coordinates are read back out, so two
        # dimensions answering to one name is a malformed space whatever built
        # it: the second silently takes the first's value everywhere the
        # parameters are looked up by name.
        names = [p.name for p in self.parameters]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"{', '.join(repr(n) for n in repeated)} names more than one "
                f"enabled parameter. Each searched parameter needs a name of "
                f"its own; the names given are {names}"
            )
        # A start is the whole space's or it is nobody's: it is proposed as
        # one point over every parameter, so there is nothing to do with one
        # written for some of them. Refused rather than honoured for those and
        # drawn for the rest, because a file that names a start has a reason
        # for it and a draw is not that reason -- and refused rather than
        # dropped, which is what used to happen: a start on three parameters
        # of five was silently ignored for all five, and the run opened on a
        # uniform draw with nothing said.
        started = [p.name for p in self.parameters if p.start is not None]
        unstarted = [p.name for p in self.parameters if p.start is None]
        if started and unstarted:
            raise ValueError(
                f"a start is one point over every searched parameter, so it "
                f"is written on all of them or on none. "
                f"{', '.join(repr(n) for n in started)} "
                f"{'has' if len(started) == 1 else 'have'} one; "
                f"{', '.join(repr(n) for n in unstarted)} "
                f"{'does' if len(unstarted) == 1 else 'do'} not. Give the "
                f"rest a start, or delete the ones that have it and let the "
                f"run open on a draw."
            )
        self.minimum = np.array([p.minimum for p in self.parameters], dtype=float)
        self.maximum = np.array([p.maximum for p in self.parameters], dtype=float)
        self.extent = self.maximum - self.minimum

    @property
    def num_params(self) -> int:
        return len(self.parameters)


    @property
    def start(self) -> np.ndarray | None:
        """Where a run opens, or ``None`` where no parameter carries a start.

        All or nothing, because the constructor refuses anything between, so
        this answers a point or it answers that there is none.
        """
        if any(p.start is None for p in self.parameters):
            return None
        return np.array([p.start for p in self.parameters], dtype=float)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """``x`` moved inside the bounds."""
        return np.clip(x, self.minimum, self.maximum)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Whether each row of ``x`` is within bounds, one answer per row.

        An array even for a single point, so reduce it with ``.all()``: numpy
        allows a truth test only on a size-one array, and the bare form works
        until the day a second row arrives.
        """
        x = np.atleast_2d(x)
        return np.all((x >= self.minimum) & (x <= self.maximum), axis=-1)

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
        centre: np.ndarray | None = None,
        region: np.ndarray | None = None,
    ) -> np.ndarray:
        """``k`` points drawn uniformly from the space.

        A ``centre`` and a ``region`` confine the draw to that trust region as
        :meth:`bounds_near` clips it; neither covers the whole space.
        """
        low, high = self.minimum, self.maximum
        if centre is not None:
            low, high = self.bounds_near(centre, region)
        return rng.uniform(low, high, size=(k, self.num_params))

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

    def bounds_near(
        self, centre: np.ndarray, region: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The bounds within ``region`` of ``centre``, kept inside the space.

        ``region`` is an absolute per-parameter distance, as
        :meth:`absolute_trust_region` returns it; ``None`` gives back the whole
        space, so a learner free to travel anywhere takes the same path.
        """
        if region is None:
            return self.minimum, self.maximum
        return (
            np.maximum(self.minimum, centre - region),
            np.minimum(self.maximum, centre + region),
        )
