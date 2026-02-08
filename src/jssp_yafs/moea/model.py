from __future__ import annotations

from dataclasses import dataclass

from jssp_yafs.scheduling.representation import Chromosome


@dataclass(slots=True)
class Individual:
    chromosome: Chromosome
    objectives: tuple[float, float, float]
    metrics: dict[str, float]
    rank: int = 0
    crowding: float = 0.0

    def copy(self) -> "Individual":
        return Individual(
            chromosome=self.chromosome.copy(),
            objectives=tuple(self.objectives),
            metrics=dict(self.metrics),
            rank=self.rank,
            crowding=self.crowding,
        )
