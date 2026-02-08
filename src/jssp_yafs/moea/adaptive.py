from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class UCB1Bandit:
    arms: list[str]
    c: float = 1.0
    counts: dict[str, int] = field(init=False)
    values: dict[str, float] = field(init=False)
    total: int = field(init=False)

    def __post_init__(self) -> None:
        self.counts = {a: 0 for a in self.arms}
        self.values = {a: 0.0 for a in self.arms}
        self.total = 0

    def select(self) -> str:
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm

        log_total = math.log(max(1, self.total))
        best_arm = self.arms[0]
        best_score = float("-inf")
        for arm in self.arms:
            mean = self.values[arm]
            bonus = self.c * math.sqrt(2.0 * log_total / self.counts[arm])
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def update(self, arm: str, reward: float) -> None:
        self.total += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
