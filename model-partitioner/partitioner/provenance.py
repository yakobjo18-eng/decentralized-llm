"""Where a number came from. Every memory figure and every cost figure
the partitioner produces carries one of these, so a plan never presents
an estimate as a measurement.

  MEASURED   -- observed on a real machine (e.g. process RSS after
                importing torch + transformers on Computer 1).
  CALCULATED -- exact arithmetic from known facts (parameter shapes
                from the real model definition x bytes per element).
  ESTIMATED  -- a formula with assumptions (activation peaks,
                quantization overheads, throughput from core counts).
  UNKNOWN    -- not determinable from the inputs.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Iterable, Optional


class Provenance(str, Enum):
    MEASURED = "MEASURED"
    CALCULATED = "CALCULATED"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


# Higher = less certain. A total is only as certain as its least
# certain part.
_WEAKNESS = {Provenance.MEASURED: 0, Provenance.CALCULATED: 1, Provenance.ESTIMATED: 2, Provenance.UNKNOWN: 3}


def weakest(items: Iterable[Provenance]) -> Provenance:
    result = Provenance.MEASURED
    seen = False
    for p in items:
        seen = True
        if _WEAKNESS[p] > _WEAKNESS[result]:
            result = p
    return result if seen else Provenance.UNKNOWN


@dataclasses.dataclass(frozen=True)
class MemoryItem:
    """One line of a memory breakdown. bytes is None only when UNKNOWN."""

    name: str
    bytes: Optional[int]
    provenance: Provenance
    basis: str

    def to_dict(self) -> dict:
        return {"name": self.name, "bytes": self.bytes, "provenance": self.provenance.value, "basis": self.basis}
