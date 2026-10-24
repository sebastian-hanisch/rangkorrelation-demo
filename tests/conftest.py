"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def four():
    """Vier Punkte (1,1), (2,3), (3,2), (4,4). Ränge = Werte. Handrechnung: d = (0, 1, 1, 0), Σd² = 2, ρ_S = 1 − 6·2/(4·15) = 0,8;
    Paare: (1,2) K, (1,3) K, (1,4) K, (2,3) D, (2,4) K, (3,4) K, also C = 5, D = 1, τ = 4/6 = 2/3."""
    return [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]


@pytest.fixture
def tied():
    """x = (1, 1, 2, 3), y = (1, 2, 2, 3): Paare (1,2) Gleichstand in x, (2,3) Gleichstand in y, die übrigen vier konkordant:
    C = 4, D = 0, T_x = 1, T_y = 1, n0 = 6; τ-a = 4/6, τ-b = 4/5 = 0,8, τ-c = 2·4/(16·2/3) = 0,75, γ = 1.
    Durchschnittsränge x: (1,5; 1,5; 3; 4), y: (1; 2,5; 2,5; 4): Σdxdy = 3,75, Σdx² = Σdy² = 4,5, ρ_S = 5/6."""
    return [1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]
