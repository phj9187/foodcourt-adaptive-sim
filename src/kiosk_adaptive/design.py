"""Explicit experimental assumptions; published menu anchors are reused, not invented facts."""
from dataclasses import dataclass, asdict, replace
import json
import numpy as np
from .models import Court, Settings, rng_for, seed_for


@dataclass(frozen=True)
class Objective:
    id: str
    taste: float
    affordability: float
    waiting: float
    customer_share: float
    profit_share: float
    power: float = 1.0

    def components(self, row):
        # All visitors are the denominator, including non-buyers with zero utility.
        c = sum(w * max(0., row[k]) ** self.power for w, k in (
            (self.taste, 'taste'), (self.affordability, 'affordability'),
            (self.waiting, 'waiting')))
        store = self.profit_share * row['margin'] - (1-self.profit_share) * row['waste']
        return c, store

    def score(self, row):
        c, store = self.components(row)
        return self.customer_share*c + (1-self.customer_share)*store


def lhs_objectives(n, seed):
    """Stratified cube -> simplex weights; not a claim to cover all possible functions.

    Exponential normalization gives uniform-simplex marginals; LHS strata apply
    to the five pre-transform coordinates, not to the final simplex coordinates.
    """
    if n < 1:
        raise ValueError('objectives must be positive')
    rng = np.random.default_rng(seed)
    cube = np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(5)])
    result = []
    for i, x in enumerate(cube):
        w = -np.log(np.clip(x[:3], 1e-12, 1)); w /= w.sum()
        result.append(Objective(f'lhs_{i:04d}', *map(float, w), float(x[3]), float(x[4])))
    # Explicit boundaries complement the interior LHS design.
    result += [Objective('customer_only', 1/3, 1/3, 1/3, 1., 1.),
               Objective('profit_only', 1/3, 1/3, 1/3, 0., 1.),
               Objective('waste_only', 1/3, 1/3, 1/3, 0., 0.)]
    return result, cube.tolist()


def make_court(settings, index, split, visitors, design_size=None):
    from .venues import generate_venue
    return generate_venue(settings,index,split,visitors,design_size)


def metrics(row, raw, profit_scale=6000.):
    den = max(1, row['visitors'])
    return dict(taste=sum(r['taste'] for r in raw['records'])/den,
                affordability=sum(r['affordability'] for r in raw['records'])/den,
                waiting=sum(r['wait_utility'] for r in raw['records'])/den,
                margin=(row['revenue']-row['food_cost'])/(den*profit_scale),
                waste=row['waste_cost']/(den*profit_scale))
