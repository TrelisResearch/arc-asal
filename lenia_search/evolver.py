"""
Evolutionary pattern search using pairwise VLM comparisons
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .lenia import Lenia, save_snapshot
from .vlm_checker import compare_patterns, check_pattern_match


@dataclass
class Candidate:
    """A candidate pattern in the population."""
    params: dict
    image_path: str
    rank: int = 0
    grid: Optional[np.ndarray] = field(default=None, repr=False)
    is_valid: bool = True


class EvolutionarySearcher:
    """Evolutionary search for Lenia patterns using pairwise VLM comparisons."""

    def __init__(
        self,
        description: str,
        output_dir: str = "evolved_patterns",
        population_size: int = 10,
        temperature: float = 2.0,
        mutation_scale: float = 0.3,
    ):
        self.description = description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.population_size = population_size
        self.temperature = temperature
        self.mutation_scale = mutation_scale

        self.population: list[Candidate] = []
        self.generation = 0
        self.best_candidate: Optional[Candidate] = None

        # Parameter ranges (widened for variety)
        self.param_ranges = {
            'mu': (0.03, 0.45),
            'sigma': (0.005, 0.12),
            'kernel_radius': (5, 28),
            'kernel_beta': (0.05, 1.2),
        }

    def random_params(self) -> dict:
        """Generate random parameters within ranges."""
        return {
            'mu': random.uniform(*self.param_ranges['mu']),
            'sigma': random.uniform(*self.param_ranges['sigma']),
            'kernel_radius': random.randint(*self.param_ranges['kernel_radius']),
            'kernel_peaks': [1.0],
            'kernel_beta': [random.uniform(*self.param_ranges['kernel_beta'])],
        }

    def mutate_params(self, params: dict) -> dict:
        """Mutate parameters with gaussian noise scaled by mutation_scale."""
        new_params = params.copy()

        # Scale mutations by parameter range
        mu_range = self.param_ranges['mu'][1] - self.param_ranges['mu'][0]
        sigma_range = self.param_ranges['sigma'][1] - self.param_ranges['sigma'][0]
        beta_range = self.param_ranges['kernel_beta'][1] - self.param_ranges['kernel_beta'][0]

        new_params['mu'] = np.clip(
            params['mu'] + random.gauss(0, mu_range * self.mutation_scale * 0.1),
            *self.param_ranges['mu']
        )

        new_params['sigma'] = np.clip(
            params['sigma'] + random.gauss(0, sigma_range * self.mutation_scale * 0.1),
            *self.param_ranges['sigma']
        )

        new_params['kernel_beta'] = [np.clip(
            params['kernel_beta'][0] + random.gauss(0, beta_range * self.mutation_scale * 0.1),
            *self.param_ranges['kernel_beta']
        )]

        # Occasionally mutate kernel radius
        if random.random() < 0.2:
            new_params['kernel_radius'] = np.clip(
                params['kernel_radius'] + random.choice([-2, -1, 1, 2]),
                *self.param_ranges['kernel_radius']
            )

        return new_params

    def evaluate_validity(self, lenia: Lenia, steps: int = 200) -> bool:
        """Run simulation and check if pattern is valid (not died/filled)."""
        lenia.run(steps=steps)

        grid = lenia.grid
        max_val = grid.max()
        mean_val = grid.mean()
        nonzero = np.count_nonzero(grid > 0.01)

        # Pattern should have some activity but not fill everything
        is_valid = (
            max_val > 0.3 and
            0.01 < mean_val < 0.8 and
            100 < nonzero < lenia.size * lenia.size * 0.8
        )

        return is_valid

    def create_candidate(self, params: dict, candidate_id: int) -> Optional[Candidate]:
        """Create a candidate from params, returns None if invalid."""
        lenia = Lenia(size=128, dt=0.1, params=params)
        is_valid = self.evaluate_validity(lenia, steps=200)

        image_path = self.output_dir / f"candidate_{candidate_id}.png"
        save_snapshot(lenia, str(image_path))

        return Candidate(
            params=params,
            image_path=str(image_path),
            grid=lenia.grid.copy(),
            is_valid=is_valid,
        )

    def select_parent(self) -> Candidate:
        """Select a parent using softmax on rank (lower rank = better = higher prob)."""
        if not self.population:
            raise ValueError("Population is empty")

        # Softmax selection: P(i) ∝ exp(-rank / temperature)
        ranks = np.array([c.rank for c in self.population])
        weights = np.exp(-ranks / self.temperature)
        weights = weights / weights.sum()

        return random.choices(self.population, weights=weights, k=1)[0]

    def insert_candidate(self, new_candidate: Candidate, opponent: Candidate, new_wins: bool):
        """Insert new candidate into population based on comparison result."""
        if new_wins:
            # Insert just above opponent
            new_candidate.rank = opponent.rank
            for c in self.population:
                if c.rank >= new_candidate.rank:
                    c.rank += 1
        else:
            # Insert just below opponent
            new_candidate.rank = opponent.rank + 1
            for c in self.population:
                if c.rank >= new_candidate.rank and c != opponent:
                    c.rank += 1

        self.population.append(new_candidate)

        # Trim population to max size (remove worst)
        if len(self.population) > self.population_size:
            self.population.sort(key=lambda c: c.rank)
            removed = self.population.pop()
            # Clean up removed candidate's image
            try:
                Path(removed.image_path).unlink(missing_ok=True)
            except:
                pass

        # Update best candidate
        self.population.sort(key=lambda c: c.rank)
        self.best_candidate = self.population[0]

    def load_existing_candidates(self, callback=None) -> bool:
        """Try to load existing candidates from output directory. Returns True if loaded."""
        results_file = self.output_dir / 'evolution_results.json'
        if not results_file.exists():
            return False

        try:
            with open(results_file) as f:
                results = json.load(f)

            # Check if we have valid candidates
            population_data = results.get('population', [])
            if not population_data:
                return False

            # Load candidates
            loaded = 0
            for cdata in population_data:
                image_path = Path(cdata['image_path'])
                if not image_path.exists():
                    continue

                candidate = Candidate(
                    params=cdata['params'],
                    image_path=str(image_path),
                    rank=cdata['rank'],
                    is_valid=cdata.get('is_valid', True),
                )
                self.population.append(candidate)
                loaded += 1

                if callback:
                    callback('load', loaded, candidate, f"Loaded existing candidate (rank {cdata['rank']})")

            if loaded > 0:
                # Re-sort by rank
                self.population.sort(key=lambda c: c.rank)
                self.best_candidate = self.population[0] if self.population else None
                self.generation = results.get('generations', 0)
                return True

        except (json.JSONDecodeError, KeyError) as e:
            if callback:
                callback('load_error', 0, None, f"Could not load existing candidates: {e}")

        return False

    def initialize_population(self, callback=None, regen=False) -> int:
        """Initialize population with random valid patterns. Returns attempts used."""
        # Try to load existing candidates first (unless regen=True)
        if not regen and self.load_existing_candidates(callback):
            if callback:
                callback('load_done', len(self.population), None,
                         f"Loaded {len(self.population)} existing candidates from previous run")
            return 0

        attempts = 0
        candidate_id = 0

        while len(self.population) < self.population_size and attempts < self.population_size * 10:
            attempts += 1
            params = self.random_params()

            candidate = self.create_candidate(params, candidate_id)
            if candidate is None:
                continue

            if not candidate.is_valid:
                if callback:
                    callback('init', attempts, None, "Invalid pattern (died/filled)")
                continue

            candidate.rank = len(self.population)
            self.population.append(candidate)
            candidate_id += 1

            if callback:
                callback('init', attempts, candidate, f"Added to population (valid)")

        return attempts

    def evolve_step(self, callback=None) -> tuple[Candidate, bool, str]:
        """
        Perform one evolution step.
        Returns (new_candidate, is_strong_match, explanation)
        """
        self.generation += 1

        # Select parent and mutate
        parent = self.select_parent()
        child_params = self.mutate_params(parent.params)

        # Create child candidate
        child_id = self.generation + self.population_size
        child = self.create_candidate(child_params, child_id)

        if not child.is_valid:
            if callback:
                callback('evolve', self.generation, child, "Invalid pattern - discarded")
            return child, False, "Invalid pattern"

        # Select random opponent from population for comparison
        opponent = random.choice(self.population)

        # Compare with VLM
        try:
            winner, confidence, explanation = compare_patterns(
                child.image_path,
                opponent.image_path,
                self.description
            )
            child_wins = (winner == 'A')
        except Exception as e:
            if callback:
                callback('evolve', self.generation, child, f"VLM error: {e}")
            return child, False, str(e)

        # Insert into population
        self.insert_candidate(child, opponent, child_wins)

        if callback:
            result = "WINS" if child_wins else "loses"
            callback('evolve', self.generation, child, f"{result} vs opponent (conf: {confidence}) - {explanation[:40]}")

        # Check for strong match using absolute check (not just relative comparison)
        is_strong_match = False
        match_explanation = explanation
        if child_wins and confidence == 'high':
            # Do an absolute check - does this actually match the description?
            try:
                matches, match_explanation = check_pattern_match(child.image_path, self.description)
                is_strong_match = matches
            except Exception:
                is_strong_match = False

        return child, is_strong_match, match_explanation

    def evolve(self, max_generations: int = 100, callback=None, regen: bool = False) -> list[Candidate]:
        """
        Run evolutionary search.

        Args:
            max_generations: Maximum number of generations
            callback: Optional callback(phase, generation, candidate, message)
            regen: If True, regenerate candidates even if existing ones found

        Returns:
            Final population sorted by rank
        """
        # Initialize population
        if callback:
            callback('start', 0, None, f"Initializing population of {self.population_size}")

        init_attempts = self.initialize_population(callback, regen=regen)

        if len(self.population) < 2:
            if callback:
                callback('error', 0, None, "Could not initialize population with enough valid patterns")
            return []

        if callback:
            callback('init_done', init_attempts, None, f"Population initialized with {len(self.population)} candidates")

        # Evolution loop
        strong_matches = []
        for gen in range(max_generations):
            child, is_strong_match, explanation = self.evolve_step(callback)

            if is_strong_match:
                strong_matches.append(child)
                if callback:
                    callback('match', gen, child, f"Strong match found! {explanation}")

                # Save strong match
                match_path = self.output_dir / f"strong_match_{len(strong_matches)}.png"
                save_snapshot(
                    Lenia(size=128, dt=0.1, params=child.params),
                    str(match_path)
                )

                # Save params
                params_path = self.output_dir / f"strong_match_{len(strong_matches)}_params.json"
                with open(params_path, 'w') as f:
                    json.dump({
                        'params': self._convert_params(child.params),
                        'explanation': explanation,
                        'generation': gen,
                    }, f, indent=2)

                if len(strong_matches) >= 3:
                    if callback:
                        callback('done', gen, None, "Found 3 strong matches!")
                    break

        # Save final results
        self.save_results()

        return sorted(self.population, key=lambda c: c.rank)

    def _convert_params(self, params: dict) -> dict:
        """Convert numpy types in params to native Python types for JSON."""
        result = {}
        for k, v in params.items():
            if isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, list):
                result[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                result[k] = v
        return result

    def save_results(self):
        """Save evolution results."""
        results = {
            'description': self.description,
            'generations': self.generation,
            'population': [
                {
                    'rank': c.rank,
                    'params': self._convert_params(c.params),
                    'image_path': c.image_path,
                    'is_valid': bool(c.is_valid),
                }
                for c in sorted(self.population, key=lambda c: c.rank)
            ],
        }

        with open(self.output_dir / 'evolution_results.json', 'w') as f:
            json.dump(results, f, indent=2)


def run_evolution(
    description: str,
    max_generations: int = 100,
    population_size: int = 10,
    output_dir: str = "evolved_patterns",
    regen: bool = False,
):
    """Run evolutionary search with console output."""
    print(f"Evolving patterns matching: '{description}'")
    print(f"Population: {population_size}, Max generations: {max_generations}")
    if regen:
        print("(Regenerating candidates from scratch)")
    print("-" * 70)

    evolver = EvolutionarySearcher(
        description=description,
        output_dir=output_dir,
        population_size=population_size,
    )

    def on_event(phase, gen, candidate, message):
        if phase == 'init':
            print(f"[INIT {gen:3d}] {message}")
        elif phase == 'load':
            print(f"[LOAD {gen:3d}] {message}")
        elif phase == 'load_done':
            print(f"\n{'='*70}")
            print(f"{message}")
            print(f"Resuming evolution...")
            print(f"{'='*70}\n")
        elif phase == 'init_done':
            print(f"\n{'='*70}")
            print(f"Population ready. Starting evolution...")
            print(f"{'='*70}\n")
        elif phase == 'evolve':
            params_str = ""
            if candidate:
                params_str = f"mu={candidate.params['mu']:.3f} σ={candidate.params['sigma']:.3f}"
            print(f"[GEN {gen:3d}] {params_str} | {message}")
        elif phase == 'match':
            print(f"\n*** STRONG MATCH at generation {gen}! ***")
            print(f"    {message}\n")
        elif phase == 'done':
            print(f"\n{'='*70}")
            print(f"{message}")
            print(f"{'='*70}")

    population = evolver.evolve(max_generations=max_generations, callback=on_event, regen=regen)

    print(f"\nFinal population (best to worst):")
    for c in population[:5]:
        valid_str = "✓" if c.is_valid else "✗"
        print(f"  Rank {c.rank}: {valid_str} mu={c.params['mu']:.3f} σ={c.params['sigma']:.3f}")

    print(f"\nResults saved to {output_dir}/")

    return population
