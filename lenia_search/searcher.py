"""
Pattern searcher for finding Lenia patterns matching descriptions
"""

import json
import random
from pathlib import Path

import numpy as np

from .lenia import Lenia, save_snapshot
from .vlm_checker import check_pattern_match


class PatternSearcher:
    """Search for Lenia patterns matching a description."""

    def __init__(self, description: str, output_dir: str = "found_patterns"):
        self.description = description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.found_patterns = []
        self.search_history = []

        # Parameter ranges to explore (widened for more variety)
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

    def mutate_params(self, params: dict, mutation_rate: float = 0.3) -> dict:
        """Mutate existing parameters slightly."""
        new_params = params.copy()

        if random.random() < mutation_rate:
            new_params['mu'] = np.clip(
                params['mu'] + random.gauss(0, 0.02),
                *self.param_ranges['mu']
            )

        if random.random() < mutation_rate:
            new_params['sigma'] = np.clip(
                params['sigma'] + random.gauss(0, 0.01),
                *self.param_ranges['sigma']
            )

        if random.random() < mutation_rate:
            new_params['kernel_beta'] = [np.clip(
                params['kernel_beta'][0] + random.gauss(0, 0.1),
                *self.param_ranges['kernel_beta']
            )]

        return new_params

    def evaluate_pattern(self, lenia: Lenia, steps: int = 200) -> tuple[bool, float]:
        """
        Run simulation and check if pattern is interesting.
        Returns (is_interesting, activity_score)
        """
        lenia.run(steps=steps)

        # Check if pattern survived and has activity
        grid = lenia.grid
        max_val = grid.max()
        mean_val = grid.mean()
        nonzero = np.count_nonzero(grid > 0.01)

        # Pattern should have some activity but not fill everything
        is_interesting = (
            max_val > 0.3 and
            0.01 < mean_val < 0.8 and
            100 < nonzero < lenia.size * lenia.size * 0.8
        )

        activity_score = mean_val * (1 - abs(mean_val - 0.3))

        return is_interesting, activity_score

    def search(self, target_count: int = 3, max_attempts: int = 100,
               callback=None) -> list[dict]:
        """
        Search for patterns matching the description.

        Args:
            target_count: Number of matching patterns to find
            max_attempts: Maximum number of attempts
            callback: Optional callback(attempt, params, image_path, matches, explanation)
        """
        attempt = 0
        temp_image = self.output_dir / "temp_snapshot.png"

        while len(self.found_patterns) < target_count and attempt < max_attempts:
            attempt += 1

            # Generate parameters (random or mutate from found)
            if self.found_patterns and random.random() < 0.5:
                base_params = random.choice(self.found_patterns)['params']
                params = self.mutate_params(base_params)
            else:
                params = self.random_params()

            # Create and run simulation
            lenia = Lenia(size=128, dt=0.1, params=params)
            is_interesting, score = self.evaluate_pattern(lenia, steps=200)

            if not is_interesting:
                if callback:
                    callback(attempt, params, None, False, "Pattern died or filled space")
                continue

            # Save snapshot for VLM
            save_snapshot(lenia, str(temp_image))

            # Check with VLM
            try:
                matches, explanation = check_pattern_match(str(temp_image), self.description)
            except Exception as e:
                if callback:
                    callback(attempt, params, str(temp_image), False, f"VLM error: {e}")
                continue

            # Record attempt
            self.search_history.append({
                'attempt': attempt,
                'params': params,
                'matches': matches,
                'explanation': explanation,
                'score': score,
            })

            if callback:
                callback(attempt, params, str(temp_image), matches, explanation)

            if matches:
                # Save the matching pattern
                pattern_id = len(self.found_patterns) + 1
                pattern_path = self.output_dir / f"pattern_{pattern_id}.png"
                save_snapshot(lenia, str(pattern_path))

                pattern_data = {
                    'id': pattern_id,
                    'params': params,
                    'explanation': explanation,
                    'image_path': str(pattern_path),
                    'score': score,
                }
                self.found_patterns.append(pattern_data)

                # Save params to JSON
                params_path = self.output_dir / f"pattern_{pattern_id}_params.json"
                with open(params_path, 'w') as f:
                    json.dump(pattern_data, f, indent=2)

                print(f"\n*** Found matching pattern {pattern_id}! ***")
                print(f"    Params: mu={params['mu']:.4f}, sigma={params['sigma']:.4f}")
                print(f"    Explanation: {explanation}\n")

        return self.found_patterns

    def save_results(self):
        """Save all results to files."""
        results = {
            'description': self.description,
            'found_patterns': self.found_patterns,
            'total_attempts': len(self.search_history),
        }

        with open(self.output_dir / 'search_results.json', 'w') as f:
            json.dump(results, f, indent=2)
