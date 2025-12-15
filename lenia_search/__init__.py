"""
Lenia pattern search package
"""

from .lenia import Lenia, save_snapshot, visualize_lenia
from .vlm_checker import check_pattern_match, compare_patterns
from .searcher import PatternSearcher
from .evolver import EvolutionarySearcher, run_evolution

__all__ = [
    'Lenia',
    'save_snapshot',
    'visualize_lenia',
    'check_pattern_match',
    'compare_patterns',
    'PatternSearcher',
    'EvolutionarySearcher',
    'run_evolution',
]
