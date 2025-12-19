#!/usr/bin/env python
"""
Lenia Pattern Search - Find cellular automaton patterns matching descriptions using VLM

Setup:
    Create a .env file with your Gemini API key:
        GEMINI_API_KEY=your_api_key_here

Usage:
    uv run main.py search "circular blob pattern" --count 3 --attempts 5
    uv run main.py evolve "butterfly pattern" --generations 10
    uv run main.py view
    uv run main.py replay 1
    uv run main.py demo
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from lenia_search import Lenia, save_snapshot, PatternSearcher, check_pattern_match, run_evolution


def cmd_search(args):
    """Run pattern search with console output."""
    print(f"Searching for patterns matching: '{args.description}'")
    print(f"Target: {args.count} patterns, Max attempts: {args.attempts}")
    print("-" * 60)

    searcher = PatternSearcher(args.description, output_dir=args.output)

    def on_attempt(attempt, params, image_path, matches, explanation):
        status = "✓ MATCH" if matches else "✗"
        print(f"[{attempt:3d}] {status} | mu={params['mu']:.3f} sigma={params['sigma']:.3f} | {explanation[:50]}")

    patterns = searcher.search(
        target_count=args.count,
        max_attempts=args.attempts,
        callback=on_attempt
    )

    searcher.save_results()

    print("-" * 60)
    print(f"Search complete. Found {len(patterns)} matching patterns.")

    if patterns:
        print("\nFound patterns saved to:")
        for p in patterns:
            print(f"  - {p['image_path']}")

    return patterns


def cmd_view(args):
    """View saved patterns."""
    output_path = Path(args.output)
    results_file = output_path / 'search_results.json'

    if not results_file.exists():
        print(f"No results found in {args.output}")
        print("Run a search first: uv run main.py search 'your description'")
        return

    with open(results_file) as f:
        results = json.load(f)

    patterns = results.get('found_patterns', [])

    if not patterns:
        print("No patterns found in results.")
        return

    print(f"Found {len(patterns)} patterns matching: '{results['description']}'")

    n_patterns = len(patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(5 * n_patterns, 5))
    if n_patterns == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        img = plt.imread(pattern['image_path'])
        ax.imshow(img)
        ax.set_title(f"Pattern {pattern['id']}\nmu={pattern['params']['mu']:.3f}, sigma={pattern['params']['sigma']:.3f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def cmd_replay(args):
    """Replay a saved pattern as animation."""
    output_path = Path(args.output)
    params_file = output_path / f'pattern_{args.pattern_id}_params.json'

    if not params_file.exists():
        print(f"Pattern {args.pattern_id} not found in {args.output}")
        return

    with open(params_file) as f:
        pattern_data = json.load(f)

    params = pattern_data['params']
    print(f"Replaying pattern {args.pattern_id}")
    print(f"Parameters: mu={params['mu']:.4f}, sigma={params['sigma']:.4f}, "
          f"kernel_radius={params['kernel_radius']}, kernel_beta={params['kernel_beta']}")

    lenia = Lenia(size=128, dt=0.1, params=params)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(lenia.grid, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'Pattern {args.pattern_id}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    def animate(frame):
        lenia.step()
        im.set_array(lenia.grid)
        ax.set_title(f'Pattern {args.pattern_id} - Step {frame}')
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=args.steps, interval=50, blit=True)
    plt.show()


def cmd_evolve(args):
    """Run evolutionary pattern search."""
    run_evolution(
        description=args.description,
        max_generations=args.generations,
        population_size=args.population,
        output_dir=args.output,
        regen=args.regen_cands,
    )


def cmd_demo(args):
    """Run a quick demo of the Lenia simulation."""
    print("Running Lenia demo simulation...")

    lenia = Lenia(size=128, dt=0.1)
    print(f"Parameters: {lenia.params}")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(lenia.grid, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Lenia Demo')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    def animate(frame):
        lenia.step()
        im.set_array(lenia.grid)
        ax.set_title(f'Lenia Demo - Step {frame}')
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Lenia Pattern Search - Find cellular automaton patterns using VLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "circular blob pattern"
  %(prog)s search "worm-like elongated shape" --count 5 --attempts 100
  %(prog)s evolve "butterfly pattern" --generations 100
  %(prog)s evolve "spiral shape" --population 15 --generations 200
  %(prog)s view
  %(prog)s replay 1 --steps 1000
  %(prog)s demo
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for patterns matching a description (random sampling)')
    search_parser.add_argument('description', type=str, help='Pattern description to search for')
    search_parser.add_argument('-c', '--count', type=int, default=3,
                               help='Number of matching patterns to find (default: 3)')
    search_parser.add_argument('-a', '--attempts', type=int, default=50,
                               help='Maximum search attempts (default: 50)')
    search_parser.add_argument('-o', '--output', type=str, default='found_patterns',
                               help='Output directory (default: found_patterns)')

    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Evolutionary search using pairwise VLM comparisons')
    evolve_parser.add_argument('description', type=str, help='Pattern description to evolve towards')
    evolve_parser.add_argument('-g', '--generations', type=int, default=100,
                               help='Maximum generations (default: 100)')
    evolve_parser.add_argument('-p', '--population', type=int, default=10,
                               help='Population size (default: 10)')
    evolve_parser.add_argument('-o', '--output', type=str, default='evolved_patterns',
                               help='Output directory (default: evolved_patterns)')
    evolve_parser.add_argument('--regen-cands', action='store_true',
                               help='Regenerate candidates from scratch (ignore existing)')

    # View command
    view_parser = subparsers.add_parser('view', help='View saved patterns')
    view_parser.add_argument('-o', '--output', type=str, default='found_patterns',
                             help='Output directory to view (default: found_patterns)')

    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Replay a saved pattern as animation')
    replay_parser.add_argument('pattern_id', type=int, help='Pattern ID to replay')
    replay_parser.add_argument('-s', '--steps', type=int, default=500,
                               help='Number of animation steps (default: 500)')
    replay_parser.add_argument('-o', '--output', type=str, default='found_patterns',
                               help='Output directory (default: found_patterns)')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a demo simulation')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'search':
        cmd_search(args)
    elif args.command == 'evolve':
        cmd_evolve(args)
    elif args.command == 'view':
        cmd_view(args)
    elif args.command == 'replay':
        cmd_replay(args)
    elif args.command == 'demo':
        cmd_demo(args)


if __name__ == '__main__':
    main()
