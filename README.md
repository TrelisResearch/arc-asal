# Simple ASAL

Inspired by: https://sakana.ai/asal/

The motivation is to use the asal approach to try and create new, difficult, ARC tasks.

Specifically:
- Given an ARC task (i.e. training example input + output grid pairs, and test input grids), create some similar and related tasks.
- There should be a task generator, and then there should be a task verifier that checks whether the generated tasks are a) ARC-like and b) related to the task at hand.

In this repo, if you run `uv run main.py`, you can start some loops to generate specific objects using cellular-automata type approaches, and then use Gemini as a verifier of what is generated (note that it doesn't quite work yet).