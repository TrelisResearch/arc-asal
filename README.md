# Simple ASAL

>![IMPORTANT]
> DEVELOPER NOTES: a) Keep in touch with Ronan via WhatsApp re progress, b) Ronan will provide written feedback at least once per month on progress / quality and will add/remove/change repo access based on quality standards/progress c) if you no longer have time to advance research, let Ronan know, d) developers who have contributed will be credited in any X / Youtube posts/videos, e) work in a new branch and make a PR if/when you wish to push. See [here](https://github.com/TrelisResearch/style-guide) for a very short style guide.

Inspired by: https://sakana.ai/asal/
Recommended reading (also find the ARC playlist on TrelisResearch and watch the interview with Puget): https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view

The motivation is to use the asal approach to try and create new, difficult, ARC tasks.

Specifically:
- Given an ARC task (i.e. training example input + output grid pairs, and test input grids), create some similar and related tasks.
- There should be a task generator, and then there should be a task verifier that checks whether the generated tasks are a) ARC-like and b) related to the task at hand.

In this repo, if you run `uv run main.py`, you can start some loops to generate specific objects using cellular-automata type approaches, and then use Gemini as a verifier of what is generated (note that it doesn't quite work yet).

Guidance on ARC:
- The point here is not just to generate pre-training data, but to be able to bootstrap tasks that are similar to a target task at test time. Then, one could train on those and potentially have a better chance of solving that test time task.
- It may be wise to try and get the simple evolution approach working here to generate things like butterfly patterns first.
- The key here is to figure out ways to build generator functions and evolve them somehow, then there's a need to have a verifier function (perhaps a vision model OR some python tests like done by NVARC) to check they are ARC-like. This isn't easy.