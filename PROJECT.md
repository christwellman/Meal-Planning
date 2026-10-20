---
# MY OVERRIDES: I own this block. The weekly refresh never edits it.
# Leave a field blank to use the auto-discovered value.
# status: active | maintenance | stalled | experimental | done
# priority: high | medium | low
purpose:
status:
priority:
next_step:
notes:
hidden: false
---

<!-- AUTO-DISCOVERED:START (regenerated each refresh; put corrections in the frontmatter above) -->
# Meal-Planning

**Repo:** https://github.com/christwellman/Meal-Planning  ·  **Path:** `/Users/christwellman/Projects/Meal-Planning`  ·  **Stack:** Python · NLTK · gensim Word2Vec · Tkinter (spaCy in recipe script)

**Purpose:** Dinner Ideas Processor: cleans a list of family dinner ideas, trains Word2Vec and flags similar or duplicate ideas by cosine similarity (Tkinter GUI plus CLI). README states a broader goal of a recipe-standardizing RAG project for LLM meal planning, which is not implemented. _(confidence: high)_

**Status:** stalled _(confidence: medium)_ — evidence: Last substantive work was a one-off burst on 2026-10-18 (commits adding dinner ideas, processor, spec, PROJECT.md), but those were commits of long-existing local files (file mtimes 2025-04-12) rather than new development. No new development in months, and the RAG goal is untouched.

**Last meaningful activity:** 2026-10-18 — Committed Dinner Ideas Processor, more family dinner ideas and functional spec (locally, not pushed); earlier real development was 2024-09-09

**Local state:** branch `main`, 0 uncommitted, 4 unpushed, 0 behind

## Recent updates
- 2026-10-18: added PROJECT.md project summary (dff7d76)
- 2026-10-18: added functional specification for Dinner Ideas Processor
- 2026-10-18: added Dinner Ideas Processor (Word2Vec similarity suggestions)
- 2026-10-18: added more family dinner ideas and tweaked entries
- 2024-09-09: initial recipes, function docs and tweaks

## Outstanding work
- [branch] 4 local commits on main not pushed to origin (includes Dinner_Ideas.py, the main script) (main (ahead 4))
- [roadmap] LLM/RAG integration described in README is not implemented; build it or trim the README (README.md)
- [other] process-recipes.py looks unfinished: NER trained on 2 toy examples, and it reads 'Recipes/' while the folder is 'recipes/' (fails on case-sensitive filesystems) (process-recipes.py:88)
- [other] Only 3 sample recipes; recipe standardization is minimal (recipes/)
- [other] Stale remote branch origin/master alongside origin/main (origin/master)

## Suggested next step
Push main to origin so Dinner_Ideas.py and the spec are backed up on GitHub, then decide whether to build the RAG piece or trim the README. _(confidence: high)_

## Findings
- Main script Dinner_Ideas.py, the spec and new dinner ideas exist only in 4 unpushed local commits, so they are not backed up on GitHub until pushed.
- Bash was unavailable, so git log and branch output were derived from .git/logs/HEAD and packed-refs, and Dinner_Ideas.py was not read.
- No secrets found in the files read.

_Analyzed 2026-10-19._
<!-- AUTO-DISCOVERED:END -->

## Previous notes (preserved verbatim from the earlier audit — edit freely)

## Goal
A simple NLP/ML tool ("Dinner Ideas Processor") that preprocesses a text file of dinner ideas (tokenizing, removing stopwords, stemming/lemmatizing), trains a Word2Vec model on them, and uses cosine similarity to suggest similar dinner ideas for meal planning — with both a Tkinter GUI and a command-line fallback. The README also describes it as a "Simple ML RAG Project to standardize recipe formats and allow them to be referenced by LLM for Meal Planning."

## Status
**Stalled.** Last commit (`3be75ec`, "minor tweaks") was 2024-09-09 in git history, though the working-directory files show later modification dates (April 2025) suggesting uncommitted local edits after that. Either way, there's been no activity in well over a year relative to today (2026-08-30). The core NLP pipeline (`Dinner_Ideas.py`) and output files (`similar_ideas.txt`, `family_dinner_ideas.txt`) exist and look complete for a first version, but the stated "RAG"/LLM-reference goal from the README doesn't appear implemented — the code is classic Word2Vec + cosine similarity, not a retrieval-augmented LLM pipeline.

## Remaining TODOs
- No TODO/FIXME comments found in the Python source.
- The README's ambition ("allow them to be referenced by LLM for Meal Planning") is not reflected in the code — there's no LLM integration in `Dinner_Ideas.py`, `process-dinner-ideas.py`, or `process-recipes.py`. This gap should either be built out or the README description scaled back.
- Only 3 recipe files exist under `recipes/` (Brown Butter Blueberry Crumble, Homemade Pita Bread, Smash Banh Mi Tacos) — a small dataset relative to the "standardize recipe formats" ambition in the README.
- Working-tree files newer than the last commit (dated April 2025 vs. commit date September 2024) suggest uncommitted local changes that were never checked in — worth reviewing with `git status`/`git diff` and committing or discarding.

## Functional Tests
- **Test**: Run `python Dinner_Ideas.py family_dinner_ideas.txt` (or with GUI available, launch it directly) with no arguments/threshold.
  **Expected**: With no CLI args, usage help prints (`Usage: python Dinner_Ideas.py <input_file> [threshold] [output_file]`); with a valid input file, the script tokenizes/cleans the ideas, trains a Word2Vec model, and prints "Processing dinner ideas from '<file>' with similarity threshold 0.7...".
- **Test**: Run `python Dinner_Ideas.py family_dinner_ideas.txt 0.7 test_output.txt`.
  **Expected**: Script prints the original dinner ideas list, a unique-ideas count, and any similar ideas above the 0.7 threshold with similarity scores; results are exported to `test_output.txt`.
- **Test**: Run `python Dinner_Ideas.py family_dinner_ideas.txt 0.99`.
  **Expected**: With an unusually high threshold, the script reports "No similar ideas found with current threshold" for most/all entries rather than crashing.
- **Test**: Launch the script on a machine with Tkinter available (no CLI args needed).
  **Expected**: A GUI window opens (per `GUI_AVAILABLE` check) allowing file selection and threshold configuration instead of falling back to CLI mode.
- **Test**: Run `python process-recipes.py` against the sample files in `recipes/`.
  **Expected**: The three `.txt` recipe files are read and processed/standardized without errors (exact output format should be verified by reading `process-recipes.py`, which wasn't inspected in this audit).
