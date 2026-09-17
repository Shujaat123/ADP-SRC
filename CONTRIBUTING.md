# Contributing

Thank you for helping improve ADP-SRC.

## Before submitting a change

1. Create a branch from the current default branch.
2. Install the development environment with `python -m pip install -e ".[test]"`.
3. Keep all learned preprocessing and model selection inside the relevant training fold.
4. Add or update tests for behavioral changes.
5. Run:

```bash
python -m pytest -q
python scripts/verify_revised_release.py
```

## Result changes

Changes that alter saved metrics must include:

- the exact command used;
- software and hardware details;
- updated fold-, repeat-, aggregate-, and OOF-level outputs where applicable;
- a clear explanation of whether the benchmark, split seeds, or selection rule changed;
- updated README and model-card summaries where applicable.

Do not overwrite frozen release results with exploratory output. Write exploratory runs to a new results directory first.

## Data and labels

Do not describe the historical AVPdb-derived control class as experimentally confirmed antidiabetic negatives. New labels must retain their evidence provenance and assay scope.

## Pull requests

Keep pull requests focused. Describe the motivation, implementation, verification commands, and any scientific interpretation affected by the change.
