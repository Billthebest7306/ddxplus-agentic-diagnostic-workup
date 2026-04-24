# Baseline Results And Next Steps

## Current Status

The project now has a proper baseline ladder scaffold instead of a single notebook:

1. strongest one-shot classifier suite
2. single-agent sequential workup baseline
3. paired comparison notebook
4. later multi-agent evidence-ledger system

The implemented notebooks are:

- [01_one_shot_classifier_baselines.ipynb](notebooks/01_one_shot_classifier_baselines.ipynb)
- [02_single_agent_sequential_baseline.ipynb](notebooks/02_single_agent_sequential_baseline.ipynb)
- [03_compare_baselines.ipynb](notebooks/03_compare_baselines.ipynb)

## What The Best Available Results Are Right Now

The strongest *available* empirical one-shot numbers in the folder still come from the archived full differential-trained run:

- [metrics.json](artifacts/_legacy/direct_dx_full/metrics.json)
- [predictions.csv](artifacts/_legacy/direct_dx_full/predictions.csv)
- [qualitative_examples.json](artifacts/_legacy/direct_dx_full/qualitative_examples.json)
- [confusion_summary.csv](artifacts/_legacy/direct_dx_full/confusion_summary.csv)

Those metrics are:

- accuracy: `31.81%`
- top-3 accuracy: `51.59%`
- top-5 accuracy: `63.79%`
- macro-F1: `31.24%`
- DDR: `93.27%`
- DDP: `40.07%`
- DDF1: `51.19%`

This archived run is effectively the old `basd_differential` one-shot baseline. The new notebook suite is set up so you can rerun that objective under the new `artifacts/one_shot/` layout and compare it directly against `basd_pathology` and `basd_joint`.

## Why This Is Good Enough For The Baseline Stage

The current one-shot model is worth keeping because it clears the assignment bar and gives the project a meaningful non-agentic reference point:

- it is a real trained model on the actual DDXPlus split
- it solves a 49-class diagnosis problem from sparse initial evidence only
- it performs far above a trivial majority-class baseline
- it already uses the case-episode abstraction that the later sequential and agentic systems will reuse

That said, it should not be the *only* benchmark in the final project story. If the final claim is that an agentic workup system is better, then the comparison set must include at least one stronger pre-agentic baseline than a single-pass classifier.

## Why The New Ladder Matters

The logic of the project is now:

- one-shot classifier asks how far a serious static model can go with only `INITIAL_EVIDENCE`
- single-agent sequential baseline asks what additional evidence gathering buys you without multi-agent coordination
- later multi-agent system asks whether coordination through an evidence ledger beats a single workup agent

That is the fair comparison sequence. It avoids the weak claim of beating only a strawman model.

## How The Work Aligns With The Agentic Medical Idea

The broader project idea is still:

> a multi-agent diagnostic workup copilot that begins with incomplete information, gathers additional evidence over time, tracks a structured evidence ledger, and then finalizes a diagnosis more robustly than a one-shot system.

The current notebooks align with that idea in a staged way:

- notebook 01 builds the strongest non-agentic one-shot comparator
- notebook 02 builds the sequential case environment and the single-agent workup baseline
- notebook 03 builds the paired evaluation harness you will reuse later for the multi-agent system

So the project has not yet reached the full agentic stage, but it is no longer “just a classifier notebook.” The environment, evidence ledger scaffold, and paired evaluation path are now in place.

## What Is Still Missing From The True Agentic System

The current implementation does **not** yet include:

- multiple cooperating agents
- coordinator logic
- safety/audit role separation
- evidence-gated planner control
- shared diagnosis-specific support vs contradiction tracking

That is intentional. The single-agent sequential baseline should come first so the later multi-agent comparison is defensible.

## Recommended Next Execution Order

1. Run notebook 01 in `full` mode and let it select the strongest one-shot comparator under `artifacts/one_shot/`.
2. Run notebook 02 in dry-run mode first to check prompts and action formatting.
3. Fill the API placeholders and run notebook 02 in live mode on the 245-case sampled benchmark.
4. Run notebook 03 to get the paired one-shot vs sequential comparison.
5. Only after that, begin the multi-agent evidence-ledger system.

## Bottom Line

The project is now in the right state:

- one-shot baseline suite implemented
- sequential single-agent baseline implemented
- comparison harness implemented
- multi-agent method still to come

That is the correct order if you want the final agentic claim to be technically credible.
