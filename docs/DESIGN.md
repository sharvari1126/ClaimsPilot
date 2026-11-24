# Design decisions

**Why an auditor, not a claims bot?** Auditing gives ground truth. Plant known errors in settled claims and you
can measure whether they are caught. A triage bot has no such check beyond labels you wrote yourself.

**Why is the core deterministic?** Payouts must be exact, reproducible and explainable to a regulator or customer.
An LLM answer that is usually right is the wrong tool for money. The LLM reads text and writes prose.

**Decision order (engine.py).** Hard denials first (policy period, cover, exclusions), then missing facts, then
fraud escalation (2+ flags), then the payout. A denial outranks missing information because a denial ground that
is already proven does not need more facts; missing information outranks fraud flags because flags need dates.

**Why hypothesis testing for root cause?** "The LLM thinks the adjuster forgot the deductible" is unverifiable.
"Ignoring the deductible reproduces the paid amount exactly" is checkable. Several mistakes can coincide on one
amount, so all matching names are returned.

**Integer rupees.** No floating point in money. Percentages use integer arithmetic with half-up rounding.

**Why a validator on the memo?** The memo is the one place the LLM writes free text about money. Requiring every
number and clause id to appear in a tool output turns "trust the model" into a mechanical check, with a template
fallback after one repair attempt.

**Why keep a regex extractor?** It is the baseline the LLM must beat and lets the app and tests run offline.
Its 0/10 on messy notes is the evidence that extraction needs an LLM.

**Known weak spots.** The payout cap (S3) cannot trigger under the 75% total-loss rule; it is a safety net.
Fraud thresholds are fixed constants, not learned. The benchmark errors are self-defined.
