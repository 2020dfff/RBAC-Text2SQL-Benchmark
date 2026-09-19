# Evaluation changelog

## 2026-09-19 — protocol v2 correction

- Check predicted SQL against the sample policy for ALLOW-labeled requests.
  Move forbidden outputs from C/W to VC/VW without changing their EX judgment.
- Keep reference-DENY non-refusals in VC/VW and report actual SQL-policy
  violations separately from failure-to-refuse.
- Count the SafeEX denominator directly from saved ALLOW labels.
- Share classification/sampling/reporting across column, CRUD, and schema
  exposure entrypoints; retain an explicitly selected v1 reproduction path.
- Reject misaligned inputs; record original indices, hashes, migration counts,
  missing EX and unresolved policy checks rather than silently guessing.
- Add offline replay and regression tests. No model generation is required.

**Results status:** published results predating this correction use v1 unless
explicitly identified otherwise. Full revised tables are pending replay of
the corresponding original outputs. Dataset labels are unchanged by this patch.
See [protocol and migration instructions](EVALUATION_PROTOCOL.md), including
checker coverage limitations and incomplete-result handling.
