## Summary

<!-- What does this change, and why? Link any related issue. -->

## Checklist

- [ ] Tests added or updated for the behaviour change, and the full suite passes (`pytest`).
- [ ] Formatted with `black --target-version py311 src tests examples scripts`.
- [ ] If a wire model changed, `schema/hmcp-0.1.schema.json` was regenerated (`python scripts/generate_schema.py`).
- [ ] Documentation and the `CHANGELOG.md` `[Unreleased]` section are updated.
- [ ] No PHI, real patient data, credentials, tokens or secrets anywhere in the change.
- [ ] No patient data, tokens or response bodies in logs or exception messages, and no chained sensitive exceptions.

## Notes for reviewers

<!-- Anything a reviewer should know: risk, rollout, follow-ups. -->
