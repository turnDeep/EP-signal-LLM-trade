# modules/ - Delegation Layer

This layer defines five AI employees for Codex-style team operation. Each module
has a clear responsibility and a small output contract. Use one owner by
default; combine modules only when the task needs handoff or independent review.

| Employee | File | Owns |
|---|---|---|
| 01 Spec Steward | `01-spec-steward.md` | scope, requirements, project memory |
| 02 Research Analyst | `02-research-analyst.md` | strategy tests, data interpretation |
| 03 Implementation Engineer | `03-implementation-engineer.md` | code changes |
| 04 Code Reviewer | `04-code-reviewer.md` | risk, regression, maintainability review |
| 05 Test Operator | `05-test-operator.md` | validation, commands, release readiness |

For actual parallel subagent work, split ownership by files or questions and
avoid overlapping write scopes.

