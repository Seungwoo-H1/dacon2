# TOOLS.md - Local Notes

## Code Analysis Pipeline

### How Multi-Agent Works

When 승우 asks for code analysis, use `sessions_spawn` to dispatch internal specialists:

1. **기획 (Architecture)** — Understand intent, structure, dependencies
2. **개발 (Engineering)** — Deep code analysis, bugs, fixes
3. **Audit (Security/Quality)** — Vulnerabilities, anti-patterns, performance

Results are collected and synthesized into a single evidence-based report.

### Dispatch Pattern

```bash
# Use sessions_spawn with context="isolated" for each specialist
# Role is defined in the task parameter
# Collect results via auto-announcement
```

### Evidence Standard

Every conclusion must reference:
- Specific file + line numbers (if available)
- Code snippets (truncated, relevant parts only)
- Documentation links (when applicable)
- Security standards (CWE, OWASP, etc. for audit findings)

### Response Format

```
## 결론 (한 줄 요약)

## 분석

### [분야]
- 발견사항: ...
- 근거: `code snippet` (파일:줄번호)

## 권장 사항

1. ...
2. ...
```

## Memory

- Daily notes: `memory/YYYY-MM-DD.md`
- Long-term: `MEMORY.md`
