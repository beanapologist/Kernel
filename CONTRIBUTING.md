# Contributing to the Lean-Verified Mathematical Universe

Thank you for your interest in contributing!  This project aims to build
a complete, machine-verified mathematical universe using Lean 4 and
Mathlib.  Every theorem must be accepted by the Lean type-checker with
**zero `sorry` placeholders**.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [elan](https://github.com/leanprover/elan) | latest | Lean version manager |
| Lean 4 | v4.14.0 (pinned in `formal-lean/lean-toolchain`) | Type-checker |
| [lake](https://github.com/leanprover/lake) | bundled with Lean | Build system |
| Mathlib4 | v4.14.0 (pinned in `formal-lean/lakefile.lean`) | Math library |

```bash
# Install elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Download the Mathlib build cache (saves ~1 hour of compilation)
cd formal-lean/
lake exe cache get

# Build everything
lake build
```

---

## Adding a New Theorem

1. **Choose or create a module file** under `formal-lean/`.
   - Use an existing file (e.g. `CriticalEigenvalue.lean`) for closely
     related results.
   - Create a new `formal-lean/MyTopic.lean` for a new domain.

2. **Follow the file header convention**:
   ```lean
   /-
     MyTopic.lean — Brief description.
   
     Sections
     ────────
     1.  First group of theorems
     2.  Second group of theorems
   -/
   import Mathlib
   -- or more specific imports, e.g. import Mathlib.Analysis.SpecialFunctions.Complex.Circle
   
   namespace MyTopic
   ...
   end MyTopic
   ```

3. **Write your theorem with a complete proof** — no `sorry` allowed:
   ```lean
   theorem my_theorem (x : ℝ) (hx : 0 < x) : 0 < x ^ 2 := by
     positivity
   ```

4. **Register the new module** in `formal-lean/lakefile.lean`:
   ```lean
   lean_lib «FormalLean» where
     roots := #[..., `MyTopic]
   ```

5. **Add an entry point** in `src/<domain>/MyTopic.lean`:
   ```lean
   import FormalLean.MyTopic
   ```

6. **Update `src/MathUniverse.lean`** to include the new import.

7. **Update `docs/overview.md`** — add the module to the domain table.

---

## Coding Standards

- **No `sorry`** — the CI workflow (`lean-proof-check.yml`) rejects any
  build containing a `sorry`.
- **Descriptive theorem names** — use `snake_case` following Mathlib conventions.
- **Inline documentation** — add a `/-` block comment above each `theorem`
  explaining what it states and why it matters.
- **Section comments** — group related theorems with `-- § N. Title` banners.
- **Minimal imports** — prefer specific Mathlib imports over `import Mathlib`
  where build time is a concern.

---

## Submitting a Pull Request

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-topic
   ```
2. Make your changes and verify locally:
   ```bash
   cd formal-lean/
   lake build
   lake exe formalLean
   ```
3. Push your branch and open a PR against `main`.
4. The CI workflow will automatically build and verify all proofs.
5. Address any reviewer feedback.

---

## Reporting Issues

- **Bug in an existing proof?** Open an issue describing which theorem is
  wrong and what the correct statement should be.
- **New mathematical direction?** Open an issue with a proposal describing
  the domain, expected theorems, and any relevant literature.
- **Documentation improvement?** PRs to `docs/` are always welcome.

---

## Code of Conduct

Be respectful and constructive.  Mathematical discourse should be
rigorous but welcoming — remember that everyone is learning.

---

## License

By contributing you agree that your contributions will be licensed under
the [MIT License](LICENSE).
