# CI/CD Pipeline — Quantum Kernel

This document describes the automated CI/CD pipeline for the Quantum Kernel
repository and explains how to extend it.

## Overview

The pipeline is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
and runs four parallel jobs on every push or pull request that touches a C++
source or header file, or the workflow file itself.

```
push / pull_request (**.cpp, **.hpp, ci.yml)
        │
        ├── lint           (clang-format check)
        ├── build-and-test (compile + run all test suites)
        ├── benchmarks     (NIST IR 8356 + PoW nonce-search)
        └── coverage       (gcov / lcov report, uploaded as artifact)
```

## Trigger Conditions

The workflow triggers on:

| Event           | Branch filter | Path filter                           |
|-----------------|---------------|---------------------------------------|
| `push`          | `main`        | `**.cpp`, `**.hpp`, `.github/workflows/ci.yml` |
| `pull_request`  | `main`        | `**.cpp`, `**.hpp`, `.github/workflows/ci.yml` |

Only commits that modify C++ source files or the workflow itself trigger a
run, keeping CI minutes usage minimal for documentation-only changes.

## Jobs

### 1. `lint` — clang-format check

Verifies that all C++ source and header files conform to `clang-format`
style rules. The step uses `--dry-run --Werror` so the job fails as soon
as any formatting deviation is detected without modifying files in CI.

**To fix locally:**
```bash
clang-format -i quantum_kernel_v2.cpp qudit_kernel.cpp \
    test_pipeline_theorems.cpp test_ipc.cpp test_qudit_kernel.cpp \
    test_interrupt_nist.cpp benchmark_nist_ir8356.cpp \
    ChiralNonlinearGate.hpp LadderChiralSearch.hpp LadderSearchBenchmark.hpp
```

### 2. `build-and-test` — Correctness tests

Compiles and runs every test suite against the hybrid kernel implementation.
Each test binary exits with a non-zero status code on failure, which fails
the CI job.

| Binary                  | Source file                    | What it validates                          |
|-------------------------|--------------------------------|--------------------------------------------|
| `test_pipeline_theorems`| `test_pipeline_theorems.cpp`   | 56 formal theorems (Theorems 3–14, Prop 4) |
| `test_ipc_bin`          | `test_ipc.cpp`                 | 119 IPC correctness tests                  |
| `test_qudit_kernel`     | `test_qudit_kernel.cpp`        | 169 qudit kernel tests                     |
| `test_interrupt_nist`   | `test_interrupt_nist.cpp`      | NIST-recommended interrupt/recovery tests  |

Each test suite uses its own `assert`-based pass/fail counter and prints a
human-readable summary; the process exits with `EXIT_FAILURE` when any test
fails.

### 3. `benchmarks` — Performance vs brute-force

Runs the benchmark suites that compare the hybrid kernel approach against
classical brute-force baselines. The benchmarks are informational (they
always exit 0) but their stdout output is captured in the job log for
review.

| Binary                     | Source file                      | What it measures                                      |
|----------------------------|----------------------------------|-------------------------------------------------------|
| `benchmark_nist_ir8356`    | `benchmark_nist_ir8356.cpp`      | Process-spawn, scheduling, memory, coherence at scale |
| `benchmark_pow_nonce_search`| `benchmark_pow_nonce_search.cpp` | Hybrid ladder vs brute-force Bitcoin PoW nonce search |

The PoW benchmark requires OpenSSL (`libssl-dev`) which is installed in the
job before compilation.

### 4. `coverage` — Code coverage report

Compiles the four test binaries with GCC's `--coverage` flag (`gcov`),
runs them to collect `.gcda` data, then uses `lcov` to produce a merged
coverage summary.  The `coverage.info` file is uploaded as a workflow
artifact named **coverage-report** and can be downloaded from the GitHub
Actions summary page.

The coverage step filters out system headers (`/usr/*`) so only
project-owned lines are counted.

## Dependencies

All jobs run on `ubuntu-latest`.  External packages installed at job start:

| Package          | Used by                         |
|------------------|---------------------------------|
| `clang-format`   | `lint` job                      |
| `libssl-dev`     | `build-and-test`, `benchmarks`  |
| `libeigen3-dev`  | `build-and-test`, `benchmarks`  |
| `lcov`           | `coverage` job                  |

## Extending the Pipeline

### Adding a new test file

1. Create `test_<name>.cpp` in the repository root following the style of
   `test_pipeline_theorems.cpp` (internal pass/fail counter, non-zero exit
   on failure).
2. Add two steps to the `build-and-test` job in `ci.yml`:
   ```yaml
   - name: Build test_<name>
     run: g++ -std=c++17 -Wall -Wextra -O2 -o test_<name> test_<name>.cpp -lm

   - name: Run test_<name>
     run: ./test_<name>
   ```
3. Add the same binary to the `coverage` job build and run steps.

### Adding a new benchmark

1. Create `benchmark_<name>.cpp` following the style of
   `benchmark_nist_ir8356.cpp`.
2. Add build and run steps to the `benchmarks` job.

### Changing the trigger paths

Edit the `paths:` lists under `on.push` and `on.pull_request` in
`ci.yml`.  GitHub Actions path filters support glob patterns (`**`).

### Publishing coverage to a service (e.g. Codecov)

Replace the **Upload coverage report** step in the `coverage` job with:
```yaml
- name: Upload to Codecov
  uses: codecov/codecov-action@v4
  with:
    files: coverage.info
```
And add `CODECOV_TOKEN` to the repository secrets.
