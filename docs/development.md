# Development checks

Run from the repository root:

```sh
just                  # show available checks
just check-fixtures   # Engram capture integrity and corruption tests
just check            # CPU tests, Clippy, rustdoc, script tests and formatting
just check-metal      # the same gate including Metal features on Apple Silicon
```

`just` is optional. The canonical gate remains `uv run scripts/check.py`
(`--metal` includes Metal); the recipes only call existing scripts. Checks do
not download model checkpoints. Build and Python tooling must be installed;
dependency resolution may still access registries.

The fixture checker covers only the Engram hash and preprojected residual-gate
JSON captures. It checks tensor names, shapes, encodings, value ranges and
little-endian payload hashes. This catches accidental fixture corruption, not
an incorrect reference implementation or a maliciously regenerated hash.
Rust parity tests remain the numerical gate. Source provenance and regeneration
instructions are in [the Engram reference](research/v41-engram.md).

## Compiler-aware custom lints

[Dylint](https://github.com/trailofbits/dylint) can load custom Rust lint
libraries. It is a candidate for rules needing resolved types or compiler
analysis, not a dependency of the current gate. Its libraries are tied to a
compiler toolchain, and Dylint attempts to build checked packages with that
toolchain ([mechanism](https://github.com/trailofbits/dylint/blob/master/docs/how_dylint_works.md)).

Add it when a concrete invariant has both violating and acceptable code
examples that Clippy or the type system cannot adequately enforce. For
example, detecting a known host-materialization operation in an explicitly
marked decode path could warrant a prototype. A lint would not prove that an
entire call graph is allocation-free or that a GPU kernel is numerically
correct. Pin and test the lint toolchain before making such a rule mandatory.
