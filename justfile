# List the available development checks.
default:
    @just --list

# Run the canonical CPU quality gate without model downloads.
check:
    uv run scripts/check.py

# Include Apple Silicon Metal feature checks.
check-metal:
    uv run scripts/check.py --metal

# Verify the two Engram tensor captures and exercise corruption detection.
check-fixtures:
    uv run scripts/check_engram_fixtures.py
    uv run scripts/test_check_engram_fixtures.py

# Qualify real agent clients against a local model (dry-run unless --run).
e2e-agents mx model model_id='qwen3-4b' *args='':
    uv run scripts/e2e_agents.py --mx {{mx}} --model {{model}} --model-id {{model_id}} {{args}}

# Compile a libFuzzer target with nightly sanitizers while preserving the
# configured rustc wrapper and shared cache.
fuzz-check target='decode_affine_row':
    RUSTUP_TOOLCHAIN=nightly RUSTC="$(rustup which rustc --toolchain nightly)" cargo fuzz check --fuzz-dir fuzz {{target}}
