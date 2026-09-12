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
