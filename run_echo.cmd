@echo off
rem Echo backend (no language model) for UI / graph work. Uses config.dev.json: same clips, persona and lexicon, but
rem its own scratch memory, state and log under data_dev\ so it never touches her real memories in data\.
cd /d "%~dp0"
call conda run -n embodied --no-capture-output python -m embodied3.server --config config.dev.json
