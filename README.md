# Fin-Chat example (LangGraph)

## Installation

```bash
poetry install
```

Create a file `.env` with

```bash
OPENAI_API_KEY="...."
FMP_API_KEY="...."
ARIZE_API_KEY="...."
ARIZE_SPACE_ID="...."
ARIZE_PROJECT_NAME="live-fin-langgraph"
```

Where `FMP_API_KEY` is the API obtained from [FMP API](https://site.financialmodelingprep.com/developer/docs).

## Quickstart

The finchat has two modes `interactive` and `eval`

### Interactive

In interactive mode you can ask your questions in a chat-like fashion.

```bash
poetry run finchat --interactive
```

Each question will create a new trace with the same run-id.

### Run Verification with Agent Contracts

You can run the pre-defined questions in `specifications.json` with the following command.

```bash
poetry run finchat --eval
```

If you want to run verification using [agent-contracts](https://github.com/relari-ai/agent-contracts), please follow the documentation here: [Agent Contracts - Finance Agent Example](https://agent-contracts.relari.ai/examples/finance-agent).

## PR Arize experiments

On every pull request to `main`, GitHub Actions runs the fin agent against the Arize dataset **`2026-ipo-stock-questions`** (space **Live Demo**) and creates a new experiment via the `ax` CLI. Results appear on the workflow run **Summary** tab.

### Required GitHub Actions secrets

| Secret | Purpose |
|--------|---------|
| `ARIZE_API_KEY` | Arize AX API key |
| `ARIZE_SPACE_KEY` | Arize space ID for Live Demo |
| `OPENAI_API_KEY` | LLM calls (`gpt-4o-mini`) |
| `FMP_API_KEY` | Financial Modeling Prep tools |

### Local / manual experiment

```bash
export ARIZE_API_KEY=... ARIZE_SPACE=... OPENAI_API_KEY=... FMP_API_KEY=...
ax profiles create default --api-key "$ARIZE_API_KEY"
poetry run python scripts/run_ipo_dataset_experiment.py --output runs.json
ax experiments create -n "local-ipo-run" \
  --dataset 2026-ipo-stock-questions \
  --space "$ARIZE_SPACE" \
  -f runs.json
```
