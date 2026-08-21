<!-- Modified for the MIRAI project, 2026. -->
# MIRAI: Bridging Planning and Execution in Dynamic Agentic RAG

MIRAI (Multi-agent Integrated Retrieval, Adaptation, and Inference) is a dynamic Agentic RAG implementation that coordinates a high-level Planner with specialized Executors for query decomposition, rewriting, retrieval, document selection, answer generation, and final synthesis. Trainable roles share one language-model policy and learn jointly from a unified experience buffer.

> Research status: the repository exposes the implementation and configuration surface needed to study the retained source setup. It does not claim that the published benchmark results were independently reproduced by the MIRAI team.

## Architecture

MIRAI executes a repeated three-stage loop:

1. **Plan:** select the first unresolved trace node and generate a role workflow.
2. **Execute:** run decomposition or solving roles in topological order.
3. **Update:** write new sub-nodes or answers to the global trace and continue until final synthesis.

Training flattens the hierarchical round/step trajectory into a unified buffer, computes GAE advantages, and updates the shared policy with PPO. A terminal F1-and-cost reward coordinates roles; local format penalties enforce role output contracts.

## Agent Roles

| Role | Function |
|---|---|
| Planner | Chooses a query-dependent workflow. |
| QDS | Produces serial, dependent sub-questions. |
| QDP | Produces independent, parallel sub-questions. |
| QR | Rewrites a question for retrieval. |
| RA | Calls the configured frozen retrieval service. |
| DS | Filters candidate documents. |
| AG | Answers from supplied evidence. |
| AS | Synthesizes the final answer from the completed trace. |

## Repository Layout

- `qa_manager/`: MIRAI Planner-Executor prompts, orchestration helpers, retrieval integration, and metrics.
- `examples/ppo_trainer/run_qwen2.5.sh`: environment-configured MIRAI training launcher.
- `examples/ppo_trainer/run_qwen2.5_test.sh`: environment-configured evaluation launcher.
- `verl/`: retained third-party reinforcement-learning framework and MIRAI integration points.
- `recipe/`, `examples/`, `tests/`, `docs/`: upstream training recipes, examples, tests, and framework documentation.

## Installation

Python 3.10 or newer is required. Most training workflows require Linux, CUDA, Ray, and substantial GPU memory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For lightweight configuration and unit tests without installing GPU extras:

```bash
python -m pip install openai jinja2 requests pytest
python -m pytest tests/mirai
```

Copy the configuration template and fill local values without committing it:

```bash
cp .env.example .env
```

## Data and Retrieval Setup

Set training, validation, model, and retrieval locations through environment variables:

```bash
export MIRAI_TRAIN_DATA=/path/to/train.parquet
export MIRAI_VAL_DATA=/path/to/validation.parquet
export MIRAI_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export MIRAI_CRITIC_PATH=Qwen/Qwen2.5-7B-Instruct
export MIRAI_RETRIEVAL_URLS=http://127.0.0.1:8000/search
```

The retrieval service accepts a JSON request containing `questions` and `N` and returns `top_k_docs`. The retained benchmark setup used an English Wikipedia index with E5 embeddings and top-five retrieval; datasets and indexes are not bundled.

## Model API Configuration

MIRAI's optional OpenAI-compatible client reads configuration from the environment:

```bash
export OPENAI_API_KEY=your-runtime-key
export OPENAI_BASE_URL=https://api.openai.com/v1
export MIRAI_LLM_MODEL=gpt-4o-mini
```

Local OpenAI-compatible endpoints may set `OPENAI_BASE_URL` and omit a real key when the server does not require authentication. Never place credentials in launch scripts.

## Training

The primary launcher is safe by default and does not enable shell tracing:

```bash
bash examples/ppo_trainer/run_qwen2.5.sh
```

Override Hydra options by appending them to the command. Set `MIRAI_LOGGER=wandb` and provide `WANDB_API_KEY` only in the environment if WandB logging is desired.

## Evaluation

```bash
bash examples/ppo_trainer/run_qwen2.5_test.sh
```

The launcher validates required paths before starting. Full benchmark evaluation requires the original datasets, retrieval index, and GPU environment; it is not part of the lightweight test suite.

## Validation

```bash
python -m compileall -q qa_manager verl
bash -n examples/ppo_trainer/run_qwen2.5.sh
bash -n examples/ppo_trainer/run_qwen2.5_test.sh
python -m pytest tests/mirai
MIRAI_PYTHON_BIN=echo MIRAI_TRAIN_DATA=/tmp/train.parquet MIRAI_VAL_DATA=/tmp/val.parquet \
  bash examples/ppo_trainer/run_qwen2.5.sh
```

## License and Third-Party Notices

MIRAI is distributed under Apache License 2.0. It is a modified derivative of the source identified in `UPSTREAM.md` and retains required notices in `NOTICE`, `Notice.txt`, and `THIRD_PARTY_NOTICES.md`. The `verl` package name and third-party contributor identities are retained because they are genuine dependency and attribution information.

The public MIRAI history begins from a sanitized snapshot to prevent a credential present in upstream history from being redistributed. This does not remove or conceal the upstream provenance.

## Citation

Citation metadata is provided in `CITATION.cff`. Please also cite the scholarly and software foundations identified in the MIRAI paper and `THIRD_PARTY_NOTICES.md`.
