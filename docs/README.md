# Documentation index

| document | what it answers |
|---|---|
| [architecture.md](architecture.md) | How the system is built, and *why* each choice was made over the alternatives that were considered. |
| [experiments.md](experiments.md) | The methodology, every measured result, and the exact commands to reproduce them. |
| [research_questions.md](research_questions.md) | The ten questions the project set out to answer, and what the measurements actually said — including where the answer is "no". |
| [telemetry.md](telemetry.md) | The signal schema, what each source provides, and how to add a new signal. |
| [limitations.md](limitations.md) | What these results do and do not support. Read this before quoting a number. |

## Reading order

**To evaluate the project:** `../README.md` → `research_questions.md` →
`limitations.md`. That is the claim, the evidence, and the caveats, in that
order.

**To work on the code:** `architecture.md` → `telemetry.md`, then the module
docstrings, which carry the reasoning for individual decisions.

**To reproduce the numbers:** `experiments.md`, which lists every command.
