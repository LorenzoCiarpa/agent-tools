# uvt-core

Shared, lightweight building blocks for the tools in
[agent-tools](https://github.com/LorenzoCiarpa/agent-tools). It has no ML dependencies.

- `uvt_core.config`: TOML config in the user config dir, with `UVT_*` env overrides
- `uvt_core.logging`: logging to stderr (stdout is reserved for results)
- `uvt_core.device`: CUDA / Apple Silicon / CPU detection without torch
- `uvt_core.output`: JSON helpers and the stdout guard used by `--json`
- `uvt_core.jobs`: `JobSpec` / `JobResult`, the contract for agents and remote execution
- `uvt_core.runner`: the shared CLI behaviour (`--json`, `--job`, `--print-job`, exit codes)

You normally get it as a dependency of a tool package. It also ships `uvt-info`,
which reports the detected device and the effective config:

```bash
uv tool install "git+https://github.com/LorenzoCiarpa/agent-tools#subdirectory=packages/core"
uvt-info --json
```
