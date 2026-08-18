# PostHog Analytics for Open Duck Mini Runtime (v3)

> **Partly superseded (2026-08-18).** This plan still describes what the runtime
> sends and why, and all of that is accurate. Two things it decided have since
> changed, both in
> [`identity-and-ownership.md`](identity-and-ownership.md), which points at the
> cross-repo design of record:
>
> 1. **The project.** This plan created a separate "Tnkr Robots" project to keep
>    robot events out of Tnkr Prod. The runtime now reports to **Tnkr Prod**
>    instead. A merge in PostHog is scoped to one project, so a robot's history
>    and the account that owns it have to be in the same one or they can never be
>    joined. Mentions of "Tnkr Robots" below, including the dashboard plan, read
>    as Tnkr Prod.
> 2. **Anonymity is now conditional.** This plan assumed a `device_id` that is
>    anonymous forever. It still is on the robot, but a signed-in owner
>    connecting the robot in Tnkr Studio links it to their Tnkr account, which
>    makes the robot's earlier events attributable. Nothing on the robot does
>    this and nothing on the robot can; see the README's telemetry section for
>    what customers are told.
>
> Everything else here stands.

## Context

We want visibility into how people's OpenDucks behave in the wild: what hardware they run (Pi model, RAM, OS, and especially which servo USB adapter chip — CH343 vs FTDI on v3), whether each install step and each API-driven setup step (motor check, calibration, IMU calibration, walk start) succeeds, and the *exact cause* when one fails. Telemetry must be respectful: enabled by default with a clear notice and an easy opt-out, anonymous device ID, and **never** the 10–50Hz joint stream (only a `cloud_streaming` boolean).

**Decisions made with user:**
- New PostHog project **"Tnkr Robots"** (org Tnkr, US cloud) created via PostHog MCP — keeps robot events out of Tnkr Prod web analytics.
- Embed the write-only `phc_` key in the repo (standard practice for ingestion-only keys).
- Enabled by default with printed notice + kill switches (`TNKR_TELEMETRY=0` env var, `~/.tnkr-telemetry.json` flag).
- Build on `v3`: reset `posthog-analytics` branch onto `v3`, PR targets `v3`.

## Identity & abuse model

- `distinct_id` = random `uuid4` generated at install/first-run, stored in `~/.tnkr-telemetry.json` (NOT `duck_config.json` — `POST /api/config` rewrites that file wholesale from `DuckConfigModel.model_dump()` and would drop extra keys). Survives `setup.sh --clean` and reinstalls.
- No PII: no hostnames, usernames, IPs (SDK `disable_geoip=True`; curl payloads set `"$geoip_disable": true`), no sessionToken/supabase creds, no joint data.
- Abuse: the key is write-only (can't read data). Every event carries `source: "openduck-runtime"` + `runtime_version` so spam can be filtered/dropped with PostHog ingestion filters; rogue device_ids can be blocked the same way. Document this in the PR.

## Files

### 0. Branch mechanics (first)
```
git checkout posthog-analytics && git reset --hard v3
```
Push with `--force-with-lease` if `origin/posthog-analytics` exists. PR base: `v3`.

### 1. PostHog MCP (no code)
Create project "Tnkr Robots"; grab the `phc_` project API key. Host `https://us.i.posthog.com`.

### 2. NEW `mini_bdx_runtime/mini_bdx_runtime/telemetry.py`
Fail-silent singleton wrapper around the `posthog` SDK (mirror `cloud_publisher.py`'s "never stall the robot" ethos — every public fn body in `try/except Exception: pass`, `posthog` import inside the try):

- Constants: `POSTHOG_API_KEY`, `POSTHOG_HOST`, `TELEMETRY_FILE = Path.home()/".tnkr-telemetry.json"`, `SOURCE = "openduck-runtime"`. Key/host are necessarily duplicated in `setup.sh` (it runs pre-clone via curl) — add cross-reference comments in BOTH files ("must match scripts/setup.sh" / "must match mini_bdx_runtime/telemetry.py"). Same contract treatment for the bash/python-duplicated property names (`pi_model`, `arch`, `ram_mb`, `os_release`): the event schema table below is the contract.
- `is_enabled()` — `TNKR_TELEMETRY` env (0/1 hard override) > file `enabled` > default True.
- `device_id()` — read/create file `{"device_id": uuid4, "enabled": true, "notice_version": 1, "created_at": iso}`; cached. On lazy creation (upgrade-path robots that never ran the new setup.sh), print ONE loud journalctl-visible line (decision 3A): `[telemetry] Anonymous usage telemetry enabled (device a1b2…). Disable: TNKR_TELEMETRY=0 or ~/.tnkr-telemetry.json` — the upgrade path gets the same notice-and-default-on treatment as setup.sh.
- `capture(event, properties=None)` — lazy `Posthog(key, host=..., disable_geoip=True)`; keyword-args call style (survives posthog-python 3.x–6.x signature changes); merges device props ∪ sticky props ∪ caller props ∪ `{"source": SOURCE}`; attaches `$set: device_properties()` on first event per process.
- `set_sticky(**props)` — merged into all later events + next `$set` (used for `servo_adapter_chip`).
- `flush(timeout=3)` / `shutdown()` — `atexit.register(shutdown)` on client creation.
- `device_properties()` (cached): `pi_model` (`/proc/device-tree/model`), `arch`, `os_release`, `python_version`, `ram_mb` (`/proc/meminfo`), `runtime_version` (`importlib.metadata.version("mini-bdx-runtime")`).

### 3. `setup.cfg`
Add `posthog>=3.0` to `install_requires`; bump `version` 0.0.1 → 0.1.0 so `runtime_version` means something.

### 4. `mini_bdx_runtime/mini_bdx_runtime/rustypot_position_hwi.py` — expose the chip
`find_servo_adapter() -> (device, chip)` holding the existing `SERVO_ADAPTER_VIDS` logic; keep `find_servo_port()` as a thin back-compat wrapper. In `HWI.__init__`: `self.servo_adapter_chip = None`; when `usb_port is None`, `usb_port, self.servo_adapter_chip = find_servo_adapter()`.

### 5. `scripts/tnkr_server.py` — middleware + contextvar enrichment + exception handlers
Why this shape: Starlette converts `HTTPException` to a response *before* outer `BaseHTTPMiddleware` sees it, so error detail is stashed by exception handlers into a request-scoped contextvar dict that the middleware reads.

- `_request_props: ContextVar[dict|None]`; helper `add_telemetry_props(**props)` that **mutates** the dict (mutation is visible across the threadpool context copy; never `.set()` from endpoints). Docstring must state WHY (run_in_threadpool copies the contextvars Context but shares the dict object), and a unit test must enforce the contract.
- `TelemetryMiddleware(BaseHTTPMiddleware)`: skips OPTIONS, non-`/api/` paths, and `TELEMETRY_EXCLUDED_PATHS = {"/api/commands", "/api/health", "/api/imu/calibrate/status"}` (50Hz + polling). Otherwise times the call and captures `api_request_completed` (status <400) or `api_request_failed` with `endpoint, method, status_code, duration_ms` + enrichments. Add order: TelemetryMiddleware first (innermost), then PrivateNetworkMiddleware, then CORSMiddleware.
- Exception handlers: `HTTPException` → `add_telemetry_props(error_type="HTTPException", error_message=str(exc.detail)[:500])` then delegate to fastapi's default handler; bare `Exception` → capture `type(exc).__name__` + `str(exc)[:500]` into telemetry props and log traceback, but return a **generic** `JSONResponse(500, {"detail": "Internal Server Error"})` — never leak `str(exc)` to HTTP clients (eng-review decision 2A; preserves current API behavior).
- Inline ASCII diagram comment in `tnkr_server.py` showing the request → TelemetryMiddleware → endpoint/exception-handler → contextvar-enrichment → capture flow (this pipeline is the non-obvious part).
- Enrichments (one-liners in existing handlers):
  - `check_motors`: `all_responsive`, `unresponsive_joints=[names]`, `responsive_count`
  - `calibration_save`: `joints_calibrated=len(calibration_offsets)`
  - `walk_start`: `cloud_streaming=bool(token and url and key)`, `has_session`, `already_running` — **never the token value**. No `flush()` call (decision 5A: the server outlives the subprocess; the SDK background thread delivers, and a 3s flush would stall the walk button on offline robots)
  - `walk_stop`: `was_running`
  - `update_config`: `expression_features_enabled` (keys only)
- Chip: in `get_hwi()` after constructing HWI → `telemetry.set_sticky(servo_adapter_chip=hwi_instance.servo_adapter_chip)`.
- `server_started` event in `__main__` before `uvicorn.run` (device specs + `$set` ride along).
- IMU outcome exactly-once, inside `_imu_calibrate_worker`: `imu_calibration_completed` (success path, `duration_s`), `imu_calibration_failed` (except path, `error_type/error_message`), `imu_calibration_stopped` (user stop, last `calibration_status`).
- Walk lifecycle (decision 1A — per-launch session object, no shared global flag): replace the bare `walk_process` global with a small `WalkSession` (dict or dataclass) holding `{proc, started_at, cloud_streaming, stop_requested: False}`. `stop_walk_process()` sets `session.stop_requested = True` on THAT session before terminate. `walk_start` spawns a daemon monitor thread closing over its own session: `proc.wait()` → capture `walk_ended {duration_s, exit_code, crashed, stop_requested, cloud_streaming}` where `crashed = rc not in (0, -SIGTERM, -SIGKILL) and not session.stop_requested`. No `flush()` in the monitor (decision 5A — atexit shutdown covers process exit). This kills the stop-A/start-B race that would mislabel clean stops as crashes. No changes to `v2_rl_walk_mujoco.py` / `cloud_publisher.py` — telemetry never touches the 50Hz loop.

### 6. `scripts/setup.sh` — consent + per-step events (curl; venv doesn't exist for steps 1–7)
Verified structure on v3: `run_step N id title fn` at lines ~667–682, `mark_done`, `cleanup()` EXIT trap at ~195, `STATE_DIR=~/.tnkr-setup`, `LOG_FILE`, `TTY_OUT` pattern, `--clean` at ~599.

- `telemetry_init()` called in `main()` after `mkdir -p $STATE_DIR`: if `~/.tnkr-telemetry.json` exists, read `device_id`/`enabled` (grep/sed, no jq) and print one-line reminder; else generate `DEVICE_ID=$(cat /proc/sys/kernel/random/uuid)`, print notice block ("We collect: setup outcomes, API errors, hardware model. Never: motion data, names, precise location. Disable: TNKR_TELEMETRY=0 or ~/.tnkr-telemetry.json"), optional `read -r -t 15 ... < /dev/tty` opt-out when `TTY_OUT=/dev/tty`, honor `TNKR_TELEMETRY=0` env, write the file. Resume never re-prompts.
- `ph_capture event props_fragment`: guarded by `$TELEMETRY_ENABLED`, `curl -s --max-time 3 -X POST $POSTHOG_HOST/i/v0/e/` with `api_key`, `distinct_id`, `properties{source, $geoip_disable:true, setup_script:true, ...}`, `|| true`. Foreground (3s cap is fine).
- Instrumentation without touching `set -e` semantics: `run_step` sets `CURRENT_STEP_NUM/ID/TITLE` + `STEP_START` before invoking, captures `setup_step_completed {step_id, step_num, duration_s}` after `mark_done`, clears `CURRENT_STEP_ID`. `cleanup()`: if exit_code≠0 and `CURRENT_STEP_ID` set → `setup_step_failed {step_id, step_num, exit_code, duration_s, error_tail}` where `error_tail` = last 20 lines of `$LOG_FILE`, ANSI-stripped, ≤2000 chars, JSON-escaped via `python3 -c` (guarded by `command -v python3`; omit tail if absent).
- `setup_started` after init (`resumed`, `steps_already_done`, `clean_install`, hw props, `$set`); `setup_completed` before `print_success` (`total_duration_s`).
- Step 8 pip list: add `"posthog>=3.0"`. `--clean` must NOT delete `~/.tnkr-telemetry.json`.
- Lint: `bash -n` + `shellcheck scripts/setup.sh`.

### 7. NEW `tests/` — pytest suite (eng-review decision 4A: 26 gap paths, repo's first Python tests)
Dev-only deps (`pytest`, `httpx` for TestClient) via a `[options.extras_require] dev` section in setup.cfg — NOT in `install_requires` (nothing extra on the Pi). A `tests/conftest.py` provides: a stub `rustypot` module on `sys.path` (raises on connect), a fake posthog client fixture capturing calls in a list (monkeypatched into `telemetry`), tmp `HOME` for the telemetry file.

- `tests/test_telemetry.py`: env-var overrides (0 wins over file-true, 1 wins over file-false), file enabled:false, file-missing default-true + lazy creation prints the notice line, device_id stable across calls, unwritable HOME fail-silent, kwargs-only capture shape, posthog-import-missing no-op, `$set` on first event only, `set_sticky` merge.
- `tests/test_server_telemetry.py` (FastAPI TestClient): 2xx → `api_request_completed` with duration; HTTPException → `api_request_failed` carrying detail; unhandled exception → generic 500 body (no `str(exc)` leak) but rich telemetry props; excluded paths (`/api/commands`, `/api/health`, imu status) + OPTIONS produce zero events; motor-check `unresponsive_joints` enrichment via stub HWI; `walk_start` props contain `cloud_streaming` and NEVER the sessionToken string (assert token substring absent from all captured payloads); contextvar mutate-only contract.
- `tests/test_walk_monitor.py`: child `sleep` + stop → `walk_ended {crashed: false, stop_requested: true}`; child exits 1 → `{crashed: true, exit_code: 1}`; stop-A-then-start-B → A's event unaffected (regression test for the 1A race).
- IMU worker: monkeypatch the adafruit import to raise → `imu_calibration_failed` captured.
- setup.sh layer: `bash -n` + shellcheck; forced step failure via the existing `Dockerfile.test` harness → `setup_step_failed` with sane `error_tail` (manual/E2E, not pytest).

### 8. NEW `.github/workflows/test.yml` — minimal CI (decision: bundle now)
~25 lines: on PRs/pushes to `v3` — checkout, setup-python 3.11, `pip install -e ./mini_bdx_runtime[dev] --no-deps` + direct dev deps, `pytest`, plus `shellcheck scripts/setup.sh`. No hardware needed (stubbed rustypot).

### 9. PostHog dashboard (via MCP, decision: build now)
After the project + key exist, create a "Robot Fleet" dashboard on Tnkr Robots: setup funnel (`setup_started` → `setup_step_completed` step 12 → `setup_completed`), `api_request_failed` trend broken down by `error_type`, device breakdown by `servo_adapter_chip` and `pi_model`, `walk_ended` crashed-rate.

### 10. Docs
README "Telemetry" section: what's collected / never collected / how to disable / where the file lives. Optional commented `# Environment=TNKR_TELEMETRY=0` in `tnkr-robot.service.template`.

## Event schema

distinct_id = device uuid. Base props everywhere: `source`, `runtime_version`, `pi_model`, `arch`, `os_release`, `python_version`, `ram_mb`, sticky `servo_adapter_chip` once known.

| Event | Source | Key properties |
|---|---|---|
| `setup_started` | setup.sh | `resumed`, `steps_already_done`, `clean_install`, hw props, `$set` |
| `setup_step_completed` | setup.sh | `step_id` (e.g. `08_pip_core`), `step_num`, `duration_s` |
| `setup_step_failed` | setup.sh EXIT trap | + `exit_code`, `error_tail` (≤2000 chars) |
| `setup_completed` | setup.sh | `total_duration_s` |
| `server_started` | server `__main__` | `server_port`, `$set` device props |
| `api_request_completed` | middleware | `endpoint`, `method`, `status_code`, `duration_ms`, enrichments |
| `api_request_failed` | middleware | + `error_type`, `error_message` (≤500 chars) |
| `imu_calibration_{completed,failed,stopped}` | IMU worker | `duration_s`, error fields / `calibration_status` |
| `walk_ended` | monitor thread | `duration_s`, `exit_code`, `crashed`, `stop_requested`, `cloud_streaming` |

Excluded entirely: `/api/commands` (50Hz), `/api/health`, `/api/imu/calibrate/status` (polling), OPTIONS. Never sent: joint data, sessionToken, supabase creds.

## Implementation order
1. Branch reset onto v3 → 2. MCP: create project + key → 3. `telemetry.py` + `setup.cfg` → 4. chip exposure in `rustypot_position_hwi.py` → 5. `tnkr_server.py` → 6. `setup.sh` → 7. tests + CI workflow → 8. README → 9. MCP dashboard → 10. verify → PR to `v3`.
(Parallelizable: the python lane (3–5) and the setup.sh lane (6) are independent once the key exists.)

## Verification (mac, no robot hardware)
0. `pytest` green locally (covers the 26 traced paths) and the new CI workflow passes on the PR.
1. Scratch venv: `python -c "from mini_bdx_runtime import telemetry; telemetry.capture('test_event', {'check': True}); telemetry.flush()"` → file created with uuid; event visible via PostHog MCP query on Tnkr Robots project.
2. Server with stub `rustypot.py` on PYTHONPATH (raises on connect): `POST /api/motors/check` → 503 + `api_request_failed` with `error_type=HTTPException`, message "Cannot connect to motor controller…". `POST /api/walk/start` with a script that exits → `walk_ended` with nonzero `exit_code`, `crashed=true`.
3. Hammer `/api/commands` + `/api/health` 50× → zero events.
4. `TNKR_TELEMETRY=0` and file `"enabled": false` → zero events.
5. `bash -n` + shellcheck on setup.sh; source `ph_capture`/`telemetry_init` in isolation → `setup_started` arrives; forced step failure → `setup_step_failed` with sane `error_tail`.
6. MCP: trend insight on `api_request_failed` by `error_type`; person profile shows `$set` hardware props.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 5 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

Eng review decisions folded into the plan: 1A per-launch walk session object (kills crash-mislabel race), 2A generic 500 to clients with rich telemetry capture, 3A lazy-enable notice on upgrade path, 4A full pytest suite (26 paths) + first CI workflow, 5A removed both flush() calls. Outside voice: skipped by user.

**UNRESOLVED:** 0
**VERDICT:** ENG CLEARED — ready to implement.
