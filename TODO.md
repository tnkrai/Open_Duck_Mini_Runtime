- [] Better handle xbox controller 
  - It's a little bit of a mess right now, how we handle directions and buttons etc
- [] Make the offsets flashing work. This will be in the motor configuration script
- [] **SECURITY: the robot API is reachable from any webpage the owner visits**
  - **What:** All 28 endpoints in `scripts/tnkr_server.py` have zero auth (`Depends(...)`
    count is 0). `CORSMiddleware` runs with `allow_origins=["*"]` and
    `allow_credentials=True`, and `PrivateNetworkMiddleware` responds
    `Access-Control-Allow-Private-Network: true` to Chrome's PNA preflight. That header is
    a *grant*, not a guard: it opts the duck in to being called from HTTPS pages.
  - **Why it matters:** the exposure is not limited to someone on your wifi. Any site a
    customer visits while their duck is powered on can drive the robot, disable torque,
    rewrite `duck_config.json`, or start a walk. DNS rebinding makes this a known,
    well-documented attack class against local IoT devices.
  - **Why it is not fixed in the telemetry work:** the PNA grant exists because the
    dashboard connects to robots from the browser (`robot_pi_post_completed` in
    `tnkr-dashboard/app/analytics/events.ts`). Tightening CORS naively breaks that flow.
    This needs its own design: probably an origin allowlist for tnkr.ai plus a
    provisioned token for anything state-changing, decided together with the dashboard.
  - **What the telemetry work did instead:** refused to make it worse.
    `GET /api/telemetry/identity` is read-only and returns only a random UUID and a
    boolean; ownership is decided in tnkr-core behind a verified Supabase token, so
    nothing reachable on the LAN can claim a robot. See
    `tnkr-studio/docs/plans/telemetry/_architecture.md`.
  - **Depends on:** coordinated change with tnkr-dashboard's robot-connect path.
