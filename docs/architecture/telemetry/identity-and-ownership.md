# Identity and ownership

**The design of record lives in another repo.** This is a pointer, so nobody
re-derives it from the runtime side and reaches a different answer:

- `tnkr-studio/docs/plans/telemetry/_architecture.md` — the whole cross-repo
  design: what each of the three repos emits, under which id, and where the
  robot's anonymous id becomes attached to the customer who owns it.
- `tnkr-studio/docs/architecture/backend.md` — the claim flow and its failure
  table, from Studio's side.

## What the runtime actually contributes

Two things, and neither of them decides anything.

**A random UUID.** `telemetry.py` mints it on first install and keeps it in
`~/.tnkr-telemetry.json`. It is the `distinct_id` on every event the robot and
`setup.sh` send. It is not derived from anything about the machine or the
person.

**A read-only endpoint.** `GET /api/telemetry/identity` returns that id and the
on/off flag. Nothing else, and there is no write counterpart. That is deliberate
and load-bearing: this server authenticates nobody and is reachable from any
webpage its owner visits, so an endpoint that recorded an owner could only ever
believe whoever asked first. `tests/test_identity_endpoint.py` pins both
properties — that the response carries no account data, and that no write
counterpart exists.

## Where ownership is actually decided

In tnkr-core, behind a verified Supabase token. Studio reads the id from the
robot, and posts it to `POST /studio/v1/robots/claim` with the customer's bearer
token; core takes the user id from the token it verified itself, never from the
request body. First claim wins there, which is what makes it safe for Studio to
call on every connect.

Studio is the only process in the world that simultaneously holds an
authenticated Tnkr session and a live connection to one specific physical robot.
That coincidence is the entire mechanism: the duck has no button and no screen,
so the available proof of possession is a locality test.

## Consequences for this repo

- **The project is Tnkr Prod**, not a separate robots project. A PostHog merge
  cannot cross projects, so a robot's history and its owner have to be in the
  same one. `setup.sh` and `telemetry.py` carry the same key and a test asserts
  they match.
- **Ducks installed before that key swap keep reporting to the old project**
  until `setup.sh` is re-run. There is no remote reconfiguration and there should
  not be. Expect the funnel to look thin for a while; that is a split fleet, not
  a product problem.
- **Reflashing the SD card mints a new id** and unlinks the robot from whoever
  owned it. `setup.sh --clean` deliberately does NOT, because that file holds the
  opt-out and a reinstall must not silently re-enable telemetry. This is why
  "reflash before you sell it" is written in the README rather than being an
  `--unlink` flag: the flag would be a second way to lose an opt-out.
- **Turning telemetry off here prevents the link entirely**, because the endpoint
  withholds the id.
