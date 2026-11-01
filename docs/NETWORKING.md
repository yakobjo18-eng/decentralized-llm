# Windows networking requirements

Per the project decision: all three computers operate on the same
private Wi-Fi/LAN. This document covers what changes when a worker
runs on a different physical machine than the orchestrator (Phase 2
onward) -- Phase 1's own smoke test runs everything over loopback
(127.0.0.1), which needs none of this.

## Required open ports

| Component | Default port | Direction |
|---|---|---|
| Orchestrator REST API | 8080/tcp | Inbound, from the frontend/dashboard machine and operators |
| Orchestrator gRPC (WorkerRegistry) | 9090/tcp | Inbound, from every worker machine |
| Worker gRPC (Health) | 50100/tcp (configurable per worker) | Inbound, from the orchestrator |
| Worker StageExecution (Phase 6, off by default, **unauthenticated**) | 50200/tcp (`execution_listen_port`) | Inbound, from the coordinator only. Bind to the private LAN IP and allow `LocalSubnet` only; see [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md) |

All of these are configurable (never hardcoded) -- see
`.env.example`, `orchestrator/orchestrator.example.yaml`, and
`worker/worker.example.yaml`.

## Windows Defender Firewall

### If all three machines are on a network profile set to "Private"

Windows generally prompts to allow an app through the firewall the
first time it listens on a port, and allowing it once is sufficient.
If the prompt does not appear (e.g. running non-interactively) or you
want to pre-authorize it, add explicit rules (run as Administrator):

```powershell
New-NetFirewallRule -DisplayName "Decentralized LLM Orchestrator REST" `
    -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private

New-NetFirewallRule -DisplayName "Decentralized LLM Orchestrator gRPC" `
    -Direction Inbound -Protocol TCP -LocalPort 9090 -Action Allow -Profile Private

New-NetFirewallRule -DisplayName "Decentralized LLM Worker gRPC" `
    -Direction Inbound -Protocol TCP -LocalPort 50100 -Action Allow -Profile Private
```

Run the worker rule on every worker machine, adjusting `-LocalPort` to
match that worker's configured `listen_port` if it differs from the
default.

### If a machine's network profile is "Public" instead

The rules above still work, but change `-Profile Private` to
`-Profile Public` (or `Any`) -- understanding that this opens the port
to anything on that network, which matters more on a public/guest
network than on a private home/office LAN. Confirm with:

```powershell
Get-NetConnectionProfile
```

If any of the three machines show `NetworkCategory : Public` and you
cannot change it to Private (some networks don't allow this), prefer
scoping the firewall rule to the other two machines' specific IPs via
`-RemoteAddress`, rather than opening the port to the whole network.

## Checking a port is actually reachable from another machine

From Computer 1, after starting the orchestrator:

```powershell
Test-NetConnection -ComputerName <computer-1-ip> -Port 9090
```

Run this from Computer 2 or Computer 3 once they're on the network, to
confirm firewall/network config before debugging the application
layer.

## No hardcoded IPs

Nothing in this codebase hardcodes a specific machine's IP address.
`WORKER_ORCHESTRATOR_ADDRESS` (or `orchestrator_address` in
`worker.yaml`) must be set to Computer 1's actual LAN IP address when
running a worker on Computer 2 or Computer 3 -- find it via:

```powershell
ipconfig | Select-String "IPv4"
```

## Heartbeat timeout and real network conditions (Phase 2)

`heartbeat_timeout_seconds` in `orchestrator.yaml` (default 15s, three
missed 5-second heartbeats) is what separates "worker is offline" from
"worker had one slow network round trip." This has only been tuned and
tested over loopback on Computer 1, where round-trip time is
effectively zero. A real Wi-Fi LAN has:

- Actual latency and occasional packet loss, which Phase 1/2 have not
  measured (that's Phase 7 -- "Latency-Aware Orchestration").
  Computer 1's own Wi-Fi vs. Ethernet, and Computer 2/3's unknown
  network hardware, could all behave differently.
- A longer worst case than loopback for a single missed heartbeat to
  actually arrive late rather than fail outright.

Until real cross-machine latency is measured, 15 seconds is a
reasonable default but not a validated one for the actual LAN this
project will run on. If workers on Computer 2/3 flap online/offline
over real Wi-Fi, raise `heartbeat_timeout_seconds` (or lower
`heartbeat_interval_seconds` in each worker's config, sending
heartbeats more often) before assuming there's a bug in the timeout
logic itself -- the logic is unit-tested (see
orchestrator/internal/registry/registry_timeout_test.go) and proven
correct over loopback via a real three-process test
(worker/tests/test_three_worker_integration.py); what's unverified is
only the *default timeout value* against real network conditions,
not whether the detection mechanism works.

## What Phase 2 has NOT verified for multi-machine deployment

To be explicit about the gap between "the code supports configurable
addresses" and "this has been tested across physical machines":

- No worker has registered from an address other than 127.0.0.1 in
  any test so far. The address-handling code path (`_own_advertised_address`
  in worker/worker/main.py, which substitutes the real hostname for
  `0.0.0.0`) has not been exercised against a real second machine.
- Firewall rules above are documented, not yet applied or tested on
  Computer 2/3 (their existence/configuration is unknown -- see
  docs/ARCHITECTURE.md's open hardware-survey item).
- This is tracked honestly in docs/DEVELOPMENT_STATUS.md as "local
  multi-process tested, physical multi-machine untested" -- the two
  are not the same claim.

## Phase 3: hardware metadata over the network

### Address advertisement (new: `advertise_host`)

The orchestrator needs an address it can dial back to reach each
worker. A worker advertises, in order of precedence:

1. `advertise_host` (`worker.yaml`) / `WORKER_ADVERTISE_HOST`, if set;
2. otherwise `listen_host`, if it is a specific address;
3. otherwise, when listening on `0.0.0.0`, the machine's hostname.

Option 3 depends on Windows name resolution (NetBIOS/LLMNR/mDNS)
working between the three machines, which is often not the case on
home and lab networks. **For Computers 2 and 3, set `advertise_host`
to the machine's LAN IP** (`ipconfig` → IPv4 Address) and keep
`listen_host: "0.0.0.0"`. No address, localhost or otherwise, is
hard-coded in code. The `127.0.0.1` values in the example configs are
local-development placeholders. `test_advertised_address_never_assumes_localhost`
covers this.

### What travels, and how big it is

- `Register` carries the full hardware profile, a few KB of protobuf.
  It is sent at startup, every `hardware_refresh_interval_seconds`
  (default 300 s), and after a reconnection.
- `Heartbeat` carries live CPU/RAM/swap metrics (tens of bytes) every
  `heartbeat_interval_seconds`. GPU metrics are added every
  `gpu_metrics_interval_seconds`.

Neither is meaningful load on a LAN.

### Connection failures and reconnection

| Situation | Worker behaviour | Verified |
|---|---|---|
| Orchestrator not up yet at worker start | retries registration with backoff (Phase 2) | real processes, loopback |
| Heartbeats fail (network down) | keeps retrying; orchestrator marks it offline after `heartbeat_timeout_seconds` | real processes, loopback (Phase 2) |
| Network returns | first successful heartbeat triggers a **fresh full profile** + re-registration | unit-level, plus the restart test below |
| Orchestrator restarted (registry lost) | heartbeat reply has `reregister_required` → re-registers with a fresh profile | real processes, loopback (`test_worker_reregisters_with_fresh_profile_after_orchestrator_restart`) |
| Worker reconnects from a **different** address after going offline | accepted as a reconnection (Phase 2 rule), profile replaced | Go unit test (`TestReconnection_ReRegisterRefreshesHardware`) |
| Simulated worker vs. an orchestrator that doesn't allow them | registration rejected, worker exits with a clear log message | real processes, loopback |

### What Phase 3 has NOT verified across machines

Everything above ran over loopback on Computer 1. **No physical
multi-computer test has been run.** In particular, these are unverified:
- `advertise_host` against a real second machine;
- whether DXGI/PDH GPU detection behaves the same on Computer 2/3's
  (unknown) GPUs and drivers;
- reconnection timing over real Wi-Fi.

Local multi-process results are not evidence of physical multi-machine
behaviour.
