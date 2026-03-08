# Phase 3: MCP Tools + OpenEnv HTTP Server + MCP-X Gateway

**Time:** 1.5 hours (Hours 4-5.5)
**Priority:** HIGH -- unlocks demo and satisfies Pipeline judging criterion (10%)
**Depends on:** Phase 2 (working environment)

---

## Files to Create

| File | Purpose | Est. Time |
|------|---------|-----------|
| `sentinelops_arena/mcp_tools.py` | FastMCP tool definitions wrapping env operations | 30 min |
| `sentinelops_arena/server.py` | `create_app()` HTTP server entry point | 15 min |
| `mcp-x/config.toml` | MCP-X per-agent access control config | 10 min |
| `mcp-x/mcp_x.py` | Copy from envbeats, no modifications needed | 5 min |
| `run_server.py` | Script to start both env server + MCP-X | 10 min |
| `tests/test_mcp.py` | MCP tool integration tests | 20 min |

---

## Step-by-Step Build Instructions

### Step 1: server.py -- OpenEnv HTTP Server (15 min)

Follow the hackathon_env template exactly.

```python
# sentinelops_arena/server.py
"""
FastAPI application for SentinelOps Arena.

Endpoints:
    POST /reset  -- Reset environment
    POST /step   -- Execute an action
    GET  /state  -- Get current state
    GET  /schema -- Get action/observation schemas
    WS   /ws     -- WebSocket for persistent sessions

Usage:
    uvicorn sentinelops_arena.server:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app
from .models import SentinelAction, SentinelObservation
from .environment import SentinelOpsArena

app = create_app(
    SentinelOpsArena,
    SentinelAction,
    SentinelObservation,
    env_name="sentinelops_arena",
    max_concurrent_envs=5,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
```

### Step 2: mcp_tools.py -- FastMCP Tool Definitions (30 min)

Expose enterprise system APIs as individual MCP tools. This is what LLM agents actually call.

```python
# sentinelops_arena/mcp_tools.py
"""
MCP tool definitions for SentinelOps Arena.

Exposes enterprise system APIs as MCP tools via FastMCP.
Tools are grouped by agent role (attacker/worker/oversight).
"""
import json
from fastmcp import FastMCP

from .environment import SentinelOpsArena
from .models import (
    SentinelAction, AgentRole, AttackType, TargetSystem,
    TicketPriority,
)

mcp = FastMCP("sentinelops", host="0.0.0.0", port=9500, stateless_http=True)

# Global environment instance (shared across MCP calls)
env = SentinelOpsArena()


# ============ Environment Control Tools ============

@mcp.tool()
def reset(seed: int = 42) -> str:
    """Reset the SentinelOps environment for a new episode."""
    obs = env.reset(seed=seed)
    return obs.model_dump_json()


@mcp.tool()
def step(action_json: str) -> str:
    """Take a step in the SentinelOps environment with a full action."""
    action = SentinelAction.model_validate_json(action_json)
    obs = env.step(action)
    return obs.model_dump_json()


@mcp.tool()
def get_state() -> str:
    """Get the current environment state (tick, scores, active attacks)."""
    return env.state.model_dump_json()


# ============ Worker Tools (Enterprise System APIs) ============

@mcp.tool()
def lookup_customer(customer_id: str) -> str:
    """Look up a customer record in the CRM system."""
    result = env.crm.lookup_customer(customer_id)
    return json.dumps(result)


@mcp.tool()
def update_tier(customer_id: str, new_tier: str) -> str:
    """Update a customer's tier level (gold/silver/bronze)."""
    result = env.crm.update_tier(customer_id, new_tier)
    return json.dumps(result)


@mcp.tool()
def add_note(customer_id: str, note: str) -> str:
    """Add a note to a customer's record."""
    result = env.crm.add_note(customer_id, note)
    return json.dumps(result)


@mcp.tool()
def get_history(customer_id: str) -> str:
    """Get interaction history for a customer."""
    result = env.crm.get_history(customer_id)
    return json.dumps(result)


@mcp.tool()
def check_balance(customer_id: str) -> str:
    """Check the billing balance for a customer."""
    result = env.billing.check_balance(customer_id)
    return json.dumps(result)


@mcp.tool()
def issue_refund(invoice_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an invoice. Must comply with current refund policy."""
    result = env.billing.issue_refund(invoice_id, amount, reason)
    return json.dumps(result)


@mcp.tool()
def apply_credit(customer_id: str, amount: float) -> str:
    """Apply a credit to a customer's account."""
    result = env.billing.apply_credit(customer_id, amount)
    return json.dumps(result)


@mcp.tool()
def generate_invoice(customer_id: str, items: str, amount: float) -> str:
    """Generate a new invoice for a customer. Items should be comma-separated."""
    item_list = [i.strip() for i in items.split(",")]
    result = env.billing.generate_invoice(customer_id, item_list, amount)
    return json.dumps(result)


@mcp.tool()
def create_ticket(customer_id: str, subject: str, priority: str = "medium") -> str:
    """Create a new support ticket."""
    result = env.ticketing.create_ticket(customer_id, subject, TicketPriority(priority))
    return json.dumps(result)


@mcp.tool()
def assign_ticket(ticket_id: str, agent_name: str) -> str:
    """Assign a ticket to an agent."""
    result = env.ticketing.assign_ticket(ticket_id, agent_name)
    return json.dumps(result)


@mcp.tool()
def escalate_ticket(ticket_id: str, reason: str) -> str:
    """Escalate a ticket to a senior agent."""
    result = env.ticketing.escalate(ticket_id, reason)
    return json.dumps(result)


@mcp.tool()
def resolve_ticket(ticket_id: str, resolution: str) -> str:
    """Resolve a ticket with the given resolution."""
    result = env.ticketing.resolve(ticket_id, resolution)
    return json.dumps(result)


@mcp.tool()
def check_sla(ticket_id: str) -> str:
    """Check SLA status for a ticket (ticks remaining before breach)."""
    result = env.ticketing.check_sla(ticket_id)
    return json.dumps(result)


@mcp.tool()
def get_schema(system: str) -> str:
    """Get the current field schema for a system (crm/billing/ticketing).
    Critical after schema drift attacks -- fields may have been renamed."""
    sys_obj = env._get_system(system)
    if sys_obj is None:
        return json.dumps({"error": f"Unknown system: {system}"})
    return json.dumps(sys_obj.get_schema())


@mcp.tool()
def get_current_policy(policy_type: str = "refund") -> str:
    """Get the current policy (refund or sla).
    Critical after policy drift attacks -- rules may have changed."""
    if policy_type == "refund":
        return json.dumps(env.billing.get_current_policy())
    elif policy_type == "sla":
        return json.dumps(env.ticketing.get_sla_rules())
    return json.dumps({"error": f"Unknown policy type: {policy_type}"})


# ============ Attacker Tools ============

@mcp.tool()
def launch_attack(attack_type: str, target_system: str, parameters_json: str = "{}") -> str:
    """Launch an attack on an enterprise system.
    Types: schema_drift, policy_drift, social_engineering, rate_limit.
    Costs 0.3 reward points per attack."""
    import json as _json
    params = _json.loads(parameters_json)
    params["attack_type"] = attack_type
    params["target_system"] = target_system
    result = env.attack_manager.launch_attack(
        AttackType(attack_type), TargetSystem(target_system), params, env.tick
    )
    return json.dumps(result)


@mcp.tool()
def pass_turn() -> str:
    """Pass the attacker's turn without launching an attack."""
    return json.dumps({"status": "passed"})


@mcp.tool()
def get_attack_budget() -> str:
    """Get the remaining attack budget for this episode."""
    budget = env.attack_manager.attack_budget if env.attack_manager else 10.0
    return json.dumps({"budget": budget})


# ============ Oversight Tools ============

@mcp.tool()
def flag_action(flagged: bool, severity: int = 3,
                violation_type: str = "policy_violation",
                explanation: str = "") -> str:
    """Flag or approve a worker action. Used by the oversight agent."""
    return json.dumps({
        "flagged": flagged,
        "severity": severity,
        "violation_type": violation_type,
        "explanation": explanation,
    })


@mcp.tool()
def get_trajectory(num_recent: int = 5) -> str:
    """Get recent action trajectory for oversight analysis."""
    trajectory = env.trajectory[-num_recent:] if env.trajectory else []
    return json.dumps(trajectory)
```

### Step 3: MCP-X Gateway Config (10 min)

```toml
# mcp-x/config.toml
[clients]
[clients.orchestrator]
auth_token = "orch-token-001"

[clients.attacker]
auth_token = "atk-token-001"

[clients.worker]
auth_token = "wrk-token-001"

[clients.oversight]
auth_token = "ovs-token-001"

[mcp_servers]
[mcp_servers.sentinelops]
url = "http://localhost:9500/mcp/"
from_client = "orchestrator"

[allow]
[allow.sentinelops]
attacker = ["launch_attack", "pass_turn", "get_attack_budget", "step", "reset", "get_state"]
worker = ["lookup_customer", "update_tier", "add_note", "get_history", "check_balance", "issue_refund", "apply_credit", "generate_invoice", "create_ticket", "assign_ticket", "escalate_ticket", "resolve_ticket", "check_sla", "get_schema", "get_current_policy", "step", "reset", "get_state"]
oversight = ["flag_action", "get_current_policy", "get_trajectory", "step", "reset", "get_state"]
```

### Step 4: Copy MCP-X (5 min)

Copy `envbeats/mcp-x/mcp_x.py` to `mcp-x/mcp_x.py`. No modifications needed -- it reads from `config.toml` in its working directory.

```bash
cp envbeats/mcp-x/mcp_x.py mcp-x/mcp_x.py
```

### Step 5: run_server.py -- Start Script (10 min)

```python
# run_server.py
"""Start both the OpenEnv HTTP server and MCP server."""
import subprocess
import sys
import time

def main():
    # Start OpenEnv HTTP server on port 8000
    env_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "sentinelops_arena.server:app",
        "--host", "0.0.0.0", "--port", "8000",
    ])

    # Start FastMCP server on port 9500
    mcp_proc = subprocess.Popen([
        sys.executable, "-c",
        "from sentinelops_arena.mcp_tools import mcp; mcp.run()"
    ])

    # Start MCP-X gateway on port 9000
    mcpx_proc = subprocess.Popen([
        sys.executable, "mcp-x/mcp_x.py", "--port", "9000"
    ])

    print("Servers started:")
    print("  OpenEnv HTTP: http://localhost:8000")
    print("  MCP (FastMCP): http://localhost:9500")
    print("  MCP-X Gateway: http://localhost:9000")

    try:
        env_proc.wait()
    except KeyboardInterrupt:
        env_proc.terminate()
        mcp_proc.terminate()
        mcpx_proc.terminate()

if __name__ == "__main__":
    main()
```

---

## VERIFY

### Test 1: OpenEnv HTTP Server
```bash
# Start server
uvicorn sentinelops_arena.server:app --port 8000 &

# Test reset
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
# Should return: {"observation": {...}, "reward": null, "done": false}

# Test step
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"agent": "attacker", "action_type": "pass"}}'
# Should return observation for worker

# Test state
curl http://localhost:8000/state
# Should return: {"episode_id": "...", "step_count": 1, "tick": 0, ...}

# Test schema
curl http://localhost:8000/schema
# Should return action/observation/state JSON schemas

kill %1
```

### Test 2: MCP Tools (FastMCP)
```python
# Start MCP server first, then:
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
import asyncio

async def test_mcp():
    async with streamablehttp_client(url="http://localhost:9500/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Available tools: {tool_names}")
            assert "reset" in tool_names
            assert "step" in tool_names
            assert "lookup_customer" in tool_names

            # Call reset
            result = await session.call_tool("reset", {"seed": 42})
            print(f"Reset result: {result.content[0].text[:100]}")

            # Call get_state
            result = await session.call_tool("get_state", {})
            print(f"State: {result.content[0].text[:100]}")

asyncio.run(test_mcp())
```

### Test 3: MCP-X Gateway (Per-Agent Isolation)
```python
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def test_mcpx():
    # Worker should see worker tools
    headers = {"Authorization": "Bearer wrk-token-001"}
    async with streamablehttp_client(url="http://localhost:9000/mcp/", headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"Worker tools: {names}")
            assert "lookup_customer" in names
            assert "launch_attack" not in names  # worker cannot attack

    # Attacker should see attacker tools
    headers = {"Authorization": "Bearer atk-token-001"}
    async with streamablehttp_client(url="http://localhost:9000/mcp/", headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"Attacker tools: {names}")
            assert "launch_attack" in names
            assert "lookup_customer" not in names  # attacker cannot use CRM

asyncio.run(test_mcpx())
```

---

## DEBUG: Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `Port 8000/9500/9000 already in use` | Previous server still running | `kill $(lsof -t -i:PORT)` |
| `ConnectionRefused on MCP-X` | MCP server not started before MCP-X | Start env server + MCP server before MCP-X |
| FastMCP `stateless_http=True` not working | Wrong FastMCP version | Check `pip show fastmcp` -- need recent version |
| MCP-X `ProxyClient` error | Dummy server hack missing | Ensure `_dummy_0` and `_dummy_1` servers in config |
| `streamablehttp_client` connection error | Async context manager issue | Must use `async with` pattern |
| `Bearer token` rejected | Token mismatch with config.toml | Verify token strings match exactly |
| MCP tool returns empty | Environment not reset | Call `reset` before other tools |
| `model_dump_json()` fails on complex types | Pydantic serialization issue | Use `json.dumps()` for dict results, `model_dump_json()` for Pydantic models |

---

## EXIT CRITERIA

- [ ] `uvicorn sentinelops_arena.server:app` starts without errors
- [ ] HTTP `/reset`, `/step`, `/state`, `/schema` all return valid JSON
- [ ] FastMCP server starts on port 9500
- [ ] All MCP tools are discoverable via `list_tools`
- [ ] `reset`, `step`, `get_state` MCP tools work
- [ ] `lookup_customer`, `issue_refund`, etc. return valid data
- [ ] MCP-X gateway starts on port 9000
- [ ] Worker token sees only worker tools
- [ ] Attacker token sees only attacker tools
- [ ] Oversight token sees only oversight tools
- [ ] Cross-role tool access denied (worker can't call launch_attack)

---

## ROLLBACK PLAN

If Phase 3 takes longer than 1.5 hours:
1. **Cut MCP-X gateway** -- submit with direct MCP only (no per-agent isolation). Add MCP-X in Phase 6 polish.
2. **Reduce MCP tools** -- only expose `reset`, `step`, `get_state` (no individual system tools). Agents call `step()` with full actions.
3. **Cut MCP entirely** -- use only HTTP server. Agents call REST endpoints directly.

Do NOT cut: `server.py` with `create_app()`. This is required for HF Spaces deployment.
