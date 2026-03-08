# Phase 3: MCP + OpenEnv HTTP Server

**Time:** 0.5 hours (Hours 6-6.5)
**Priority:** MEDIUM -- MCPEnvironment did most of the work in Phase 2
**Depends on:** Phase 2 (working environment with MCP tools)

**KEY CHANGE:** MCPEnvironment handles MCP tool routing automatically. Phase 3 is now just creating the HTTP server entry point and verifying everything works end-to-end. MCP-X gateway is CUT.

---

## Files to Create

| File | Purpose | Est. Time |
|------|---------|-----------|
| `sentinelops_arena/server.py` | `create_app()` HTTP server entry point | 10 min |
| Verify MCP tools via HTTP | End-to-end test | 10 min |
| Verify WebSocket + MCP | Integration test | 10 min |

---

## Step-by-Step Build Instructions

### Step 1: server.py -- OpenEnv HTTP Server (10 min)

This is trivial -- follow the hackathon_env template exactly.

```python
# sentinelops_arena/server.py
"""
HTTP server for SentinelOps Arena.

Endpoints:
    POST /reset  -- Reset environment
    POST /step   -- Execute an action (including ListToolsAction, CallToolAction)
    GET  /state  -- Get current state
    GET  /schema -- Get action/observation schemas
    WS   /ws     -- WebSocket for persistent sessions (supports /mcp)

The MCPEnvironment base class handles MCP tool routing automatically.
Agents can discover tools via ListToolsAction and call them via CallToolAction.

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

### Step 2: Verify HTTP + MCP Integration (10 min)

```bash
# Start server
uvicorn sentinelops_arena.server:app --port 8000 &

# Test reset
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Test step (regular action)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"agent": "attacker", "action_type": "pass"}}'

# Test step (MCP list_tools -- auto-routed by MCPEnvironment)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"type": "list_tools"}}'
# Should return available MCP tools

# Test step (MCP call_tool -- auto-routed by MCPEnvironment)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"type": "call_tool", "tool_name": "lookup_customer", "arguments": {"customer_id": "C000"}}}'
# Should return customer data

# Test state
curl http://localhost:8000/state

# Test schema
curl http://localhost:8000/schema

kill %1
```

### Step 3: Verify WebSocket MCP Path (10 min)

```python
# Quick WebSocket test
import asyncio
import json
import websockets

async def test_ws():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "data": {"seed": 42}}))
        resp = json.loads(await ws.recv())
        print(f"Reset: {resp['type']}")

        # MCP via WebSocket
        await ws.send(json.dumps({
            "type": "mcp",
            "data": {"method": "tools/list", "params": {}, "id": 1}
        }))
        resp = json.loads(await ws.recv())
        print(f"MCP tools via WS: {resp}")

asyncio.run(test_ws())
```

---

## What MCPEnvironment Gives Us For Free

| Feature | How |
|---------|-----|
| MCP tool discovery | `ListToolsAction` -> returns all tools with schemas |
| MCP tool invocation | `CallToolAction(tool_name, arguments)` -> calls FastMCP tool |
| Reserved name validation | Rejects tools named `reset`, `step`, `state`, `close` |
| Timeout handling | Configurable timeout on tool calls |
| Error categorization | `ToolError` with types: execution_error, invalid_args, tool_not_found, timeout |
| WebSocket MCP path | `/ws` endpoint supports `type: "mcp"` messages |
| Async support | `_run_async_safely()` handles both sync and async contexts |

## What We DON'T Need (CUT)

| Removed | Reason |
|---------|--------|
| `mcp_tools.py` | MCP tools defined inside `environment.py` via FastMCP |
| `mcp-x/` directory | MCP-X gateway CUT -- MCPEnvironment handles tool exposure |
| `config.toml` | No MCP-X = no per-agent access control config |
| `run_server.py` | Single server is enough |
| Per-agent JWT tokens | Nice-to-have, not needed for demo/judging |

---

## VERIFY

### Test 1: HTTP Server starts
```bash
uvicorn sentinelops_arena.server:app --port 8000
# Should start without errors
# Should show "Uvicorn running on http://0.0.0.0:8000"
```

### Test 2: All endpoints return valid JSON
```bash
# Reset -> Observation JSON
# Step -> Observation JSON
# State -> State JSON
# Schema -> Action/Observation/State schemas
```

### Test 3: MCP tools discoverable via HTTP
```bash
# POST /step with ListToolsAction -> list of tools
# Verify: lookup_customer, issue_refund, get_schema, launch_attack etc. all present
# Verify: no reserved names (reset, step, state, close)
```

### Test 4: MCP tools callable via HTTP
```bash
# POST /step with CallToolAction -> tool result
# Call lookup_customer("C000") -> customer data
# Call get_schema("crm") -> field list
# Call get_current_policy("refund") -> policy values
```

---

## DEBUG: Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `Port 8000 already in use` | Previous server running | `kill $(lsof -t -i:8000)` |
| `create_app()` fails with type error | Wrong argument types | Pass class (not instance), Action class, Observation class |
| MCP tools not showing up | Tools defined after `super().__init__()` | Define tools BEFORE calling `super().__init__(mcp)` |
| `ValueError: reserved names` | Tool named `reset` or `step` | Rename the tool |
| WebSocket MCP not working | Wrong message format | Use `{"type": "mcp", "data": {"method": "tools/list", ...}}` |
| `ListToolsAction` not recognized | `create_app` doesn't know about MCP types | May need to pass both `SentinelAction` and MCP action types to create_app |

---

## EXIT CRITERIA

- [ ] `uvicorn sentinelops_arena.server:app` starts without errors
- [ ] HTTP `/reset`, `/step`, `/state`, `/schema` return valid JSON
- [ ] `ListToolsAction` via `/step` returns all enterprise system tools
- [ ] `CallToolAction` via `/step` successfully calls tools
- [ ] WebSocket `/ws` endpoint accepts connections

---

## ROLLBACK PLAN

Phase 3 is already minimal. If it takes longer than 30 minutes:
1. **Skip WebSocket verification** -- HTTP-only is fine for demo
2. **Skip schema endpoint check** -- not needed for judging
3. **If `create_app()` fails entirely** -- serve the Gradio app directly without the OpenEnv HTTP layer. The environment still works via direct Python calls.

Do NOT cut: `server.py` with `create_app()`. This is required for HF Spaces deployment.
