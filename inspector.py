"""Enhanced environment inspector functions for Gradio Dataframe components."""

import pandas as pd

from sentinelops_arena.environment import SentinelOpsArena


def get_all_customers(env: SentinelOpsArena) -> pd.DataFrame:
    """Return all CRM customers as a DataFrame."""
    rows = []
    for cid, rec in env.crm.customers.items():
        tier = rec.get("tier", "")
        rows.append({
            "customer_id": rec.get("customer_id", cid),
            "name": rec.get("name", ""),
            "tier": tier.value if hasattr(tier, "value") else str(tier),
            "region": rec.get("region", ""),
            "lifetime_value": rec.get("lifetime_value", 0.0),
        })
    if not rows:
        return pd.DataFrame(columns=["customer_id", "name", "tier", "region", "lifetime_value"])
    return pd.DataFrame(rows)


def get_all_invoices(env: SentinelOpsArena) -> pd.DataFrame:
    """Return all billing invoices as a DataFrame."""
    rows = []
    for iid, rec in env.billing.invoices.items():
        status = rec.get("status", "")
        rows.append({
            "invoice_id": rec.get("invoice_id", iid),
            "customer_id": rec.get("customer_id", ""),
            "amount": rec.get("amount", 0.0),
            "status": status.value if hasattr(status, "value") else str(status),
        })
    if not rows:
        return pd.DataFrame(columns=["invoice_id", "customer_id", "amount", "status"])
    return pd.DataFrame(rows)


def get_all_tickets(env: SentinelOpsArena) -> pd.DataFrame:
    """Return all ticketing system tickets as a DataFrame."""
    rows = []
    for tid, rec in env.ticketing.tickets.items():
        priority = rec.get("priority", "")
        status = rec.get("status", "")
        rows.append({
            "ticket_id": rec.get("ticket_id", tid),
            "customer_id": rec.get("customer_id", ""),
            "subject": rec.get("subject", ""),
            "priority": priority.value if hasattr(priority, "value") else str(priority),
            "status": status.value if hasattr(status, "value") else str(status),
            "sla_deadline_tick": rec.get("sla_deadline_tick", 0),
        })
    if not rows:
        return pd.DataFrame(columns=["ticket_id", "customer_id", "subject", "priority", "status", "sla_deadline_tick"])
    return pd.DataFrame(rows)


def get_task_queue(env: SentinelOpsArena) -> pd.DataFrame:
    """Return the task queue as a DataFrame with truncated messages."""
    rows = []
    for task in env.tasks:
        d = task.model_dump() if hasattr(task, "model_dump") else task
        msg = str(d.get("message", ""))
        task_type = d.get("task_type", "")
        rows.append({
            "task_id": d.get("task_id", ""),
            "customer_id": d.get("customer_id", ""),
            "task_type": task_type.value if hasattr(task_type, "value") else str(task_type),
            "message": msg[:60] + ("..." if len(msg) > 60 else ""),
            "arrival_tick": d.get("arrival_tick", 0),
        })
    if not rows:
        return pd.DataFrame(columns=["task_id", "customer_id", "task_type", "message", "arrival_tick"])
    return pd.DataFrame(rows)


def get_env_config_html(env: SentinelOpsArena) -> str:
    """Return styled HTML showing environment configuration."""
    refund = env.billing.refund_policy.model_dump()
    sla = env.ticketing.sla_rules.model_dump()

    css = (
        "font-family: 'IBM Plex Mono', monospace;"
        "background: var(--sentinel-surface);"
        "color: var(--sentinel-text);"
        "padding: 20px;"
        "border-radius: 8px;"
        "border: 1px solid var(--sentinel-border);"
    )
    heading_css = (
        "color: var(--sentinel-green);"
        "font-size: 14px;"
        "font-weight: bold;"
        "margin: 16px 0 8px 0;"
        "text-transform: uppercase;"
        "letter-spacing: 1.5px;"
    )
    table_css = (
        "width: 100%;"
        "border-collapse: collapse;"
        "margin-bottom: 12px;"
    )
    th_css = (
        "text-align: left;"
        "padding: 6px 12px;"
        "border-bottom: 1px solid var(--sentinel-border);"
        "color: var(--sentinel-blue);"
        "font-size: 12px;"
    )
    td_css = (
        "padding: 6px 12px;"
        "border-bottom: 1px solid rgba(201, 209, 217, 0.1);"
        "font-size: 13px;"
    )
    val_css = (
        "color: var(--sentinel-green);"
        "font-weight: bold;"
    )

    def row(key, value):
        return (
            f"<tr>"
            f"<td style='{td_css}'>{key}</td>"
            f"<td style='{td_css} {val_css}'>{value}</td>"
            f"</tr>"
        )

    html = f"<div style='{css}'>"

    # Environment params
    html += f"<div style='{heading_css}'>Environment Parameters</div>"
    html += f"<table style='{table_css}'>"
    html += f"<tr><th style='{th_css}'>Parameter</th><th style='{th_css}'>Value</th></tr>"
    html += row("MAX_TICKS", env.MAX_TICKS)
    html += row("NUM_CUSTOMERS", env.NUM_CUSTOMERS)
    html += row("NUM_INVOICES", env.NUM_INVOICES)
    html += row("NUM_TICKETS", env.NUM_TICKETS)
    html += row("NUM_TASKS", env.NUM_TASKS)
    html += "</table>"

    # Refund policy
    html += f"<div style='{heading_css}'>Refund Policy</div>"
    html += f"<table style='{table_css}'>"
    html += f"<tr><th style='{th_css}'>Rule</th><th style='{th_css}'>Value</th></tr>"
    html += row("Window (ticks)", refund["window_ticks"])
    html += row("Requires Approval", refund["requires_approval"])
    html += row("Max Amount", f"${refund['max_amount']:,.2f}")
    html += "</table>"

    # SLA rules
    html += f"<div style='{heading_css}'>SLA Rules (ticks to resolve)</div>"
    html += f"<table style='{table_css}'>"
    html += f"<tr><th style='{th_css}'>Priority</th><th style='{th_css}'>Deadline (ticks)</th></tr>"
    html += row("High", sla["high"])
    html += row("Medium", sla["medium"])
    html += row("Low", sla["low"])
    html += "</table>"

    html += "</div>"
    return html
