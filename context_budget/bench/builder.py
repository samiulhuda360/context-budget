"""Building blocks for benchmark conversations: a message builder and realistic tool-output generators."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

FIRST = [
    "Sarah",
    "James",
    "Emma",
    "Tom",
    "Claire",
    "Olivia",
    "Jack",
    "Grace",
    "Liam",
    "Chloe",
    "Noah",
    "Ella",
    "Ben",
    "Lucy",
    "Sam",
    "Hannah",
    "Oliver",
    "Mia",
    "Ryan",
    "Zoe",
    "Kate",
    "Daniel",
    "Amy",
    "Luke",
]
LAST = [
    "Collins",
    "Walker",
    "Clarke",
    "Turner",
    "Hughes",
    "Bennett",
    "Foster",
    "Murray",
    "Reid",
    "Harper",
    "Wallace",
    "Brooks",
    "Marsh",
    "Doyle",
    "Fletcher",
    "Sutton",
    "Lowe",
    "Barker",
]
REGIONS = ["Auckland", "Wellington", "Canterbury", "Otago", "Waikato", "Nelson", "Hawke's Bay", "Northland"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def tool(tool_name: str, description: str, /, **params: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in params.items()}, "required": list(params)},
        },
    }


TOOLS = {
    "fetch_logs": tool("fetch_logs", "Fetch recent log lines for a service", service="service name", since="ISO time"),
    "get_deploys": tool("get_deploys", "List recent deployments", service="service name"),
    "run_command": tool("run_command", "Run a read-only shell command on a host", host="host", command="command"),
    "run_sql": tool("run_sql", "Run a read-only SQL query on the analytics replica", query="SQL"),
    "fetch_url": tool("fetch_url", "Fetch a web page", url="URL"),
    "list_files": tool("list_files", "List files under a path in the repository", path="path"),
    "read_file": tool("read_file", "Read a file from the repository", path="path"),
    "search_products": tool("search_products", "Search the supplier catalogue", query="search text"),
    "get_customer": tool("get_customer", "Look up a customer account", name="customer name"),
    "list_tickets": tool("list_tickets", "List support tickets", account="account"),
    "search_docs": tool("search_docs", "Search the internal policy documents", query="search text"),
    "get_job_log": tool("get_job_log", "Get the log of a CI job", job="job id"),
}


@dataclass
class Conversation:
    id: str
    family: str
    system: str
    rng: random.Random
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    _calls: int = 0

    def __post_init__(self) -> None:
        self.messages.append({"role": "system", "content": self.system})

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def call(self, name: str, result: str, say: str | None = None, /, **args: str) -> None:
        self._calls += 1
        cid = f"call_{self.id.replace('-', '_')}_{self._calls}"
        if name not in self.tools_used:
            self.tools_used.append(name)
        self.messages.append(
            {
                "role": "assistant",
                "content": say,
                "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
            }
        )
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": result})

    def to_dict(self, question: str, answers: list[list[str]], evidence: list[Any], reference: str, kind: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "kind": kind,
            "question": question,
            "reference_answer": reference,
            "answers": answers,
            "evidence": evidence,
            "tools": [TOOLS[t] for t in self.tools_used],
            "messages": self.messages,
        }


# --- generators --------------------------------------------------------------------------------------------


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def hexid(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def timestamp(rng: random.Random, hour: int, minute: int, second: int) -> str:
    return f"2026-09-14T{hour:02d}:{minute:02d}:{second:02d}.{rng.randint(0, 999):03d}Z"


def log_lines(rng: random.Random, service: str, n: int, start_hour: int = 2) -> list[str]:
    """Typical service logs: request lines, health checks, cache stats and the odd harmless warning."""
    paths = ["/api/orders", "/api/cart", "/api/products", "/api/users/me", "/api/stock", "/api/prices", "/api/session"]
    lines = []
    sec = 0
    for _ in range(n):
        sec += rng.randint(1, 4)
        t = timestamp(rng, start_hour + sec // 3600, (sec // 60) % 60, sec % 60)
        r = rng.random()
        if r < 0.55:
            p = rng.choice(paths)
            if rng.random() < 0.5:
                p += f"/{rng.randint(10000, 99999)}"
            lines.append(
                f"{t} INFO  [{service}] {rng.choice(['GET', 'GET', 'POST', 'PUT'])} {p} {rng.choice([200, 200, 200, 201, 204, 304])} {rng.randint(3, 180)}ms req={hexid(rng)}"
            )
        elif r < 0.80:
            lines.append(f"{t} INFO  [{service}] health check ok status=UP uptime={rng.randint(1000, 99999)}s")
        elif r < 0.92:
            lines.append(f"{t} DEBUG [{service}] cache stats hits={rng.randint(1000, 9000)} misses={rng.randint(10, 400)} evictions={rng.randint(0, 30)}")
        elif r < 0.97:
            lines.append(
                f"{t} INFO  [{service}] background job {rng.choice(['metrics-flush', 'session-sweep', 'price-refresh'])} finished in {rng.randint(20, 900)}ms"
            )
        else:
            lines.append(f"{t} WARN  [{service}] slow request {rng.choice(paths)} took {rng.randint(900, 2400)}ms (threshold 800ms)")
    return lines


def df_output(rng: random.Random, host: str) -> str:
    rows = ["Filesystem      Size  Used Avail Use% Mounted on"]
    for fs, mnt in [("/dev/nvme0n1p1", "/"), ("/dev/nvme1n1", "/var/lib/data"), ("tmpfs", "/run"), ("/dev/nvme0n1p15", "/boot/efi")]:
        size = rng.choice([30, 50, 100, 200, 500])
        used = rng.randint(1, size - 1)
        rows.append(f"{fs:<15} {size}G {used:>4}G {size - used:>4}G {100 * used // size:>3}% {mnt}")
    return f"$ df -h   (host {host})\n" + "\n".join(rows)


def ps_output(rng: random.Random, host: str) -> str:
    procs = [
        "nginx: worker process",
        "node server.js",
        "postgres: autovacuum",
        "python worker.py",
        "sshd",
        "containerd",
        "kubelet",
        "node-exporter",
        "fluent-bit",
    ]
    rows = [f"$ ps aux --sort=-%cpu | head -40   (host {host})", "USER       PID %CPU %MEM    VSZ   RSS TTY STAT START   TIME COMMAND"]
    for _ in range(40):
        rows.append(
            f"{rng.choice(['root', 'app', 'postgres', 'www-data']):<9} {rng.randint(100, 65000):>5} {rng.random() * 30:>4.1f} {rng.random() * 12:>4.1f} "
            f"{rng.randint(10000, 2000000):>7} {rng.randint(1000, 900000):>6} ?   Ssl  02:{rng.randint(10, 59)}   {rng.randint(0, 59)}:{rng.randint(10, 59)} {rng.choice(procs)}"
        )
    return "\n".join(rows)


def tickets_json(rng: random.Random, account: str, n: int) -> str:
    subjects = [
        "Invoice shows wrong GST",
        "Cannot reset password",
        "Order arrived damaged",
        "Request for bulk pricing",
        "Delivery delayed",
        "Change billing address",
        "Export to CSV fails",
        "Add a new user",
    ]
    items = [
        {
            "ticket": f"TCK-{rng.randint(10000, 99999)}",
            "account": account,
            "subject": rng.choice(subjects),
            "status": rng.choice(["open", "pending", "solved", "solved", "closed"]),
            "priority": rng.choice(["low", "normal", "normal", "high"]),
            "opened": f"2026-0{rng.randint(6, 9)}-{rng.randint(10, 28)}",
            "assignee": rng.choice(FIRST),
        }
        for _ in range(n)
    ]
    return json.dumps(items, indent=2)


def filler_turn(c: Conversation, rng: random.Random) -> None:
    """An unrelated earlier step: the kind of side task that fills long agent sessions."""
    kind = rng.choice(["df", "ps", "tickets", "chat", "chat"])
    host = f"web-{rng.randint(1, 9)}"
    if kind == "df":
        c.user(f"While you're there, how's disk space on {host}?")
        c.call("run_command", df_output(rng, host), host=host, command="df -h")
        c.assistant(f"Disk on {host} looks fine; nothing is above 90%.")
    elif kind == "ps":
        c.user(f"Can you check what's using CPU on {host}?")
        c.call("run_command", ps_output(rng, host), host=host, command="ps aux --sort=-%cpu | head -40")
        c.assistant(f"Nothing unusual on {host}; the top processes are the usual web workers.")
    elif kind == "tickets":
        acct = rng.choice(["Acme Outdoor Gear", "Acme Garden Supply", "Acme Coffee Roasters"])
        c.user(f"Quickly, are there any high-priority tickets for {acct}?")
        c.call("list_tickets", tickets_json(rng, acct, rng.randint(12, 30)), account=acct)
        c.assistant(f"I checked the ticket list for {acct}; a couple are high priority but none are urgent.")
    else:
        topic = rng.choice(
            [
                (
                    "Can you remind me how our on-call rota works this week?",
                    "This week the primary on-call rotates daily and the secondary covers the whole week; handovers are at 9am.",
                ),
                ("What's the naming convention for new branches?", "Branches use type/short-description, for example fix/cart-rounding or feat/export-csv."),
                ("Should I loop in the support team yet?", "Not yet; let's confirm the cause first so support can give customers an accurate update."),
                ("Draft a one-line status update for the team channel.", "Status: investigating, no customer data affected, next update in 30 minutes."),
            ]
        )
        c.user(topic[0])
        c.assistant(topic[1])
