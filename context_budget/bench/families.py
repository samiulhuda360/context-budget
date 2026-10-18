"""The eight conversation families of the benchmark.

Each function builds one long agent conversation for an invented business, with tool outputs in the shapes real
agents see (logs, database rows, web pages, file listings, CI output, documents), a final question, the labelled
answer, and the evidence strings the answer depends on.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .builder import FIRST, LAST, MONTHS, REGIONS, Conversation, filler_turn, hexid, log_lines, person

ANSWER_RULE = " When the user asks a direct question, answer in at most two sentences from what you have already found, without calling more tools."

SYSTEM_OPS = (
    "You are the operations assistant for Acme Outdoor Gear, an online outdoor retailer. Use the tools to investigate and answer briefly and precisely."
    + ANSWER_RULE
)
SYSTEM_DATA = "You are the data assistant for Acme Outdoor Gear. You can query the analytics replica with run_sql. Answer with exact figures." + ANSWER_RULE
SYSTEM_SHOP = "You are a purchasing assistant for Acme Adventure Tours, a small tour operator. You research gear for the team." + ANSWER_RULE
SYSTEM_CODE = "You are a coding assistant working in the Acme Payments monorepo. Be precise about file paths and values." + ANSWER_RULE
SYSTEM_CS = "You are the account-management assistant for Acme Cloud Backup, a business backup service." + ANSWER_RULE
SYSTEM_POLICY = "You are the customer-service assistant for Acme Outdoor Gear. Answer from the policy documents." + ANSWER_RULE
SYSTEM_CI = "You are a CI assistant for the Acme Payments engineering team. You read build logs and explain failures." + ANSWER_RULE


def fillers(c: Conversation, rng: random.Random, lo: int, hi: int) -> None:
    for _ in range(rng.randint(lo, hi)):
        filler_turn(c, rng)


# --- 1. log incident -----------------------------------------------------------------------------------------

INCIDENTS: list[dict[str, Any]] = [
    {
        "service": "checkout-api",
        "report": "checkout-api has been returning 502s since about 3am. Can you find out why?",
        "line": "ERROR [checkout-api] db pool exhausted: 20/20 connections busy on {host} (waited 5000ms) request=POST /api/checkout",
        "host": "orders-db-{n}",
        "question": "So what caused the 502s on checkout, and which database host was involved?",
        "answers": [["exhaust", "pool"], ["{host}"]],
        "reference": "The connection pool to {host} was exhausted (20/20 connections busy), so checkout requests timed out.",
    },
    {
        "service": "payments-worker",
        "report": "Payments have been stuck in 'processing' for an hour. Please investigate.",
        "line": "WARN  [payments-worker] upstream acquirer {host} did not answer after 30000ms; circuit breaker opening for card payments",
        "host": "acq-nz-{n}",
        "question": "What was holding up the payments, and which acquirer node was it?",
        "answers": [
            ["30000", "30 s", "30-second", "30 second", "timed out", "time out", "timeout", "did not answer", "not respond", "unresponsive", "circuit"],
            ["{host}"],
        ],
        "reference": "The acquirer node {host} stopped answering (30 s timeouts) and the circuit breaker opened.",
    },
    {
        "service": "inventory-sync",
        "report": "Stock levels on the website stopped updating overnight. What happened?",
        "line": "ERROR [inventory-sync] TLS handshake failed: certificate expired for host {host} (notAfter=2026-09-13T23:59:59Z)",
        "host": "feed{n}.supplier.acme.example",
        "question": "Why did the stock sync stop, and which host was it talking to?",
        "answers": [["certificate", "cert", "tls"], ["{host}"]],
        "reference": "The TLS certificate for {host} expired at midnight on 13 September, so the sync could not connect.",
    },
    {
        "service": "search-indexer",
        "report": "The search indexer keeps restarting. Can you see why?",
        "line": "FATAL [search-indexer] container {host} OOMKilled: memory limit 512Mi exceeded while loading synonyms",
        "host": "search-indexer-{n}c9d",
        "question": "Why does the indexer keep restarting, and what memory limit is it hitting?",
        "answers": [["oom", "out of memory", "memory"], ["512"]],
        "reference": "The container {host} is OOMKilled: it exceeds its 512Mi memory limit while loading synonyms.",
    },
    {
        "service": "order-events",
        "report": "Order confirmation emails stopped going out around 4am. What broke?",
        "line": "ERROR [order-events] write failed: No space left on device (/var/lib/queue/segment-00412.log) on node {host}",
        "host": "worker-{n}",
        "question": "Why did the order emails stop, and on which node?",
        "answers": [["space", "disk", "full"], ["{host}"]],
        "reference": "Node {host} ran out of disk space, so the order-events queue could not write.",
    },
]


def log_incident(i: int, rng: random.Random) -> dict[str, Any]:
    spec = INCIDENTS[i % len(INCIDENTS)]
    host = spec["host"].format(n=rng.randint(2, 9))
    c = Conversation(f"log-{i:02d}", "log_incident", SYSTEM_OPS, rng)
    fillers(c, rng, 0, 2)
    c.user(spec["report"])
    deploys = [
        {
            "service": spec["service"],
            "version": f"v2.{rng.randint(10, 60)}.{k}",
            "deployed_at": f"2026-09-{rng.randint(1, 12):02d}T{rng.randint(0, 23):02d}:00Z",
            "by": rng.choice(FIRST),
        }
        for k in range(rng.randint(15, 30))
    ]
    c.call("get_deploys", json.dumps(deploys, indent=2), "Let me check recent deploys first.", service=spec["service"])
    lines = log_lines(rng, spec["service"], rng.randint(500, 1000))
    pos = int(len(lines) * rng.uniform(0.3, 0.7))
    needle = spec["line"].format(host=host)
    stamp = lines[pos].split(" ")[0]
    lines.insert(pos, f"{stamp} {needle}")
    for k in range(rng.randint(2, 5)):  # the same failure repeating a little later
        p = pos + 1 + k * rng.randint(3, 9)
        lines.insert(p, f"{lines[p - 1].split(' ')[0]} {needle}")
    c.call(
        "fetch_logs",
        "\n".join(lines),
        "No deploy in the last 24 hours, so this is not a bad release. Pulling the logs.",
        service=spec["service"],
        since="2026-09-14T02:00:00Z",
    )
    c.assistant("I've got the logs; there's a lot of routine traffic in there.")
    fillers(c, rng, 0, 1)
    c.user(spec["question"])
    answers = [[a.format(host=host) for a in group] for group in spec["answers"]]
    return c.to_dict(spec["question"], answers, [needle], spec["reference"].format(host=host), "needle in a long log")


# --- 2. database rows ----------------------------------------------------------------------------------------


def orders(rng: random.Random, n: int, region: str | None = None) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        out.append(
            {
                "order_id": f"ORD-{rng.randint(50000, 99999)}",
                "customer": person(rng),
                "region": region or rng.choice(REGIONS),
                "status": rng.choices(["delivered", "shipped", "processing", "refunded", "cancelled"], [60, 15, 10, 8, 7])[0],
                "total_nzd": round(rng.uniform(19, 900), 2),
                "placed_at": f"2026-08-{rng.randint(1, 31):02d}",
                "items": rng.randint(1, 9),
            }
        )
    return out


def db_rows(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"db-{i:02d}", "db_rows", SYSTEM_DATA, rng)
    fillers(c, rng, 0, 2)
    kind = ["lookup", "count", "sum", "lookup", "count"][i % 5]
    region = rng.choice(REGIONS)
    rows = orders(rng, rng.randint(140, 320), region if kind != "lookup" else None)
    c.user(f"Pull last month's orders{' for ' + region if kind != 'lookup' else ''}; I've got a few questions about them.")
    where = f"region = '{region}' AND " if kind != "lookup" else ""
    sql = f"SELECT order_id, customer, region, status, total_nzd, placed_at, items FROM orders WHERE {where}placed_at >= '2026-08-01'"  # noqa: S608 - text shown to the model, never executed
    c.call("run_sql", json.dumps(rows, indent=2), query=sql)
    c.assistant(f"I have {len(rows)} orders from August. What would you like to know?")
    fillers(c, rng, 0, 1)
    if kind == "lookup":
        row = rows[int(len(rows) * rng.uniform(0.25, 0.75))]
        q = f"Who placed {row['order_id']}, and what was its total?"
        ans = [[row["customer"]], [f"{row['total_nzd']:.2f}".rstrip("0").rstrip(".")]]
        ev = [row["order_id"], row["customer"]]
        ref = f"{row['order_id']} was placed by {row['customer']} for NZ${row['total_nzd']:.2f}."
    elif kind == "count":
        status = rng.choice(["refunded", "cancelled", "processing"])
        n = sum(1 for r in rows if r["status"] == status)
        q = f"How many of those {region} orders were {status}?"
        ans = [[str(n)]]
        ev = [{"count_of": [f'"status": "{status}"', f'"status":"{status}"'], "equals": n, "or_any_of": [f'"{status}":{{"count":{n},']}]
        ref = f"{n} of the {len(rows)} {region} orders were {status}."
    else:
        refunded = [r for r in rows if r["status"] == "refunded"]
        total = round(sum(r["total_nzd"] for r in refunded), 2)
        q = f"What is the combined value of the refunded {region} orders?"
        ans = [[f"{total:,.2f}", f"{total:.2f}"]]
        ev = [
            {
                "all_of": [r["order_id"] for r in refunded],
                "or_any_of": [f'"refunded":{{"count":{len(refunded)},"items_sum":{float(sum(r["items"] for r in refunded))},"total_nzd_sum":{total}}}'],
            }
        ]
        ref = f"The {len(refunded)} refunded orders total NZ${total:,.2f}."
    c.user(q)
    return c.to_dict(q, ans, ev, ref, f"database rows ({kind})")


# --- 3. web research -----------------------------------------------------------------------------------------

PRODUCTS = ["TrailPro X200 tent", "Summit 45L pack", "Ridgeline down jacket", "Alpine stove kit", "Coastal kayak paddle", "Nomad sleeping bag"]


def product_page(rng: random.Random, name: str, specs: dict[str, str]) -> str:
    nav = "\n".join(
        f'<li class="nav-item"><a href="/c/{k}">{k.title()}</a></li>'
        for k in ["tents", "packs", "jackets", "cooking", "paddling", "sleeping", "sale", "brands", "gift cards", "stores"] * 3
    )
    css = "\n".join(f".c{k}{{margin:{k}px;padding:{k % 7}px;color:#{hexid(rng, 6)}}}" for k in range(rng.randint(60, 120)))
    script = "\n".join(
        f"window.__data_{k}={{id:'{hexid(rng)}',t:{rng.randint(1000, 9999)},flags:[{','.join(str(rng.randint(0, 1)) for _ in range(12))}]}};"
        for k in range(rng.randint(30, 60))
    )
    rows = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in specs.items())
    reviews = "\n".join(
        f'<div class="review"><p class="author">{rng.choice(FIRST)} {rng.choice(LAST)[0]}.</p><p>{rng.choice(["Great value for the price.", "Held up well on a wet weekend in the Tararuas.", "A bit heavy but very sturdy.", "Packs down small, easy to carry.", "Zips feel cheap but it does the job.", "Bought a second one for my partner."])} '
        f"{rng.choice(['Would recommend.', 'Four stars.', 'Customer service was helpful.', 'Delivery took a week.'])}</p></div>"
        for _ in range(rng.randint(12, 25))
    )
    footer = "\n".join(
        f'<a href="/help/{k}">{k}</a>' for k in ["shipping", "returns", "warranty", "contact", "privacy", "terms", "careers", "stores", "sitemap"] * 2
    )
    return (
        f"<!doctype html><html><head><title>{name} | Acme Outfitters</title><style>{css}</style><script>{script}</script></head><body>"
        f'<header><nav><ul class="nav">{nav}</ul></nav></header><main><h1>{name}</h1>'
        f"<p>The {name} is built for multi-day trips in changeable weather.</p>"
        f'<h2>Specifications</h2><table class="specs">{rows}</table><h2>Reviews</h2>{reviews}</main>'
        f"<footer>{footer}<p>Copyright 2026 Acme Outfitters Ltd</p></footer></body></html>"
    )


def web_research(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"web-{i:02d}", "web_research", SYSTEM_SHOP, rng)
    picks = rng.sample(PRODUCTS, 3)
    target = picks[rng.randint(0, 2)]
    specs_by: dict[str, dict[str, str]] = {}
    c.user(f"We're kitting out guides for the summer season. Compare the {', '.join(picks)} for me.")
    for p in picks:
        specs = {
            "Weight": f"{rng.uniform(0.4, 4.5):.1f} kg",
            "Warranty": f"{rng.choice([1, 2, 3, 5, 10])} years",
            "Price": f"NZ${rng.randint(89, 1299)}",
            "Colour": rng.choice(["Teal", "Charcoal", "Sand", "Forest green"]),
            "Material": rng.choice(["Ripstop nylon", "Polyester 75D", "Recycled nylon", "Aluminium"]),
            "Code": f"SKU-{rng.randint(10000, 99999)}",
        }
        specs_by[p] = specs
        url = "https://shop.acme-outfitters.example/p/" + p.lower().replace(" ", "-")
        c.call("fetch_url", product_page(rng, p, specs), url=url)
    c.assistant("I've pulled the three product pages.")
    fillers(c, rng, 0, 2)
    s = specs_by[target]
    model = target.split(" ")[1] if "X200" in target else target.rsplit(" ", 1)[0]
    variant = i % 3
    if variant == 0:
        q = f"What's the warranty on the {model}, and what does it weigh?"
        ans = [[s["Warranty"].split()[0] + " year"], [s["Weight"].replace(" kg", "")]]
        ev = [s["Warranty"], s["Weight"]]
        ref = f"The {target} has a {s['Warranty']} warranty and weighs {s['Weight']}."
    elif variant == 1:
        q = f"How much does the {model} cost, and what's its SKU code?"
        ans = [[s["Price"].replace("NZ$", "")], [s["Code"]]]
        ev = [s["Price"], s["Code"]]
        ref = f"The {target} costs {s['Price']} (code {s['Code']})."
    else:
        q = f"How heavy is the {model}, and what is it made of?"
        ans = [[s["Weight"].replace(" kg", "")], [s["Material"].split()[0]]]
        ev = [s["Weight"], s["Material"]]
        ref = f"The {target} weighs {s['Weight']} and is made of {s['Material'].lower()}."
    c.user(q)
    return c.to_dict(q, ans, ev, ref, "spec in an HTML page")


# --- 4. file listing and source --------------------------------------------------------------------------------


def file_tree(rng: random.Random, n_migrations: int) -> list[str]:
    services = ["billing", "ledger", "payouts", "risk", "notifications", "gateway", "reports"]
    dirs = ["handlers", "models", "clients", "utils", "tests", "schemas"]
    words = [
        "invoice",
        "refund",
        "card",
        "token",
        "webhook",
        "retry",
        "ledger",
        "account",
        "audit",
        "currency",
        "fx",
        "batch",
        "export",
        "limit",
        "fraud",
        "payout",
    ]
    files = []
    for s in services:
        for d in dirs:
            for _ in range(rng.randint(8, 22)):
                files.append(f"services/{s}/{d}/{rng.choice(words)}_{rng.choice(words)}.py")
        files.append(f"services/{s}/config/settings.yaml")
        files.append(f"services/{s}/README.md")
    for k in range(1, n_migrations + 1):
        files.append(f"db/migrations/{k:04d}_{rng.choice(words)}_{rng.choice(['add', 'drop', 'alter', 'index'])}.sql")
    files += [f"web/src/components/{rng.choice(words).title()}{rng.choice(['Card', 'Table', 'Modal', 'Form'])}.tsx" for _ in range(rng.randint(150, 260))]
    rng.shuffle(files)
    return sorted(set(files))


def config_file(rng: random.Random, service: str, key_line: str) -> str:
    lines = [
        f"# {service} service settings",
        "service:",
        f"  name: {service}",
        f"  port: {rng.randint(8000, 8999)}",
        "logging:",
        "  level: info",
        "  format: json",
    ]
    for section in ["database", "cache", "queue", "http_client", "feature_flags", "metrics"]:
        lines.append(f"{section}:")
        for _ in range(rng.randint(6, 14)):
            lines.append(
                f"  {rng.choice(['timeout_ms', 'pool_size', 'enabled', 'endpoint', 'batch_size', 'ttl_s', 'max_idle', 'region'])}_{rng.randint(1, 99)}: {rng.choice(['true', 'false', str(rng.randint(1, 5000)), 'ap-southeast-2'])}"
            )
    pos = rng.randint(10, len(lines) - 5)
    lines[pos:pos] = ["payment_retry:", key_line, f"  backoff_ms: {rng.randint(200, 2000)}"]
    return "\n".join(lines)


def file_listing(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"files-{i:02d}", "file_listing", SYSTEM_CODE, rng)
    n_mig = rng.randint(18, 60)
    tree = file_tree(rng, n_mig)
    c.user("I'm new to this repo. Give me a lay of the land first.")
    c.call("list_files", "\n".join(tree), path=".")
    c.assistant(f"The repo has {len(tree)} files: seven Python services, a React front end under web/ and SQL migrations under db/.")
    fillers(c, rng, 0, 2)
    if i % 2 == 0:
        service = rng.choice(["billing", "payouts", "gateway"])
        limit = rng.randint(3, 12)
        key_line = f"  max_attempts: {limit}"
        c.user("Where do we configure how many times a failed card payment is retried?")
        c.call("read_file", config_file(rng, service, key_line), path=f"services/{service}/config/settings.yaml")
        c.assistant("Found the settings file; reading through it.")
        q = "Which file sets the payment retry limit, and what is the limit?"
        ans = [[f"services/{service}/config/settings.yaml", f"{service}/config/settings.yaml"], [str(limit)]]
        ev: list[Any] = [key_line.strip()]
        ref = f"services/{service}/config/settings.yaml sets payment_retry.max_attempts to {limit}."
        kind = "value in a config file"
    else:
        q = "How many SQL migration files are there under db/migrations?"
        ans = [[str(n_mig)]]
        ev = [{"count_of": ["db/migrations/"], "equals": n_mig}]
        ref = f"There are {n_mig} migration files."
        kind = "count in a file listing"
    c.user(q)
    return c.to_dict(q, ans, ev, ref, kind)


# --- 5. a constraint stated early -------------------------------------------------------------------------------

CONSTRAINTS = [
    ("Budget for the whole order is NZ${budget} and it has to arrive in Queenstown by {date}.", "budget"),
    ("Hard limits: we can spend at most NZ${budget}, and the gear must be delivered by {date}.", "spend"),
]


def catalogue(rng: random.Random, query: str) -> str:
    items = [
        {
            "sku": f"SKU-{rng.randint(10000, 99999)}",
            "name": f"{rng.choice(['Acme', 'Kea', 'Tussock', 'Ridgeline'])} {query} {rng.choice(['Lite', 'Pro', 'Max', 'Trek', 'Basecamp'])}",
            "price_nzd": rng.randint(39, 899),
            "in_stock": rng.choice([True, True, False]),
            "ships_in_days": rng.randint(1, 14),
            "rating": round(rng.uniform(3.2, 4.9), 1),
            "description": rng.choice(
                ["Lightweight and packable.", "Built for alpine conditions.", "Best seller for school groups.", "Includes a lifetime repair service."]
            ),
        }
        for _ in range(rng.randint(25, 45))
    ]
    return json.dumps(items, indent=2)


def early_constraint(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"brief-{i:02d}", "early_constraint", SYSTEM_SHOP, rng)
    budget = rng.randint(12, 48) * 100 + rng.choice([0, 50])
    day = rng.randint(3, 27)
    month = MONTHS[rng.randint(9, 11)]
    date = f"{day} {month}"
    template, word = CONSTRAINTS[i % 2]
    c.user(f"I'm ordering gear for {rng.randint(8, 20)} guides. " + template.format(budget=f"{budget:,}", date=date) + " Let's look at options.")
    c.assistant("Got it. I'll search the catalogue category by category.")
    for q in rng.sample(["tent", "sleeping bag", "rain jacket", "headlamp", "first aid kit", "stove", "water filter"], rng.randint(4, 6)):
        c.user(f"Show me {q}s.")
        c.call("search_products", catalogue(rng, q.title()), query=q)
        c.assistant(f"Here are the {q} options; several are in stock and ship within a week.")
        if rng.random() < 0.5:
            filler_turn(c, rng)
    if i % 4 == 3:
        q = "Before I approve this: how much did I say we could spend, and when does it need to arrive?"
    else:
        q = f"Remind me of the {word} limit and the delivery deadline I gave you at the start."
    c.user(q)
    return c.to_dict(
        q,
        [[f"{budget:,}", str(budget)], [str(day)]],
        [f"NZ${budget:,}", date],
        f"The limit is NZ${budget:,} and the gear must arrive by {date}.",
        "constraint stated early",
    )


# --- 6. a stale tool result referenced later -----------------------------------------------------------------------


def customer_record(rng: random.Random, name: str, acct: str, renewal: str) -> str:
    rec = {
        "account_id": acct,
        "name": name,
        "company": f"{rng.choice(LAST)} {rng.choice(['Builders', 'Dental', 'Accounting', 'Law', 'Architects'])} Ltd",
        "plan": rng.choice(["Team 2TB", "Business 10TB", "Starter 500GB"]),
        "renewal_date": renewal,
        "seats": rng.randint(3, 60),
        "billing_contact": f"{name.split(maxsplit=1)[0].lower()}@{rng.choice(['example.com', 'example.org'])}",
        "devices": [
            {
                "device": f"{rng.choice(['LAPTOP', 'DESKTOP', 'NAS'])}-{hexid(rng, 4).upper()}",
                "last_backup": f"2026-09-{rng.randint(1, 14):02d}",
                "size_gb": rng.randint(20, 900),
            }
            for _ in range(rng.randint(15, 40))
        ],
        "invoices": [
            {"invoice": f"INV-{rng.randint(20000, 29999)}", "amount_nzd": round(rng.uniform(40, 900), 2), "paid": rng.choice([True, True, False])}
            for _ in range(rng.randint(8, 18))
        ],
    }
    return json.dumps(rec, indent=2)


def date_spellings(iso: str) -> list[str]:
    """The same date as 2026-10-12, October 12, 2026 or 12 October 2026."""
    y, m, d = (int(x) for x in iso.split("-"))
    month = MONTHS[m - 1]
    return [iso, f"{month} {d}, {y}", f"{d} {month} {y}"]


def stale_reference(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"stale-{i:02d}", "stale_reference", SYSTEM_CS, rng)
    name = person(rng)
    acct = f"ACB-{rng.randint(100000, 999999)}"
    renewal = f"2026-{rng.randint(10, 12)}-{rng.randint(10, 28)}"
    c.user(f"Look up {name}'s account; they emailed about storage.")
    c.call("get_customer", customer_record(rng, name, acct, renewal), name=name)
    c.assistant(f"Found {name.split()[0]}'s account. They have a lot of devices backing up; storage use is close to the plan limit.")
    for _ in range(rng.randint(3, 5)):
        other = person(rng)
        c.user(f"Now check {other}'s account too.")
        c.call(
            "get_customer", customer_record(rng, other, f"ACB-{rng.randint(100000, 999999)}", f"2027-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"), name=other
        )
        c.assistant(f"{other.split()[0]}'s account looks healthy; all devices backed up this week.")
        if rng.random() < 0.5:
            filler_turn(c, rng)
    if i % 10 < 7:
        q = f"Going back to {name.split()[0]}: what's their renewal date and account ID?"
    else:
        q = "Going back to the very first account I asked about: what's its renewal date and account ID?"
    c.user(q)
    return c.to_dict(
        q,
        [date_spellings(renewal), [acct]],
        [renewal, acct],
        f"{name}'s account {acct} renews on {renewal}.",
        "old result referenced" + (" indirectly" if i % 10 >= 7 else ""),
    )


# --- 7. duplicated documents --------------------------------------------------------------------------------------

POLICY = {
    "returns": "Returns policy. Unopened items can be returned within {days} days of delivery for a full refund. Opened items can be returned within {days} days if they are in resaleable condition; a restocking fee of {fee}% applies to opened items. Refunds are paid to the original payment method within 5 working days.",
    "warranty": "Warranty policy. Acme-branded products carry a two-year warranty against manufacturing faults. Claims need proof of purchase. Damage from misuse, accidents or normal wear is not covered. Repairs are offered first; a replacement or refund is offered if a repair is not possible.",
    "shipping": "Shipping policy. Orders over NZ$150 ship free within New Zealand. Standard delivery takes 2 to 5 working days; rural addresses can take two days longer. Orders placed before 1pm ship the same day.",
    "pricematch": "Price match policy. We match the price of an identical in-stock item from a New Zealand retailer's own website. Price matching excludes clearance sales, marketplace sellers and bundles.",
    "loyalty": "Loyalty programme. Members earn one point per dollar spent. 500 points can be redeemed for a NZ$25 voucher. Points expire after 18 months without a purchase.",
    "privacy": "Privacy summary. Customer details are used only to fulfil orders and, with consent, for marketing. Customers can ask for a copy of their data or for it to be deleted at any time.",
}


def duplicate_docs(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"docs-{i:02d}", "duplicate_docs", SYSTEM_POLICY, rng)
    days = rng.choice([14, 21, 30, 45, 60])
    fee = rng.choice([10, 15, 20, 25])
    docs = {k: v.format(days=days, fee=fee) for k, v in POLICY.items()}
    for k in docs:  # policies are long in practice: add the boilerplate sections
        docs[k] += "\n\n" + " ".join(
            rng.choice(
                [
                    "This policy applies to purchases made online and in our stores.",
                    "It does not limit your rights under the Consumer Guarantees Act.",
                    "We review this policy every year and publish changes on our website.",
                    "Staff may use discretion in exceptional circumstances with a manager's approval.",
                    "Gift cards and vouchers are excluded unless stated otherwise.",
                ]
            )
            for _ in range(rng.randint(10, 18))
        )
    customer = rng.choice(FIRST)
    c.user(f"{customer} wants to return a tent they've used once. What can we offer?")
    for n, q in enumerate(["return opened item", "refund tent used", "restocking fee", "returns warranty", "return condition"]):
        hits = rng.sample(list(docs), 3)
        if "returns" not in hits and (n == 0 or rng.random() < 0.7):
            hits[0] = "returns"
        result = "\n\n---\n\n".join(f"[{k}.md]\n{docs[k]}" for k in hits)
        c.call("search_docs", result, query=q)
    c.assistant("The search returned the returns policy several times, along with the warranty and shipping policies.")
    fillers(c, rng, 0, 2)
    q = f"So how many days does {customer} have to return an opened item, and what restocking fee applies?"
    c.user(q)
    if not any("[returns.md]" in str(m.get("content")) for m in c.messages):
        c.call("search_docs", f"[returns.md]\n{docs['returns']}", query="returns")
    return c.to_dict(
        q,
        [[str(days)], [f"{fee}%", f"{fee} percent", f"{fee} per cent"]],
        [f"within {days} days", f"{fee}%"],
        f"{days} days for opened items, with a {fee}% restocking fee.",
        "repeated documents",
    )


# --- 8. CI failure -------------------------------------------------------------------------------------------------


def ci_log(rng: random.Random, failing: str, assertion: str, n: int) -> str:
    mods = ["test_invoice", "test_refund", "test_ledger", "test_fx", "test_payout", "test_webhook", "test_tokens", "test_limits", "test_audit"]
    out = [
        "Run actions/checkout@v4",
        "Run actions/setup-python@v5",
        "Successfully installed " + " ".join(f"pkg{k}-{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}" for k in range(40)),
        "============================= test session starts ==============================",
        f"platform linux -- Python 3.12.{rng.randint(1, 7)}, pytest-8.3.{rng.randint(1, 5)}",
        f"collected {n} items",
    ]
    for k in range(n):
        out.append(f"tests/{rng.choice(mods)}.py::test_case_{k:04d}_{hexid(rng, 4)} PASSED [{100 * k // n:3d}%]")
    pos = rng.randint(n // 3, 2 * n // 3) + 6
    out.insert(pos, f"tests/{failing} FAILED [{100 * (pos - 6) // n:3d}%]")
    out += [
        "=================================== FAILURES ===================================",
        f"___________________________ {failing.split('::')[1]} ___________________________",
        "    def test():",
        f">       {assertion}",
        f"E       AssertionError: {assertion.replace('assert ', 'assert ')}",
        "",
        f"tests/{failing.split('::', maxsplit=1)[0]}:{rng.randint(20, 300)}: AssertionError",
        "=========================== short test summary info ============================",
        f"FAILED tests/{failing} - AssertionError",
        f"========== 1 failed, {n - 1} passed, {rng.randint(0, 9)} warnings in {rng.uniform(40, 300):.2f}s ==========",
        "Error: Process completed with exit code 1.",
    ]
    return "\n".join(out)


def ci_failure(i: int, rng: random.Random) -> dict[str, Any]:
    c = Conversation(f"ci-{i:02d}", "ci_failure", SYSTEM_CI, rng)
    a, b = rng.randint(100, 999), rng.randint(0, 99)
    name = rng.choice(["test_gst_rounding", "test_refund_partial", "test_fx_conversion_nzd_aud", "test_payout_batch_limit", "test_webhook_signature"])
    failing = f"{rng.choice(['test_invoice', 'test_refund', 'test_fx', 'test_payout', 'test_webhook'])}.py::{name}"
    assertion = f"assert {a}.{b:02d} == {a}.{(b + 1) % 100:02d}"
    c.user("Last night's build on main went red. What broke?")
    for k in range(rng.randint(1, 2)):  # earlier green runs the agent looked at first
        c.call(
            "get_job_log",
            ci_log(rng, "test_smoke.py::test_ok", "assert True", rng.randint(200, 400)).replace("FAILED", "XFAIL"),
            "Checking the previous runs first." if k == 0 else None,
            job=f"job-{rng.randint(10000, 99999)}",
        )
    c.assistant("The earlier runs had only expected failures. Pulling last night's job.")
    fillers(c, rng, 0, 1)
    c.user("Okay, now last night's run please.")
    c.call("get_job_log", ci_log(rng, failing, assertion, rng.randint(400, 900)), job=f"job-{rng.randint(10000, 99999)}")
    c.assistant("Got the log for last night's run.")
    q = "Which test failed, and what values did the assertion compare?"
    c.user(q)
    return c.to_dict(q, [[name], [f"{a}.{b:02d}"]], [failing, assertion], f"{failing} failed: {assertion}.", "failure in a CI log")


FAMILIES = [log_incident, db_rows, web_research, file_listing, early_constraint, stale_reference, duplicate_docs, ci_failure]
