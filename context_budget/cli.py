"""The `ctxb` command line.

ctxb analyse conversation.json     show what would be cut and why (no model call)
ctxb serve --upstream mock         run the proxy (mock upstream for a keyless demo)
ctxb demo                          analyse a sample incident conversation
ctxb bench [--out DIR] [--check]   rebuild the benchmark conversations
ctxb eval offline|model|report     run the evaluation and write eval/
ctxb check-policy policy.yaml      validate a policy file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .llm import LLM, NoModel, RateLimited
from .policy import STRATEGIES, PolicyError, load


def _messages(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    messages = data if isinstance(data, list) else data.get("messages")
    if not isinstance(messages, list):
        raise SystemExit(f"{path}: expected a list of messages or an object with a 'messages' list")
    return messages


def _print_table(rows: list[list[str]], headers: list[str], width: int = 70) -> None:
    rows = [[c if len(c) <= width else c[: width - 3] + "..." for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    right = {2, 3}
    print("  ".join(h.rjust(w) if i in right else h.ljust(w) for i, (h, w) in enumerate(zip(headers, widths, strict=True))))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.rjust(w) if i in right else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))).rstrip())


def cmd_analyse(args: argparse.Namespace) -> int:
    from .analysis import explain, table
    from .pipeline import compress
    from .proxy import page
    from .tokens import count_text

    route = load(args.policy).route(args.route)
    if args.strategies is not None or args.budget or args.relevance:
        chosen = args.strategies if args.strategies is not None else ",".join(route.enabled)
        enabled = [s.strip() for s in chosen.split(",") if s.strip()]
        unknown = set(enabled) - set(STRATEGIES)
        if unknown:
            raise SystemExit(f"unknown strategy {', '.join(sorted(unknown))}; choose from {', '.join(STRATEGIES)}")
        route = route.with_strategies(enabled, args.budget, args.relevance)
    messages = _messages(args.conversation)
    count_text("warm up")  # load the tokenizer first so the timing shows the steady state, as in the running proxy
    result = compress(messages, route)
    if args.json:
        print(json.dumps(explain(result), ensure_ascii=False, indent=1))
        return 0
    if args.html:
        html = page().replace(
            "<script>\nconst STRAT", f"<script>window.CTXB_DATA = {json.dumps(explain(result), ensure_ascii=False)};</script>\n<script>\nconst STRAT", 1
        )
        Path(args.html).write_text(html, encoding="utf-8")
        print(f"wrote {args.html}")
    print(f"route {route.name}: budget {route.budget_tokens:,} tokens, strategies {', '.join(route.enabled) or 'none'}\n")
    _print_table(table(result), ["#", "message", "before", "after", "action", "why"])
    by = result.saved_by()
    print(f"\ntokens: {result.tokens_before:,} -> {result.tokens_after:,}  (saved {result.saved_pct:.1f}%, {result.elapsed_ms:.1f} ms)")
    print("saved by: " + ", ".join(f"{k} {v:,}" for k, v in by.items() if v) if any(by.values()) else "nothing needed cutting")
    if result.jev:
        j = result.jev
        if j["fallback"]:
            print(f"Jev was not used ({j['fallback']}); BM25 chose the earlier turns instead")
        else:
            print(
                f"Jev: {j['calls']} call(s), {j['input_tokens']:,} input tokens, {j['latency_ms'] / 1000:.2f} s ({j['model']}{', from cache' if j['cached'] == j['calls'] else ''})"
            )
    if result.over_budget:
        print(f"still {result.over_budget:,} tokens over budget: the pinned messages alone are larger than the budget")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .proxy import create_app

    policy = load(args.policy)
    print(f"context-budget proxy on http://{args.host}:{args.port}/v1  (page: http://{args.host}:{args.port}/)  upstream: {args.upstream or policy.upstream}")
    uvicorn.run(create_app(policy, args.upstream), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from . import bench

    conv = next(c for c in bench.load() if c["id"] == "log-01")
    path = Path(args.out)
    path.write_text(json.dumps({"messages": conv["messages"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    log = next(m["content"] for m in conv["messages"] if m["role"] == "tool" and "[payments-worker]" in m["content"])
    print(f"An agent investigating stuck payments: {len(conv['messages'])} messages, including a {log.count(chr(10)) + 1:,}-line log.")
    print(f"The question: {conv['question']}\n")
    return cmd_analyse(
        argparse.Namespace(conversation=str(path), policy=args.policy, route=None, strategies=None, budget=None, relevance=None, json=False, html=None)
    )


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench
    from .paths import EVAL

    convs = bench.build()
    manifest = EVAL / "bench-manifest.json"
    digest = bench.digest(convs)
    if args.check:
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        bad = [k for k in expected if expected[k] != digest.get(k)]
        if bad:
            print(f"{len(bad)} conversations differ from the manifest: {', '.join(bad[:10])}")
            return 1
        print(f"all {len(convs)} conversations match the manifest")
        return 0
    if args.out:
        bench.write(convs, Path(args.out))
        print(f"wrote {len(convs)} conversations to {args.out}")
    if args.write_manifest:
        manifest.write_text(json.dumps(digest, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {manifest}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import model, offline, report

    if args.which == "offline":
        from .jev import Jev, JevError

        jev = Jev(offline=args.offline or args.check, timeout_s=60.0)
        try:
            data = offline.run(jev=jev)
        except (offline.JevFallback, JevError) as e:
            print(f"stopped: a Jev reply is missing or failed ({e}). Set TYPESAFE_API_KEY to record it, or use the committed cache/jev.")
            return 2
        if args.check:
            ok = report.matches_committed("offline.json", data)
            print("offline results match eval/offline.json" if ok else "offline results differ from eval/offline.json")
            return 0 if ok else 1
        print(f"wrote {report.save_json('offline.json', data)} ({jev.live_calls} live Jev calls, {jev.cached_calls} from cache)")
        return 0
    if args.which == "report":
        print(f"wrote {report.write_results()}")
        return 0
    from .jev import Jev, JevError

    llm = LLM(offline=args.offline)
    try:
        result = model.run(llm, progress=(lambda s: None) if args.quiet else print, partial=args.partial, jev=Jev(offline=args.offline, timeout_s=60.0))
    except (offline.JevFallback, JevError) as e:
        print(f"stopped: a Jev reply is missing or failed ({e}). Set TYPESAFE_API_KEY to record it, or use the committed cache/jev.")
        return 2
    except RateLimited as e:
        print(f"stopped: {e}. {llm.live_calls} live calls were made; rerun later and the cache resumes from here.")
        return 3
    except NoModel as e:
        print(f"stopped: {e}. Set AI_API_KEY to record the missing replies, or use the committed cache.")
        return 2
    if args.check:
        ok = report.matches_committed("model.json", result)
        print(f"model results {'match' if ok else 'differ from'} eval/model.json ({llm.cached_calls} replies replayed from the cache)")
        return 0 if ok else 1
    print(f"wrote {report.save_json('model.json', result)} ({llm.live_calls} live calls, {llm.cached_calls} from cache)")
    return 0


def cmd_check_policy(args: argparse.Namespace) -> int:
    try:
        policy = load(args.policy)
    except (PolicyError, OSError) as e:
        print(f"invalid: {e}")
        return 1
    for r in policy.routes:
        models = ", ".join(r.models) or "(any)"
        print(f"route {r.name}: models {models}; budget {r.budget_tokens:,}; strategies {', '.join(r.enabled) or 'none'}; relevance {r.relevance}")
    print("policy is valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctxb", description="context-budget: shrink what AI agents send to the model.")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyse", aliases=["analyze"], help="show what would be cut and why")
    a.add_argument("conversation", help="a JSON file: a list of messages, or a chat request with 'messages'")
    a.add_argument("--policy", help="policy YAML (default: built-in defaults)")
    a.add_argument("--route", help="route name in the policy")
    a.add_argument("--strategies", help=f"comma-separated subset of {','.join(STRATEGIES)} ('' for none)")
    a.add_argument("--budget", type=int, help="token budget")
    a.add_argument("--relevance", choices=["bm25", "jev"], help="how earlier turns are ranked when over budget (jev needs TYPESAFE_API_KEY)")
    a.add_argument("--json", action="store_true", help="print the full analysis as JSON")
    a.add_argument("--html", help="also write a before/after page to this file")
    a.set_defaults(func=cmd_analyse)

    s = sub.add_parser("serve", help="run the proxy")
    s.add_argument("--policy")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--upstream", help="upstream base URL, or 'mock' for a local stand-in")
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("demo", help="analyse a sample incident conversation")
    d.add_argument("--policy", default=None)
    d.add_argument("--out", default="demo-conversation.json")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("bench", help="rebuild the benchmark conversations")
    b.add_argument("--out", help="write the conversations as JSON files to this folder")
    b.add_argument("--check", action="store_true", help="check the rebuilt conversations match eval/bench-manifest.json")
    b.add_argument("--write-manifest", action="store_true")
    b.set_defaults(func=cmd_bench)

    e = sub.add_parser("eval", help="run the evaluation")
    e.add_argument("which", choices=["offline", "model", "report"])
    e.add_argument("--offline", action="store_true", help="model: replay from the cache only")
    e.add_argument("--quiet", action="store_true")
    e.add_argument("--partial", action="store_true", help="model: use only conversations whose replies are all recorded")
    e.add_argument("--check", action="store_true", help="offline/model: compare with the committed results instead of writing them")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("check-policy", help="validate a policy file")
    c.add_argument("policy")
    c.set_defaults(func=cmd_check_policy)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
