"""The `ctxb` command line."""

from __future__ import annotations

import argparse
import sys

from .llm import LLM, NoModel, RateLimited


def cmd_eval_model(args: argparse.Namespace) -> int:
    from .evaluation import model
    from .evaluation.report import save_json

    llm = LLM(offline=args.offline)
    try:
        result = model.run(llm)
    except RateLimited as e:
        print(f"stopped: {e}. {llm.live_calls} live calls were made; rerun later and the cache resumes from here.")
        return 3
    except NoModel as e:
        print(f"stopped: {e}. Set AI_API_KEY to record the missing replies, or use the committed cache.")
        return 2
    path = save_json("model.json", result)
    print(f"wrote {path} ({llm.live_calls} live calls, {llm.cached_calls} from cache)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctxb", description="context-budget: shrink what agents send to the model.")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="run the evaluation")
    evs = ev.add_subparsers(dest="which", required=True)
    m = evs.add_parser("model", help="answer quality with a real model (cached)")
    m.add_argument("--offline", action="store_true", help="replay from the cache only")
    m.set_defaults(func=cmd_eval_model)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
