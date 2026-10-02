"""
Tables comparing the original robomimic simulation (robosuite on CPU MuJoCo) with this repo's MuJoCo Warp
POMDP, from the results of `robosuite_baseline.py` and `warp_pomdp.py`, written between the markers of
benchmarks/README.md.

    uv run python benchmarks/report.py
"""
import argparse
import json
from pathlib import Path

BEGIN, END = "<!-- results -->", "<!-- /results -->"
TASKS = {"lift": "Lift (ph)", "can": "Can (ph)", "square": "Square (mh)", "transport": "Transport (ph)",
         "tool_hang": "Tool Hang (ph)"}


def _rate(x):
    return f"{x / 1e3:.1f}k" if x >= 1e3 else f"{x:.0f}"


def _row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def tables(robosuite, warp):
    tasks = [t for t in TASKS if t in robosuite["tasks"] and t in warp["tasks"]]
    workers = robosuite["tasks"][tasks[0]]["low_dim"]["workers"]
    out = [f"Simulators: {robosuite['simulator']} ({robosuite['cpu']} CPU threads); {warp['simulator']}.", ""]
    worlds = sorted({n for t in tasks for n in warp["tasks"][t]["low_dim"]}, key=int)
    camera_worlds = sorted({n for t in tasks for n in warp["tasks"][t]["cameras"]}, key=int)

    out += ["### Throughput, env steps per second (step + observe, low-dim observations)", "",
            _row("task", "robosuite, 1 process", f"robosuite, {workers} processes", *[f"Warp, {n} worlds" for n in worlds]),
            _row(*["---"] * (3 + len(worlds)))]
    for t in tasks:
        r, w = robosuite["tasks"][t]["low_dim"], warp["tasks"][t]["low_dim"]
        out.append(_row(TASKS[t], _rate(r["single"]), _rate(r["parallel"]), *[_rate(w[n]) if n in w else "—" for n in worlds]))

    out += ["", "### Throughput with two 84x84 RGB cameras, env steps per second", "",
            _row("task", "robosuite, 1 process", f"robosuite, {workers} processes", *[f"Warp, {n} worlds" for n in camera_worlds]),
            _row(*["---"] * (3 + len(camera_worlds)))]
    for t in tasks:
        r, w = robosuite["tasks"][t]["cameras"], warp["tasks"][t]["cameras"]
        out.append(_row(TASKS[t], _rate(r["single"]), _rate(r["parallel"]), *[_rate(w[n]) if n in w else "—" for n in camera_worlds]))

    out += ["", "### Fidelity to the released demonstrations", "",
            "Open-loop replay of each demo's recorded actions from its first state, and state errors after 1 and 10 "
            "recorded actions from recorded states (median / 95th percentile over 20 states per demo).", "",
            _row("task", "simulator", "replay success", "object error, 1 step (mm)", "object error, 10 steps (mm)",
                 "arm joint error, 1 step (mrad)", "arm joint error, 10 steps (mrad)"),
            _row(*["---"] * 7)]
    for t in tasks:
        for label, results in (("robosuite", robosuite), ("Warp", warp)):
            r = results["tasks"][t]
            mm = lambda h, k, s: f"{r[f'{h}_step'][k]['median'] * s:.2f} / {r[f'{h}_step'][k]['p95'] * s:.2f}"
            out.append(_row(TASKS[t], label, f"{r['replay_success']}/{r['replay_demos']}", mm(1, "obj", 1e3),
                            mm(10, "obj", 1e3), mm(1, "arm", 1e3), mm(10, "arm", 1e3)))

    out += ["", "### Transferred demonstrations (datasets/warp)", "",
            "Every released demo re-simulated in the Warp POMDP by `robomimic.data.transfer`; deviation is the largest "
            "distance of any object from its recorded position over the demo (median / 95th percentile / max).", "",
            _row("task", "demos", "successful in Warp", "largest object deviation (mm)", "final object deviation (mm)"),
            _row(*["---"] * 5)]
    for t in tasks:
        r = warp["tasks"][t]["transferred"]
        mm = lambda k: " / ".join(f"{r[k][s] * 1e3:.1f}" for s in ("median", "p95", "max"))
        out.append(_row(TASKS[t], r["demos"], r["successful"], mm("largest_object_deviation"), mm("final_object_deviation")))
    return "\n".join(out)


def main(args):
    robosuite, warp = (json.loads(Path(p).read_text()) for p in (args.robosuite, args.warp))
    readme = Path(args.readme)
    text = readme.read_text()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    report = tables(robosuite, warp)
    readme.write_text(text[:start] + "\n" + report + "\n" + text[end:])
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robosuite", default="benchmarks/results/robosuite.json")
    parser.add_argument("--warp", default="benchmarks/results/warp.json")
    parser.add_argument("--readme", default="benchmarks/README.md")
    main(parser.parse_args())
