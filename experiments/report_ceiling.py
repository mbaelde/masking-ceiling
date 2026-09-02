"""Turn the ceiling_sweep journals into the table the decision is taken on.

    uv run python experiments/report_ceiling.py scratch/logs/ceiling/*.jsonl
    uv run python experiments/report_ceiling.py --check

One row per (probe, source, K). The column that decides is `d(m*)`: the mean
over test tracks of sdr_dmgmm - sdr_best_real, paired track by track, because
the tracks differ in difficulty by more than the margin being measured. A probe
still running is reported on the tracks it has, with their count.

Oracles come from the `method: oracle` records of the same journal, matched on
(track, source). A dmgmm record whose oracle is missing is dropped rather than
compared against another track.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# the knob each probe turns, printed so a row is readable without its filename
KNOBS = ("cov_type", "reg_covar", "energy_pct", "n_train", "max_iter")

# A test excerpt where one stem is silent poses no separation problem, and its
# ceiling is +inf, so a single such track moves a mean by hundreds of dB. The
# journal already measures it: the mixture taken as the estimate of source A has
# error B, so sdr_mixture is the A/B energy ratio in dB. Above this the quieter
# source is inaudible under the other and the excerpt is dropped, both sources at
# once so the pairing stays symmetric. Healthy MUSDB excerpts sit under 20 dB.
SILENT_DB = 60.0


def silent_tracks(records: list[dict]) -> set[tuple[str, str]]:
    """The (probe, track) pairs where one source is silent under the other."""
    return {
        (r["probe"], r["track"])
        for r in records
        if r["method"] == "oracle" and not abs(r["sdr_mixture"]) < SILENT_DB
    }


def load(paths: list[Path]) -> list[dict]:
    """The rows of every journal given, tagged with the file they came from.

    A journal is a run's stdout and holds lines that are not rows: a killed run
    leaves its last one half-written, and a script that prints anything of its
    own puts it there too. Neither is a reason to lose the whole file, so a line
    that does not parse is skipped and said out loud.
    """
    records = []
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"{path.name}: skipped a line that is not a row", file=sys.stderr)
                continue
            records.append({"probe": path.stem, **row})
    return records


def rows(records: list[dict]) -> list[dict]:
    silent = silent_tracks(records)
    records = [r for r in records if (r["probe"], r["track"]) not in silent]
    oracle = {
        (r["probe"], r["track"], r["source"]): r
        for r in records
        if r["method"] == "oracle"
    }
    groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for r in records:
        if r["method"] != "dmgmm":
            continue
        ref = oracle.get((r["probe"], r["track"], r["source"]))
        if ref is not None:
            groups[(r["probe"], r["source"], r["k"])].append((r, ref))

    out = []
    for (probe, source, k), pairs in sorted(groups.items()):
        fit, refs = [p[0] for p in pairs], [p[1] for p in pairs]
        mean = lambda xs: statistics.fmean(xs)  # noqa: E731
        out.append(
            {
                "probe": probe,
                "knobs": " ".join(f"{key}={refs[0][key]}" for key in KNOBS),
                "source": source,
                "k": k,
                "tracks": len(pairs),
                "sdr": mean([r["sdr"] for r in fit]),
                "d_best_real": mean(
                    [r["sdr"] - o["sdr_best_real"] for r, o in pairs]
                ),
                "d_irm": mean([r["sdr"] - o["sdr_irm"] for r, o in pairs]),
                "phase_deg": mean([r["phase_median_deg"] for r in fit]),
                "gain_above_one": mean([r["gain_above_one"] for r in fit]),
                "fit_min": mean([r["fit_seconds"] for r in fit]) / 60,
            }
        )
    return out


def class_rows(records: list[dict]) -> list[dict]:
    """One row per (probe, source) over the oracle records alone.

    The three subclasses of the real-mask class are read side by side here, the
    measured oracle and its analytic ceiling for each. It is a separate reading
    from `rows` because it needs no fit: a journal produced with ORACLES_ONLY has
    no dmgmm record at all and would otherwise report nothing.
    """
    silent = silent_tracks(records)
    oracle = [
        r for r in records
        if r["method"] == "oracle" and (r["probe"], r["track"]) not in silent
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in oracle:
        groups[(r["probe"], r["source"])].append(r)

    out = []
    for (probe, source), items in sorted(groups.items()):
        row = {"probe": probe, "source": source, "tracks": len(items),
               "nfft": items[0].get("nfft")}
        for name, key in (("best_real", "spectral_ceiling"),
                          ("positive", "ceiling_positive"),
                          ("clipped", "ceiling_clipped")):
            row[f"sdr_{name}"] = statistics.fmean([r[f"sdr_{name}"] for r in items])
            # a journal predating the subclass ceilings has the measured column
            # but not the analytic one, and a missing key is not a zero
            values = [r[key] for r in items if key in r]
            row[f"ceiling_{name}"] = statistics.fmean(values) if values else float("nan")
        out.append(row)
    return out


CASCADE_KEYS = ("r1", "r2", "r3", "r4", "r4_corrected")


def _finite_mean(values: list[float]) -> float:
    """Mean of the finite entries, nan if there are none.

    An excerpt whose M4 fit is saturated reports an infinite residual ratio, and
    one such excerpt would carry the mean of the whole column. Dropping it is only
    honest next to the count of what was kept, which `render_cascade` prints.
    """
    kept = [v for v in values if v == v and abs(v) != float("inf")]
    return statistics.fmean(kept) if kept else float("nan")


def cascade_rows(records: list[dict]) -> list[dict]:
    """One row per (probe, source) over the four rungs of the operator chain.

    Separate from `class_rows` because it answers the other question: not what the
    best mask on *this* frame leaves, but what the best operator of each class,
    fixed over the excerpt, leaves. The gaps are paired excerpt by excerpt like
    every other margin in this file, the excerpts differing by more than the gaps
    being measured, and the 3->4 gap is taken against the *corrected* M4 since the
    raw one is fitted with 4F parameters per bin.
    """
    silent = silent_tracks(records)
    oracle = [
        r for r in records
        if r["method"] == "oracle"
        and (r["probe"], r["track"]) not in silent
        and "cascade_r1" in r
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in oracle:
        groups[(r["probe"], r["source"])].append(r)

    out = []
    for (probe, source), items in sorted(groups.items()):
        row = {
            "probe": probe, "source": source, "tracks": len(items),
            "nfft": items[0].get("nfft"),
            "frames": _finite_mean([r["cascade_frames"] for r in items]),
            "rank": _finite_mean([r["cascade_rank"] for r in items]),
            "kept": sum(1 for r in items if abs(r["cascade_r4"]) != float("inf")),
        }
        for key in CASCADE_KEYS:
            row[key] = _finite_mean([r[f"cascade_{key}"] for r in items])
        for name, (low, high) in (("g12", ("r1", "r2")), ("g23", ("r2", "r3")),
                                  ("g34", ("r3", "r4_corrected"))):
            row[name] = _finite_mean(
                [r[f"cascade_{high}"] - r[f"cascade_{low}"] for r in items]
            )
        out.append(row)
    return out


def render_cascade(rows_: list[dict]) -> str:
    header = (
        f"{'probe':<16} {'src':>3} {'nfft':>5} {'n':>3} {'T':>6} {'rank':>5} "
        f"{'M1':>7} {'M2':>7} {'M3':>7} {'M4':>7} {'M4c':>7} | "
        f"{'1->2':>6} {'2->3':>6} {'3->4':>6}"
    )
    lines = [
        "one operator per class fixed over the excerpt, SDR of its residual, mean over excerpts",
        header, "-" * len(header),
    ]
    for r in rows_:
        lines.append(
            f"{r['probe']:<16} {r['source']:>3} {str(r['nfft']):>5} {r['tracks']:>3} "
            f"{r['frames']:>6.0f} {r['rank']:>5.0f} "
            f"{r['r1']:>7.2f} {r['r2']:>7.2f} {r['r3']:>7.2f} {r['r4']:>7.2f} "
            f"{r['r4_corrected']:>7.2f} | "
            f"{r['g12']:>+6.2f} {r['g23']:>+6.2f} {r['g34']:>+6.2f}"
        )
        if r["kept"] < r["tracks"]:
            lines.append(
                f"  {r['probe']}/{r['source']}: M4 saturated on "
                f"{r['tracks'] - r['kept']} of {r['tracks']} excerpts, dropped from "
                "the M4 columns"
            )
    return "\n".join(lines)


def render_classes(rows_: list[dict]) -> str:
    header = (
        f"{'probe':<16} {'src':>3} {'nfft':>5} {'n':>3} "
        f"{'M1':>7} {'M+':>7} {'M[0,1]':>7} | "
        f"{'M1*':>7} {'M+*':>7} {'M[0,1]*':>8}"
    )
    lines = [
        "measured oracle (left) and analytic ceiling (right), mean over excerpts",
        header, "-" * len(header),
    ]
    for r in rows_:
        lines.append(
            f"{r['probe']:<16} {r['source']:>3} {str(r['nfft']):>5} {r['tracks']:>3} "
            f"{r['sdr_best_real']:>7.2f} {r['sdr_positive']:>7.2f} {r['sdr_clipped']:>7.2f} | "
            f"{r['ceiling_best_real']:>7.2f} {r['ceiling_positive']:>7.2f} "
            f"{r['ceiling_clipped']:>8.2f}"
        )
    return "\n".join(lines)


def render(rows_: list[dict]) -> str:
    header = (
        f"{'probe':<16} {'src':>3} {'K':>3} {'n':>2} "
        f"{'SDR':>7} {'d(m*)':>7} {'d(IRM)':>7} {'phase':>6} {'g>1':>5} {'fit_min':>7}"
    )
    lines = [header, "-" * len(header)]
    knobs = {}
    for r in rows_:
        knobs[r["probe"]] = r["knobs"]
        lines.append(
            f"{r['probe']:<16} {r['source']:>3} {r['k']:>3} {r['tracks']:>2} "
            f"{r['sdr']:>7.2f} {r['d_best_real']:>+7.2f} {r['d_irm']:>+7.2f} "
            f"{r['phase_deg']:>5.1f}d {r['gain_above_one']:>5.2f} {r['fit_min']:>7.1f}"
        )
    lines += ["", "probes:"] + [f"  {p:<16} {k}" for p, k in sorted(knobs.items())]
    return "\n".join(lines)


def render_dropped(silent: set[tuple[str, str]]) -> str:
    """Never let an exclusion pass unsaid: a silent drop reads as full coverage."""
    if not silent:
        return f"no excerpt dropped (silence threshold {SILENT_DB:.0f} dB)"
    listed = "\n".join(f"  {probe:<16} {track}" for probe, track in sorted(silent))
    return f"dropped, one source silent above {SILENT_DB:.0f} dB:\n{listed}"


def demo() -> None:
    """The pairing is the whole point of this file, so it gets the check."""
    base = {
        "cov_type": "full",
        "reg_covar": 1e-6,
        "energy_pct": 0.0,
        "n_train": 25,
        "max_iter": 10,
    }
    records = [
        # easy track: oracle high; hard track: oracle low. Unpaired means would
        # mix them and hide the margin, so the two are deliberately far apart.
        {"probe": "p", "track": "easy", "source": 0, "method": "oracle",
         "sdr_best_real": 20.0, "sdr_irm": 15.0, "sdr_mixture": 3.0, **base},
        {"probe": "p", "track": "hard", "source": 0, "method": "oracle",
         "sdr_best_real": 5.0, "sdr_irm": 2.0, "sdr_mixture": -8.0, **base},
        # one stem silent: infinite ceiling, would swamp any mean it enters
        {"probe": "p", "track": "silent", "source": 0, "method": "oracle",
         "sdr_best_real": float("inf"), "sdr_irm": float("inf"),
         "sdr_mixture": float("-inf"), **base},
        {"probe": "p", "track": "silent", "source": 1, "method": "oracle",
         "sdr_best_real": 310.0, "sdr_irm": 310.0, "sdr_mixture": 311.0, **base},
        {"probe": "p", "track": "easy", "source": 0, "method": "dmgmm", "k": 8,
         "sdr": 21.0, "phase_median_deg": 10.0, "gain_above_one": 0.1,
         "fit_seconds": 60.0, **base},
        {"probe": "p", "track": "hard", "source": 0, "method": "dmgmm", "k": 8,
         "sdr": 7.0, "phase_median_deg": 20.0, "gain_above_one": 0.3,
         "fit_seconds": 120.0, **base},
        {"probe": "p", "track": "silent", "source": 0, "method": "dmgmm", "k": 8,
         "sdr": float("-inf"), "phase_median_deg": 0.0, "gain_above_one": 0.0,
         "fit_seconds": 60.0, **base},
        # no oracle for this one: it must be dropped, not compared to another track
        {"probe": "p", "track": "orphan", "source": 0, "method": "dmgmm", "k": 8,
         "sdr": 99.0, "phase_median_deg": 0.0, "gain_above_one": 0.0,
         "fit_seconds": 0.0, **base},
    ]
    (row,) = rows(records)
    assert silent_tracks(records) == {("p", "silent")}, silent_tracks(records)
    assert row["tracks"] == 2, row["tracks"]
    assert abs(row["sdr"] - 14.0) < 1e-9, row["sdr"]
    assert abs(row["d_best_real"] - 1.5) < 1e-9, row["d_best_real"]
    assert abs(row["d_irm"] - 5.5) < 1e-9, row["d_irm"]
    assert abs(row["fit_min"] - 1.5) < 1e-9, row["fit_min"]
    assert "reg_covar=1e-06" in row["knobs"], row["knobs"]

    # a journal with a self-check line at the top and a half-written last line is
    # what a real interrupted run looks like: every row between the two must survive
    journal = Path(tempfile.gettempdir()) / "report_ceiling_demo.jsonl"
    journal.write_text(
        'self-check: something a script printed\n{"track": "t", "k": 8}\n{"track": "u",\n'
    )
    assert load([journal]) == [{"probe": journal.stem, "track": "t", "k": 8}], load([journal])
    journal.unlink()

    # the class table reads the same oracle records, and the silent excerpt has to
    # be dropped there too; the analytic columns are absent from these fixtures,
    # so they must come back as nan rather than as a number
    classes = class_rows(
        [{**r, "nfft": 1024, "sdr_positive": r["sdr_best_real"],
          "sdr_clipped": r["sdr_irm"]} for r in records if r["method"] == "oracle"]
    )
    assert len(classes) == 1, classes
    assert classes[0]["tracks"] == 2, classes[0]
    assert abs(classes[0]["sdr_positive"] - 12.5) < 1e-9, classes[0]
    assert classes[0]["ceiling_positive"] != classes[0]["ceiling_positive"], classes[0]

    # the cascade table: one excerpt saturated at M4 must leave the M4 columns
    # without taking the M1..M3 columns with it, and be counted
    cascade = cascade_rows(
        [{**r, "nfft": 1024, "cascade_frames": 2584.0, "cascade_rank": 1026.0,
          "cascade_r1": 10.0, "cascade_r2": 10.5, "cascade_r3": 11.0,
          "cascade_r4": 12.0, "cascade_r4_corrected": 11.5}
         for r in records if r["method"] == "oracle" and r["track"] == "easy"]
        + [{**base, "probe": "p", "track": "loud", "source": 0, "method": "oracle",
            "sdr_mixture": 1.0, "nfft": 1024, "cascade_frames": 2584.0,
            "cascade_rank": 1026.0, "cascade_r1": 20.0, "cascade_r2": 20.5,
            "cascade_r3": 21.0, "cascade_r4": float("inf"),
            "cascade_r4_corrected": float("inf")}]
    )
    assert len(cascade) == 1, cascade
    assert cascade[0]["tracks"] == 2 and cascade[0]["kept"] == 1, cascade[0]
    assert abs(cascade[0]["r1"] - 15.0) < 1e-9, cascade[0]
    assert abs(cascade[0]["r4"] - 12.0) < 1e-9, cascade[0]
    assert abs(cascade[0]["g12"] - 0.5) < 1e-9, cascade[0]
    assert "M4 saturated on 1 of 2" in render_cascade(cascade), render_cascade(cascade)
    print("demo ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--check":
        demo()
    elif args[0] == "--cascade":
        records = load([Path(a) for a in args[1:]])
        print(render_cascade(cascade_rows(records)))
        print()
        print(render_dropped(silent_tracks(records)))
    elif args[0] == "--classes":
        records = load([Path(a) for a in args[1:]])
        print(render_classes(class_rows(records)))
        print()
        print(render_dropped(silent_tracks(records)))
    else:
        records = load([Path(a) for a in args])
        print(render(rows(records)))
        print()
        print(render_dropped(silent_tracks(records)))
