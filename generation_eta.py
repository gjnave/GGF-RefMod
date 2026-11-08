"""Whole-job estimates from completed phases, live sampling speed and past renders."""
import json
import math
from pathlib import Path
from statistics import median


def phases(request):
    plan = [("load", 15), ("encode", 5), ("sample", 60)]
    if request.get("upscale"):
        plan = [("load", 12), ("encode", 4), ("sample", 42), ("upscale", 10), ("refine", 17)]
    return plan + [("decode_video", 10), ("decode_audio", 3), ("save", 2)]


def signature(request):
    # Seed and prompt text vary between runs. References and adapters affect runtime.
    return (
        request.get("video_profile", {}).get("active", "MiniMax H3 INT8"),
        request.get("preset", "turbo"), request.get("steps"),
        request.get("width"), request.get("height"), request.get("frames"),
        bool(request.get("upscale")), request.get("up_width") if request.get("upscale") else None,
        request.get("up_height") if request.get("upscale") else None,
        request.get("refine_steps") if request.get("upscale") else None,
        tuple(Path(p).name for p in request.get("mods", [])),
        tuple(request.get("strengths", [])),
        tuple((Path(a["path"]).name, a["strength"]) for a in request.get("loras", [])),
        bool(request.get("first_frame")), bool(request.get("last_frame")),
    )


def history(request, rows):
    """Recent matching runs only. Old receipts can still supply a total duration."""
    durations = []
    stage_times = {}
    for row in reversed(rows):
        if row.get("status") != "complete":
            continue
        try:
            previous = json.loads((Path(row["job_dir"]) / "request.json").read_text(encoding="utf-8"))
            if signature(previous) != signature(request):
                continue
            records = list(row.get("clip_records", {}).values())
            if not records:
                seconds = row.get("finished", 0) - row.get("started", 0)
                records = [{"elapsed": seconds / max(1, previous.get("batch", 1))}]
            for record in records:
                elapsed = record.get("elapsed", 0)
                if math.isfinite(elapsed) and elapsed > 0:
                    durations.append(elapsed)
                    for phase, seconds in record.get("phase_seconds", {}).items():
                        if math.isfinite(seconds) and seconds > 0:
                            stage_times.setdefault(phase, []).append(seconds)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(durations) >= 5:
            break
    return {
        "total": median(durations[-8:]) if durations else None,
        "phases": {phase: median(values[-8:]) for phase, values in stage_times.items()},
    }


def initialize(request, baseline, now):
    plan=phases(request)
    scale=baseline["total"]/sum(weight for _,weight in plan) if baseline.get("total") else None
    return dict(plan=plan, baseline=baseline, phase="load", phase_started=now,
                clip=request.get("resume_from",0)+1, batch=max(1, request.get("batch", 1)),
                clip_seconds={}, samples=[], percent=100*request.get("resume_from",0)/max(1,request.get("batch",1)),
                predictions={}, scale=scale)


def finish_phase(timing, now):
    phase = timing["phase"]
    if phase:
        seconds=max(0, now - timing["phase_started"])
        timing["clip_seconds"][phase] = seconds
        timing["predictions"][phase] = seconds
        if phase=="sample" and seconds>0:
            timing["scale"]=seconds/dict(timing["plan"])[phase]
        timing["phase"] = None


def stage(timing, phase, clip, now):
    if phase is None:
        return
    if phase == timing["phase"] and clip == timing["clip"]:
        return
    finish_phase(timing, now)
    if clip != timing["clip"]:
        timing["clip_seconds"] = {}
        timing["clip"] = clip
        timing["predictions"] = {}
        total=timing["baseline"].get("total")
        if total:
            timing["scale"]=total/sum(weight for _,weight in timing["plan"])
    timing.update(phase=phase, phase_started=now, samples=[])


def progress(timing, value, total, now):
    if timing["phase"] not in ("sample", "refine") or total <= 0:
        return
    samples = timing["samples"]
    if samples and (total != samples[-1][2] or value < samples[-1][1]):
        samples.clear()
    if not samples or value > samples[-1][1]:
        samples.append((now, value, total))
        timing["samples"] = samples[-8:]


def clip_finished(timing, record, now):
    finish_phase(timing, now)
    record["phase_seconds"] = dict(timing["clip_seconds"])
    # A finished clip is an immediate timing baseline for the rest of this batch.
    timing["baseline"] = dict(total=sum(timing["clip_seconds"].values()),
                              phases=dict(timing["clip_seconds"]))


def estimate(timing, status, started, now):
    elapsed = max(0, now - started)
    if status == "complete":
        return dict(percent=100, remaining=0, elapsed=elapsed)
    if status != "running":
        return dict(percent=timing["percent"], remaining=None, elapsed=elapsed)
    plan = dict(timing["plan"])
    phase = timing["phase"]
    weights = sum(plan.values())
    baseline = timing["baseline"]
    scale = timing["scale"]
    predictions = {**baseline.get("phases", {}), **timing["predictions"]}
    samples = timing["samples"]
    # Use successive steps so initial GPU transfer time does not dominate the forecast.
    if phase in ("sample", "refine") and phase in plan and samples:
        last_time, value, total = samples[-1]
        if len(samples) > 1:
            first_time, first_value, _ = samples[0]
            rate = (last_time - first_time) / max(1, value - first_value)
        else:
            rate = (last_time - timing["phase_started"]) / max(1, value)
        if rate > 0:
            predictions[phase] = max(0, last_time - timing["phase_started"]) + rate * max(0, total - value)
            scale = predictions[phase] / plan[phase]
            timing["predictions"][phase] = predictions[phase]
            timing["scale"] = scale
    names = list(plan)
    index = names.index(phase) if phase in plan else len(names)
    spent = max(0, now - timing["phase_started"]) if phase else 0
    if scale is None:
        # Phase boundaries give a coarse overall bar while the first estimate is learning.
        done = sum(plan[name] for name in names[:index])
        percent = ((timing["clip"] - 1) + done / weights) / timing["batch"] * 100
        remaining = None
    else:
        budgets = {name: max(.1, predictions.get(name, scale * weight)) for name, weight in plan.items()}
        remaining = sum(budgets[name] for name in names[index + 1:])
        if phase:
            # If a phase overruns its estimate, extend its budget instead of counting to zero.
            if samples and phase in ("sample", "refine"):
                stalled = max(0, now - samples[-1][0])
                allowance = max(2, (budgets[phase] / max(1, samples[-1][2])))
                remaining += max(allowance, budgets[phase] - spent) + max(0, stalled - allowance)
            else:
                remaining += max(2, budgets[phase] - spent, spent * .15)
        remaining += max(0, timing["batch"] - timing["clip"]) * sum(budgets.values())
        percent = 100 * elapsed / max(.1, elapsed + remaining)
    timing["percent"] = max(timing["percent"], min(99, percent))
    return dict(percent=timing["percent"], remaining=remaining, elapsed=elapsed)


def duration(seconds):
    minutes = max(1, math.ceil(seconds / 60))
    if seconds < 60:
        return f"{max(1, math.ceil(seconds / 5) * 5)}s"
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
