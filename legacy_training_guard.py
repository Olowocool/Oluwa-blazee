"""Fail-closed policy for legacy model pipelines with unverifiable features.

The legacy trainer/evaluator uses full-history team features, current-season
stats for historical rows, randomized synthetic market fields, and shuffled
train/test splits. The legacy retrainer also previously admitted post-result
features. Do not report its accuracy or write new artifacts until it is
rebuilt around point-in-time inputs and chronological validation.
"""

LEGACY_PIPELINE_BLOCK_REASON = (
    "Legacy training/evaluation is blocked: this pipeline does not yet prove "
    "that every feature was available before the target game and uses "
    "non-chronological evaluation or synthetic/post-result inputs. Existing "
    "metrics from this pipeline are not evidence of out-of-sample performance. "
    "No new model was trained or written. Rebuild on verified pregame features "
    "and a chronological holdout before re-enabling it."
)


def finalize_automation_status(summary):
    """Mark automation partial when any requested step was blocked or failed."""
    finalized = dict(summary)
    steps = finalized.get("steps", {})
    incomplete = [
        name for name, result in steps.items()
        if isinstance(result, dict)
        and result.get("status") in {"blocked", "error"}
    ]
    if incomplete:
        finalized["status"] = "partial"
        finalized["incomplete_steps"] = incomplete
        finalized["message"] = (
            "Automation finished with blocked or failed steps. "
            "Review the step results; do not treat this run as a complete model update."
        )
    return finalized


def blocked_legacy_pipeline_result(pipeline_name):
    """Return a uniform safe result without exposing data or writing artifacts."""
    return {
        "status": "blocked",
        "pipeline": str(pipeline_name),
        "message": LEGACY_PIPELINE_BLOCK_REASON,
        "reason_code": "unverified_point_in_time_features",
        "training_allowed": False,
        "evaluation_allowed": False,
        "model_written": False,
        "metrics_valid": False,
    }
