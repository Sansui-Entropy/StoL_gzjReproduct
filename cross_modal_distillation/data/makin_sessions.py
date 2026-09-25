"""MakinRT session roles for the official hold-out protocol.

The 18 evaluation sessions are generalization-only. The distilled student and
the fine-tuned teacher use MonkeyI_20160622_01. Teacher pretraining uses the
other Monkey I sessions that are neither evaluated nor distilled.
"""

MAKIN_EVAL_SESSIONS = [
    "MonkeyI_20160624_03",
    "MonkeyI_20160627_01",
    "MonkeyI_20160916_01",
    "MonkeyI_20160921_01",
    "MonkeyI_20160927_04",
    "MonkeyI_20160927_06",
    "MonkeyI_20160930_02",
    "MonkeyI_20160930_05",
    "MonkeyI_20161005_06",
    "MonkeyI_20161006_02",
    "MonkeyI_20161007_02",
    "MonkeyI_20161011_03",
    "MonkeyI_20161014_04",
    "MonkeyI_20161017_02",
    "MonkeyI_20161024_03",
    "MonkeyI_20161025_04",
    "MonkeyI_20161026_03",
    "MonkeyI_20161027_03",
]

# Kept so existing imports continue to mean the 18 generalization sessions.
MAKIN_LFP_SESSIONS = MAKIN_EVAL_SESSIONS

DISTILL_SESSION = "MonkeyI_20160622_01"

TEACHER_PRETRAIN_SESSIONS = [
    "MonkeyI_20160630_01",
    "MonkeyI_20160915_01",
    "MonkeyI_20161013_03",
    "MonkeyI_20161206_02",
    "MonkeyI_20161207_02",
    "MonkeyI_20161212_02",
    "MonkeyI_20161220_02",
    "MonkeyI_20170123_02",
    "MonkeyI_20170124_01",
    "MonkeyI_20170127_03",
    "MonkeyI_20170131_02",
]


def require_makin_session(session: str, allowed) -> str:
    allowed = list(allowed)
    if session not in allowed:
        raise ValueError(f"Session '{session}' is not in {allowed}")
    return session
