"""The plan reply schema, its caps, and rune-boundary truncation.

One JSON object per seat per plan turn (optionally wrapped in a single outer
Markdown `````json ... ````` fence). Extra surrounding prose is rejected;
unknown keys are ignored; omitted optional fields use documented defaults,
while explicitly provided invalid values are rejected.

Every free-text field is truncated on **rune** boundaries, never byte
boundaries -- a byte-cut multi-byte rune is exactly what makes replay bytes
fail a strict JSON parser while still rendering in a browser. In Python a
``str`` slice is already a code-point slice, so the truncator is this one
helper, applied to ``say``, ``note``, the registered prompt, policy names, the
engine ``talk`` string and every error string that can reach the replay.
"""

from __future__ import annotations

import json
import re
from typing import Any

from collab_cooking.coworld.plan_protocol import PlanControl

# Field caps, exactly as the design note's table states them.
STATION_CAP = 12
RECIPE_CAP = 6
ZONE_CAP = 6
ALIAS_CAP = 8
SAY_RUNES = 120
# A feed line that carries a `say` is "<alias>: <say>", so its cap has to be
# the say cap PLUS the prefix -- capping the whole line at SAY_RUNES silently
# dropped the last runes of every full-cap remark (r2 review R2-O4).
FEED_RUNES = SAY_RUNES + ALIAS_CAP + 2
NOTE_RUNES = 200
PROMPT_RUNES = 1200
POLICY_NAME_RUNES = 48
TALK_RUNES = 140
ERROR_RUNES = 240

RECIPES: tuple[str, ...] = ("salad", "soup", "fries", "any")
ZONES: tuple[str, ...] = ("left", "right", "pass", "any")

REJECTION_CAUSES: tuple[str, ...] = (
    "timeout",
    "parse",
    "illegal_station",
    "invalid_field",
    "rate_budget",
    "transport",
    "disabled",
)

RETRY_HINT = (
    "Your last reply was rejected. Reply with ONLY one JSON object "
    '(optionally in a ```json ... ``` fence) with "station" from LEGAL STATIONS.'
)

_OUTER_FENCE_RE = re.compile(r"^```(?:json)?[ \t]*\r?\n(.*?)\r?\n```$", re.IGNORECASE | re.DOTALL)


def truncate_runes(text: Any, cap: int) -> str:
    """Cut `text` to at most `cap` code points. Never splits a rune."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= cap:
        return text
    return text[:cap]


class PlanError(Exception):
    """A reply that cannot become a directive. Carries its rejection cause."""

    def __init__(self, cause: str, detail: str = "") -> None:
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause = cause if cause in REJECTION_CAUSES else "parse"
        self.detail = truncate_runes(detail, ERROR_RUNES)


def extract_object(text: str) -> str:
    """Extract a single JSON object, optionally wrapped in one outer Markdown fence.

    Arbitrary leading or trailing prose is rejected rather than scraped.
    """
    if not isinstance(text, str):
        raise PlanError("parse", "reply was not text")
    stripped = text.strip()
    if not stripped:
        raise PlanError("parse", "empty reply")
    if stripped.startswith("```"):
        match = _OUTER_FENCE_RE.match(stripped)
        if not match:
            raise PlanError("parse", "malformed markdown code fence")
        stripped = match.group(1).strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise PlanError("parse", "reply must be a single JSON object")
    return stripped


def _parse_optional_enum(
    raw: dict[str, Any],
    key: str,
    allowed: tuple[str, ...] | list[str],
    default: str,
    cap: int,
) -> str:
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    if not isinstance(value, str):
        raise PlanError("invalid_field", f"{key} must be a string")
    cleaned = truncate_runes(value, cap)
    if cleaned not in allowed:
        raise PlanError("invalid_field", f"{key}={cleaned or '(empty)'}")
    return cleaned


def _parse_optional_text(raw: dict[str, Any], key: str, cap: int) -> str:
    if key not in raw or raw[key] is None:
        return ""
    value = raw[key]
    if not isinstance(value, str):
        raise PlanError("invalid_field", f"{key} must be a string")
    return truncate_runes(value, cap)


def parse_plan(text: str, legal_stations: list[str], ally_aliases: list[str]) -> PlanControl:
    """Validate one reply against `LEGAL STATIONS` and the schema.

    `station` is required and must be in `legal_stations`. Omitted optional
    fields use their documented defaults; explicitly provided invalid values
    or wrong types are rejected rather than silently coerced.
    """
    span = extract_object(text)
    try:
        raw = json.loads(span)
    except Exception as exc:  # noqa: BLE001 - any decode failure is one cause
        raise PlanError("parse", f"json: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanError("parse", "reply was not a JSON object")

    raw_station = raw.get("station")
    if not isinstance(raw_station, str):
        raise PlanError("illegal_station", "station=(missing or non-string)")
    station = truncate_runes(raw_station, STATION_CAP)
    if station not in legal_stations:
        raise PlanError("illegal_station", f"station={station or '(missing)'}")

    recipe = _parse_optional_enum(raw, "recipe", RECIPES, "any", RECIPE_CAP)
    zone = _parse_optional_enum(raw, "zone", ZONES, "any", ZONE_CAP)
    handoff = _parse_optional_enum(raw, "handoff", ally_aliases, "none", ALIAS_CAP)
    yield_to = _parse_optional_enum(raw, "yield_to", ally_aliases, "none", ALIAS_CAP)
    say = _parse_optional_text(raw, "say", SAY_RUNES)
    note = _parse_optional_text(raw, "note", NOTE_RUNES)

    return PlanControl(
        station=station,
        recipe=recipe,
        zone=zone,
        handoff=handoff,
        yield_to=yield_to,
        say=say,
        note=note,
    )
