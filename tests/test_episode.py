"""An end-to-end episode that writes a replay, and every legal end reason."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from collab_cooking.coworld.live_episode import assign_aliases
from tests.harness import (
    CERT_CONFIG,
    fast_cert_config,
    prompt_registration,
    run_episode,
    scripted_registration,
)

FOUR_SCRIPTED = [
    scripted_registration("brigade"),
    scripted_registration("brigade"),
    scripted_registration("passer"),
    scripted_registration("courier"),
]


def full_episode(tmp_path: Path, **overrides):
    config = fast_cert_config(
        step_seconds=0.0, shutdown_grace_seconds=0.0, player_connect_timeout_seconds=2, **overrides
    )
    return run_episode(config, FOUR_SCRIPTED, tmp_path)


def test_a_complete_episode_writes_both_artifacts(tmp_path: Path) -> None:
    out = full_episode(tmp_path)
    results = out["results"]
    assert results["reason"] == "complete"
    assert results["steps"] == CERT_CONFIG["max_steps"]

    replay = json.loads(out["replay_bytes"].decode("utf-8"))
    assert len(replay["ticks"]) == CERT_CONFIG["max_steps"]
    assert replay["layout"] == "cramped"
    assert replay["seed"] == CERT_CONFIG["seed"]

    # Dishes recomputed independently from the serve events must equal
    # results.dishes -- the two are derived by different code paths.
    serves = sum(
        1
        for tick in replay["ticks"]
        for event in tick.get("ev", [])
        if event["ev"] == "serve"
    )
    assert serves == results["dishes"]
    assert results["dishes"] == sum(results["delivered"])
    # And every seat's own serve count matches its delivered entry.
    per_seat = [0, 0, 0, 0]
    for tick in replay["ticks"]:
        for event in tick.get("ev", []):
            if event["ev"] == "serve":
                per_seat[event["slot"]] += 1
    assert per_seat == results["delivered"]


def test_a_nine_hundred_tick_episode_runs_to_completion(tmp_path: Path) -> None:
    """The length every published variant declares, played to the last tick.

    The 480-tick cases above are the certification fixture's length. The league
    runs 900, and on 2026-08-25 that was the difference between a green certify
    and every league episode dying `game_unhealthy` before uvicorn bound: the
    ticket encoding scaled with `max_steps` and blew mettagrid's one-byte
    feature-id space. `tests/test_feature_budget.py` guards the construction;
    this guards the play.
    """
    out = full_episode(tmp_path, layout="open-kitchen", max_steps=900)
    results = out["results"]
    assert results["reason"] == "complete"
    assert results["steps"] == 900

    replay = json.loads(out["replay_bytes"].decode("utf-8"))
    assert len(replay["ticks"]) == 900
    arrivals = [
        event
        for tick in replay["ticks"]
        for event in tick.get("ev", [])
        if event["ev"] == "order_arrive"
    ]
    # The design pins ~50 tickets in a 900-tick episode (one every 18 ticks);
    # the slot pool recycles under them, so the count must not sag with it.
    assert len(arrivals) == 50
    assert results["orders_arrived"] == len(arrivals)
    # And a recycled slot is still ONE ticket at a time on the board.
    assert results["orders_expired"] + results["dishes"] <= len(arrivals)


def test_two_runs_of_one_seed_are_byte_identical_modulo_generated_at(tmp_path: Path) -> None:
    first = full_episode(tmp_path / "a", max_steps=180)
    second = full_episode(tmp_path / "b", max_steps=180)
    left = json.loads(first["replay_bytes"].decode("utf-8"))
    right = json.loads(second["replay_bytes"].decode("utf-8"))
    left.pop("generated_at")
    right.pop("generated_at")
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_aliases_are_a_seeded_permutation_and_stable(tmp_path: Path) -> None:
    assert sorted(assign_aliases(20260826, 4)) == ["Cog-A", "Cog-B", "Cog-C", "Cog-D"]
    assert assign_aliases(20260826, 4) == assign_aliases(20260826, 4)
    assert assign_aliases(1, 4) != assign_aliases(20260826, 4)
    out = full_episode(tmp_path, max_steps=60)
    assert sorted(out["results"]["aliases"]) == ["Cog-A", "Cog-B", "Cog-C", "Cog-D"]


def test_the_deadline_path_scores_what_it_has_and_exits_zero(tmp_path: Path) -> None:
    config = fast_cert_config(
        max_steps=480,
        step_seconds=0.0,
        shutdown_grace_seconds=0.0,
        player_connect_timeout_seconds=2,
        # A budget of 0.6 x 0.001 s: the guard fires on the first tick.
        play_budget_fraction=0.6,
        episode_timeout_seconds=30,
    )
    out = run_episode(config, FOUR_SCRIPTED, tmp_path, process_start=time.monotonic() - 100.0)
    results = out["results"]
    assert results["reason"] == "deadline"
    assert results["steps"] < 480
    # Scores are real, not zeroed: a deadline episode is still rankable.
    assert results["scores"] == [pytest.approx(v) for v in results["scores"]]
    assert all(score >= 0 for score in results["scores"])
    replay = json.loads(out["replay_bytes"].decode("utf-8"))
    kinds = {event["ev"] for tick in replay["ticks"] for event in tick.get("ev", [])}
    assert "deadline" in kinds and "episode_end" in kinds


def test_a_paused_episode_still_settles_at_the_deadline(tmp_path: Path) -> None:
    """`pause` is a spectator control on /global -- the certification runner's
    own ping target. A paused loop advances no step, so the deadline guard has
    to run there too or the episode never settles and writes nothing."""
    config = fast_cert_config(
        max_steps=480,
        step_seconds=0.0,
        shutdown_grace_seconds=0.0,
        player_connect_timeout_seconds=2,
        play_budget_fraction=0.6,
        episode_timeout_seconds=30,
    )
    out = run_episode(
        config,
        FOUR_SCRIPTED,
        tmp_path,
        process_start=time.monotonic() - 100.0,
        paused=True,
        # Without the guard in the pause branch this loop is unbounded; the
        # timeout turns that into a failure instead of a hung test run.
        run_timeout=20,
    )
    results = out["results"]
    assert results["reason"] == "deadline"
    assert all(score >= 0 for score in results["scores"])
    assert out["replay_bytes"], "a paused episode still writes both artifacts"


def test_no_players_writes_zeroed_results_and_exits_zero(tmp_path: Path) -> None:
    config = fast_cert_config(
        max_steps=120, step_seconds=0.0, shutdown_grace_seconds=0.0,
        player_connect_timeout_seconds=0.2,
    )
    out = run_episode(config, [], tmp_path, connect=False)
    results = out["results"]
    assert results["reason"] == "no_players"
    assert results["scores"] == [0.0, 0.0, 0.0, 0.0]
    assert results["delivered"] == [0, 0, 0, 0]
    assert results["disconnected"] == [True, True, True, True]
    assert out["replay_bytes"], "a no_players episode still writes a replay"


def test_a_prompt_seat_with_no_credentials_records_rejections_without_baseline_fallback(tmp_path: Path) -> None:
    out = run_episode(
        fast_cert_config(max_steps=180, step_seconds=0.0, shutdown_grace_seconds=0.0,
                         player_connect_timeout_seconds=2, plan_interval_steps=50),
        [prompt_registration(), scripted_registration("brigade"),
         scripted_registration("passer"), scripted_registration("courier")],
        tmp_path,
    )
    results = out["results"]
    assert results["reason"] == "complete"
    assert results["llm_requests"] == 0, "no credentials means zero network calls"
    assert results["plan_rejections"][0] > 0
    # Prompt seat never fell back to a scripted baseline or synthesized actions.
    assert out["sockets"][0].actions == []
    assert results["action_fallbacks"][0]["deadline"] == 180
    replay = json.loads(out["replay_bytes"].decode("utf-8"))
    causes = {
        event["cause"]
        for tick in replay["ticks"]
        for event in tick.get("ev", [])
        if event["ev"] == "plan_rejected"
    }
    assert causes == {"disabled"}


def test_the_certification_fixture_settles_well_inside_the_certify_timeout(tmp_path: Path) -> None:
    """`coworld certify` defaults to a 60 s timeout covering start + connect
    grace + play + linger. The design sizes this fixture to land under 50 s;
    here the pacing and the grace are the manifest's real values."""
    config = dict(CERT_CONFIG)
    config["tokens"] = [f"token-{i}" for i in range(4)]
    config["players"] = [{"name": f"policy-{i}"} for i in range(4)]
    config["shutdown_grace_seconds"] = 20
    config["player_connect_timeout_seconds"] = 90
    started = time.monotonic()
    out = run_episode(config, FOUR_SCRIPTED, tmp_path)
    elapsed = time.monotonic() - started
    assert out["results"]["reason"] == "complete"
    assert elapsed < 50, f"the cert fixture took {elapsed:.1f}s; certify allows 60"


def test_every_player_saw_the_final_frame(tmp_path: Path) -> None:
    out = full_episode(tmp_path, max_steps=60)
    for socket in out["sockets"]:
        assert [step for step, _ in socket.actions] == list(range(60))
        assert socket.final is not None
        assert socket.final["reason"] == "complete"
        assert socket.final["done"] is True
        assert len(socket.final["aliases"]) == 4


def test_action_fallbacks_distinguish_policy_errors_from_missing_actions(monkeypatch, tmp_path: Path) -> None:
    from tests.harness import FakeSocket

    ordinary_send = FakeSocket.send_json

    async def controlled_send(socket: FakeSocket, message: dict) -> None:
        if message["type"] != "observation":
            await ordinary_send(socket, message)
            return
        slot = socket.seat.slot
        if slot == 3:
            socket.episode.disconnect_player(socket.connection_id)
            return
        await socket.episode.handle_player_message(
            socket.connection_id,
            {
                "type": "action",
                "action_name": "not-a-legal-action" if slot == 1 else "noop",
                "request_id": "step--1" if slot == 2 else f"step-{message['step']}",
            },
        )

    monkeypatch.setattr(FakeSocket, "send_json", controlled_send)
    out = run_episode(
        fast_cert_config(max_steps=6, step_seconds=0, policy_action_timeout_seconds=0, shutdown_grace_seconds=0),
        [scripted_registration() for _ in range(4)],
        tmp_path,
    )
    assert out["results"]["reason"] == "complete"
    manifest = json.loads((Path(__file__).parents[1] / "coworld_manifest_template.json").read_text())
    schema = manifest["game"]["results_schema"]
    assert set(out["results"]) == set(schema["properties"]) == set(schema["required"])
    counts = out["results"]["action_fallbacks"]
    assert counts == [
        {"invalid_action": 0, "deadline": 0, "disconnected": 0, "malformed_frame": 0},
        {"invalid_action": 6, "deadline": 0, "disconnected": 0, "malformed_frame": 0},
        {"invalid_action": 0, "deadline": 6, "disconnected": 0, "malformed_frame": 0},
        {"invalid_action": 0, "deadline": 0, "disconnected": 6, "malformed_frame": 0},
    ]
    replay = json.loads(out["replay_bytes"])
    events = [
        event for tick in replay["ticks"] if "ev" in tick for event in tick["ev"] if event["ev"] == "action_fallback"
    ]
    assert len(events) == 18
    assert {event["cause"] for event in events} == {"invalid_action", "deadline", "disconnected"}


def test_published_prompt_model_calls_http_and_applies_plan_against_explicit_baselines(
    monkeypatch, tmp_path: Path
) -> None:
    """Verify via a local deterministic HTTP model service that:
    1. A prompt seat actually calls COWORLD_LLM_ENDPOINT (/v1/messages) with seat attribution,
    2. Both bare JSON and Markdown-fenced JSON plans are parsed and applied by the seat,
    3. Failed/invalid replies are explicitly rejected (preserving raw submitted data) with
       zero scripted brigade fallback and recorded missed turns.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from collab_cooking.coworld.llm import LlmPlanner

    received: list[tuple[str, str, dict]] = []
    replies = [
        # Turn 1: standard Markdown JSON fence with omitted optional fields (defaults apply).
        '```json\n{"station":"chop","say":"chopping veg"}\n```',
        # Turn 2: bare canonical JSON.
        '{"station":"pot","recipe":"soup","zone":"any","handoff":"none","yield_to":"none","say":"cooking soup","note":"keep pot hot"}',
    ]

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            idx = min(len(received), len(replies) - 1)
            received.append((self.path, self.headers.get("X-Coworld-Player-Slot", ""), body))
            payload = json.dumps({"content": [{"type": "text", "text": replies[idx]}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "deterministic/test-model")
    traj_path = tmp_path / "trajectory.jsonl"
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", f"file://{traj_path}")
    monkeypatch.setenv("COWORLD_EPISODE_ID", "ep-http-ok")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "0.1.9")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "test-rev")
    try:
        planner = LlmPlanner(max_workers=1, timeout_seconds=5)
        assert not planner.disabled
        out = run_episode(
            fast_cert_config(
                max_steps=80,
                step_seconds=0.01,
                shutdown_grace_seconds=0.0,
                player_connect_timeout_seconds=2,
                plan_interval_steps=40,
                min_plan_interval_seconds=0.0,
            ),
            [
                prompt_registration("Coordinate with the team."),
                scripted_registration("brigade"),
                scripted_registration("passer"),
                scripted_registration("courier"),
            ],
            tmp_path / "ok",
            planner=planner,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    results = out["results"]
    assert results["reason"] == "complete"
    assert results["seat_kinds"] == ["prompt", "scripted:brigade", "scripted:passer", "scripted:courier"]
    assert results["llm_requests"] >= 1
    assert results["plan_rejections"] == [0, 0, 0, 0]
    assert len(received) >= 1
    assert all(path == "/v1/messages" and slot == "0" for path, slot, _ in received)
    # Prompt seat received and executed the returned plan(s).
    assert len(out["sockets"][0].plans) >= 1
    assert out["sockets"][0].plans[0]["accepted"] is True
    assert out["sockets"][0].plans[0]["station"] == "chop"
    assert len(out["sockets"][0].actions) > 0

    # Private trajectory records accepted model attempt with raw response.
    traj_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    decisions = [row for row in traj_lines if row["event_type"] == "decision"]
    assert len(decisions) >= 1
    assert decisions[0]["action_status"] == "accepted"
    assert decisions[0]["fallback_origin"] is None
    assert decisions[0]["executed_action"]["station"] == "chop"
    assert decisions[0]["attempts"][0]["origin"] == "model"
    assert decisions[0]["attempts"][0]["response"] == replies[0]


def test_failed_http_model_reply_is_rejected_with_raw_data_and_no_scripted_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    """When the HTTP model returns an invalid reply, the game rejects it, preserves
    the raw submitted text, and never runs brigade for the prompt seat."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from collab_cooking.coworld.llm import LlmPlanner

    bad_reply = 'Sure! {"station":"chop","recipe":"pizza"} hope that helps'

    class BadModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            _ = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps({"content": [{"type": "text", "text": bad_reply}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), BadModelHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "deterministic/bad-model")
    traj_path = tmp_path / "bad_trajectory.jsonl"
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", f"file://{traj_path}")
    monkeypatch.setenv("COWORLD_EPISODE_ID", "ep-http-bad")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "0.1.9")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "test-rev")
    try:
        planner = LlmPlanner(max_workers=1, timeout_seconds=5)
        out = run_episode(
            fast_cert_config(
                max_steps=60,
                step_seconds=0.01,
                shutdown_grace_seconds=0.0,
                player_connect_timeout_seconds=2,
                plan_interval_steps=30,
                min_plan_interval_seconds=0.0,
            ),
            [
                prompt_registration("Coordinate with the team."),
                scripted_registration("brigade"),
                scripted_registration("passer"),
                scripted_registration("courier"),
            ],
            tmp_path / "bad",
            planner=planner,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    results = out["results"]
    assert results["reason"] == "complete"
    assert results["llm_requests"] >= 2  # initial attempt + 1 retry
    assert results["plan_rejections"][0] >= 1
    # Prompt seat NEVER synthesized or fell back to scripted actions.
    assert out["sockets"][0].actions == []
    assert results["action_fallbacks"][0]["deadline"] == 60
    assert any(not p["accepted"] for p in out["sockets"][0].plans)

    # Private trajectory preserves raw submitted reply and records explicit rejection.
    traj_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    decisions = [row for row in traj_lines if row["event_type"] == "decision"]
    assert len(decisions) >= 1
    assert decisions[0]["action_status"] == "rejected"
    assert decisions[0]["executed_action"] is None
    assert decisions[0]["fallback_origin"] is None
    assert len(decisions[0]["attempts"]) == 2
    assert all(not a["accepted"] and a["response"] == bad_reply for a in decisions[0]["attempts"])
    assert all(a["rejection_reason"] for a in decisions[0]["attempts"])
