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


def test_no_players_fails_unscored_without_results(tmp_path: Path) -> None:
    config = fast_cert_config(
        max_steps=120, step_seconds=0.0, shutdown_grace_seconds=0.0,
        player_connect_timeout_seconds=0.2,
    )
    out = run_episode(config, [], tmp_path, connect=False)
    assert out["results"] is None
    assert out["replay_bytes"] == b""
    assert out["player_failure"] is not None
    assert out["player_failure"]["kind"] == "episode_inconclusive"
    assert out["player_failure"]["failed_policy_index"] == 0


def test_a_prompt_seat_with_no_credentials_fails_unscored_before_first_tick(
    monkeypatch, tmp_path: Path
) -> None:
    traj_path = tmp_path / "disabled_trajectory.jsonl"
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", f"file://{traj_path}")
    monkeypatch.setenv("COWORLD_EPISODE_ID", "ep-disabled")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "0.1.10")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "test-rev")
    out = run_episode(
        fast_cert_config(max_steps=180, step_seconds=0.0, shutdown_grace_seconds=0.0,
                         player_connect_timeout_seconds=2, plan_interval_steps=50),
        [prompt_registration(), scripted_registration("brigade"),
         scripted_registration("passer"), scripted_registration("courier")],
        tmp_path,
    )
    assert out["results"] is None
    assert out["replay_bytes"] == b""
    assert out["player_failure"] is not None
    assert out["player_failure"]["kind"] == "episode_inconclusive"
    assert out["player_failure"]["failed_policy_index"] == 0
    assert int(out["episode"].sim.current_step) == 0
    assert out["sockets"][0].actions == []
    traj_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    ep_event = next(row for row in traj_lines if row["event_type"] == "episode")
    assert ep_event["status"] == "failed"
    assert ep_event["participant_outcomes"] is None


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


def test_explicit_noop_succeeds_while_illegal_missing_and_disconnected_fail_unscored(
    monkeypatch, tmp_path: Path
) -> None:
    from tests.harness import FakeSocket

    ordinary_send = FakeSocket.send_json

    # 1. Explicit valid noop from all seats completes normally with a real score.
    async def explicit_noop_send(socket: FakeSocket, message: dict) -> None:
        if message["type"] != "observation":
            await ordinary_send(socket, message)
            return
        await socket.episode.handle_player_message(
            socket.connection_id,
            {"type": "action", "action_name": "noop", "request_id": f"step-{message['step']}"},
        )

    monkeypatch.setattr(FakeSocket, "send_json", explicit_noop_send)
    ok = run_episode(
        fast_cert_config(max_steps=6, step_seconds=0, policy_action_timeout_seconds=0.5, shutdown_grace_seconds=0),
        [scripted_registration() for _ in range(4)],
        tmp_path / "noop",
    )
    assert ok["results"] is not None
    assert ok["results"]["reason"] == "complete"
    assert ok["player_failure"] is None
    assert all(all(v == 0 for v in row.values()) for row in ok["results"]["action_fallbacks"])

    # 2. Illegal action -> typed player_error, no score, no substitute noop.
    async def illegal_send(socket: FakeSocket, message: dict) -> None:
        if message["type"] != "observation":
            await ordinary_send(socket, message)
            return
        await socket.episode.handle_player_message(
            socket.connection_id,
            {
                "type": "action",
                "action_name": "not-a-legal-action" if socket.seat.slot == 1 else "noop",
                "request_id": f"step-{message['step']}",
            },
        )

    monkeypatch.setattr(FakeSocket, "send_json", illegal_send)
    bad_action = run_episode(
        fast_cert_config(max_steps=6, step_seconds=0, policy_action_timeout_seconds=0.5, shutdown_grace_seconds=0),
        [scripted_registration() for _ in range(4)],
        tmp_path / "illegal",
    )
    assert bad_action["results"] is None
    assert bad_action["player_failure"]["kind"] == "player_error"
    assert bad_action["player_failure"]["failed_policy_index"] == 1
    assert int(bad_action["episode"].sim.current_step) == 0

    # 3. Missing / stale action deadline -> typed player_error, no score.
    async def stale_send(socket: FakeSocket, message: dict) -> None:
        if message["type"] != "observation":
            await ordinary_send(socket, message)
            return
        await socket.episode.handle_player_message(
            socket.connection_id,
            {
                "type": "action",
                "action_name": "noop",
                "request_id": "step--1" if socket.seat.slot == 2 else f"step-{message['step']}",
            },
        )

    monkeypatch.setattr(FakeSocket, "send_json", stale_send)
    missed = run_episode(
        fast_cert_config(max_steps=6, step_seconds=0, policy_action_timeout_seconds=0, shutdown_grace_seconds=0),
        [scripted_registration() for _ in range(4)],
        tmp_path / "missed",
    )
    assert missed["results"] is None
    assert missed["player_failure"]["kind"] == "player_error"
    assert missed["player_failure"]["failed_policy_index"] == 2
    assert int(missed["episode"].sim.current_step) == 0

    # 4. Disconnected player -> typed episode_inconclusive, no score.
    async def disconnect_send(socket: FakeSocket, message: dict) -> None:
        if message["type"] != "observation":
            await ordinary_send(socket, message)
            return
        if socket.seat.slot == 3:
            socket.episode.disconnect_player(socket.connection_id)
            return
        await socket.episode.handle_player_message(
            socket.connection_id,
            {"type": "action", "action_name": "noop", "request_id": f"step-{message['step']}"},
        )

    monkeypatch.setattr(FakeSocket, "send_json", disconnect_send)
    lost = run_episode(
        fast_cert_config(max_steps=6, step_seconds=0, policy_action_timeout_seconds=0.5, shutdown_grace_seconds=0),
        [scripted_registration() for _ in range(4)],
        tmp_path / "disconnect",
    )
    assert lost["results"] is None
    assert lost["player_failure"]["kind"] == "episode_inconclusive"
    assert lost["player_failure"]["failed_policy_index"] == 3
    assert int(lost["episode"].sim.current_step) == 0


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
    # Prompt seat received and executed the returned plan(s) starting at tick 0.
    assert len(out["sockets"][0].plans) >= 1
    assert out["sockets"][0].plans[0]["accepted"] is True
    assert out["sockets"][0].plans[0]["step"] == 0
    assert out["sockets"][0].plans[0]["station"] == "chop"
    assert [step for step, _ in out["sockets"][0].actions] == list(range(80))
    assert results["action_fallbacks"][0] == {"invalid_action": 0, "deadline": 0, "disconnected": 0, "malformed_frame": 0}

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

    assert out["results"] is None
    assert out["replay_bytes"] == b""
    assert out["player_failure"] is not None
    assert out["player_failure"]["kind"] == "player_error"
    assert out["player_failure"]["failed_policy_index"] == 0
    assert out["episode"].planner.requests == 1  # single request per turn, no hidden retry
    assert int(out["episode"].sim.current_step) == 0
    assert out["sockets"][0].actions == []

    # Private trajectory preserves raw submitted reply and records failed episode with no scores.
    traj_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    decisions = [row for row in traj_lines if row["event_type"] == "decision"]
    ep_event = next(row for row in traj_lines if row["event_type"] == "episode")
    assert ep_event["status"] == "failed"
    assert ep_event["participant_outcomes"] is None
    assert len(decisions) == 1
    assert decisions[0]["action_status"] == "rejected"
    assert decisions[0]["executed_action"] is None
    assert decisions[0]["fallback_origin"] is None
    assert len(decisions[0]["attempts"]) == 1
    assert not decisions[0]["attempts"][0]["accepted"]
    assert decisions[0]["attempts"][0]["response"] == bad_reply
    assert decisions[0]["attempts"][0]["rejection_reason"]


def test_http_503_model_unavailable_fails_unscored_with_private_trace(
    monkeypatch, tmp_path: Path
) -> None:
    """When COWORLD_LLM_ENDPOINT returns HTTP 503, the episode fails immediately before
    tick 0 with episode_inconclusive, private trace, no score, and no substitute noops."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from collab_cooking.coworld.llm import LlmPlanner

    calls: list[str] = []

    class UnavailableHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            _ = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            payload = b'{"error":"service unavailable"}'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), UnavailableHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "deterministic/unavailable")
    traj_path = tmp_path / "unavailable_trajectory.jsonl"
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", f"file://{traj_path}")
    monkeypatch.setenv("COWORLD_EPISODE_ID", "ep-http-503")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "0.1.10")
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
            tmp_path / "unavail",
            planner=planner,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert len(calls) == 1, "one request per planning turn, no hidden retry"
    assert out["results"] is None
    assert out["replay_bytes"] == b""
    assert out["player_failure"] is not None
    assert out["player_failure"]["kind"] == "episode_inconclusive"
    assert out["player_failure"]["failed_policy_index"] == 0
    assert int(out["episode"].sim.current_step) == 0
    assert out["sockets"][0].actions == []

    traj_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    decisions = [row for row in traj_lines if row["event_type"] == "decision"]
    ep_event = next(row for row in traj_lines if row["event_type"] == "episode")
    assert ep_event["status"] == "failed"
    assert ep_event["participant_outcomes"] is None
    assert len(decisions) == 1
    assert decisions[0]["action_status"] == "rejected"
    assert decisions[0]["executed_action"] is None
    assert len(decisions[0]["attempts"]) == 1
    assert decisions[0]["attempts"][0]["http_status"] == 503


@pytest.mark.parametrize("scenario", ["success", "http_503", "malformed_action", "player_loss"])
def test_real_server_and_player_websocket_end_to_end(tmp_path: Path, scenario: str) -> None:
    """Exercise the complete real server and player processes over real WebSockets
    against a deterministic local HTTP model fixture for all four outcomes."""
    import asyncio
    import os
    import socket
    import subprocess
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    import websockets

    http_calls: list[dict] = []

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            http_calls.append({"path": self.path, "slot": self.headers.get("X-Coworld-Player-Slot", ""), "body": body})
            if scenario == "http_503":
                payload = b'{"error":"unavailable"}'
                self.send_response(503)
            else:
                payload = json.dumps(
                    {
                        "id": f"msg-{len(http_calls)}",
                        "model": "fixture/collab_cooking",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 12},
                        "content": [{"type": "text", "text": '{"station":"chop","recipe":"soup","say":"on it"}'}],
                    }
                ).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    http_thread = Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        game_port = sock.getsockname()[1]

    work = tmp_path / scenario
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "config.json"
    results_path = work / "results.json"
    replay_path = work / "replay.json"
    failure_path = work / "player_failure.json"
    traj_path = work / "trajectory.jsonl"

    config = fast_cert_config(
        max_steps=20,
        step_seconds=0.01,
        policy_action_timeout_seconds=1.0,
        shutdown_grace_seconds=0.0,
        player_connect_timeout_seconds=15,
        plan_interval_steps=10,
        min_plan_interval_seconds=0.0,
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    game_env = {
        **os.environ,
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(game_port),
        "COGAME_CONFIG_URI": f"file://{config_path}",
        "COGAME_RESULTS_URI": f"file://{results_path}",
        "COGAME_SAVE_REPLAY_URI": f"file://{replay_path}",
        "COGAME_PLAYER_FAILURE_URI": f"file://{failure_path}",
        "COGAME_SAVE_TRAJECTORY_URI": f"file://{traj_path}",
        "COWORLD_EPISODE_ID": f"ep-ws-{scenario}",
        "COWORLD_GAME_VERSION": "0.1.10",
        "COWORLD_SOURCE_REVISION": "test-rev",
        "COWORLD_LLM_ENDPOINT": f"http://127.0.0.1:{http_server.server_port}",
        "COWORLD_LLM_MODEL": "fixture/collab_cooking",
    }

    game_proc = subprocess.Popen(
        [sys.executable, "-m", "collab_cooking.coworld.server"],
        env=game_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    player_procs: list[subprocess.Popen[str]] = []
    try:
        # Wait for /healthz
        import urllib.request

        for _ in range(100):
            if game_proc.poll() is not None:
                raise RuntimeError(f"game exited early: {game_proc.stderr.read() if game_proc.stderr else ''}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{game_port}/healthz", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.05)

        seat_envs = [
            {"PLAYER_PROMPT": "Coordinate with the team."},
            {"PLAYER_SCRIPTED": "brigade"},
            {"PLAYER_SCRIPTED": "passer"},
            {"PLAYER_SCRIPTED": "courier"},
        ]
        spawn_slots = range(4) if scenario in ("success", "http_503") else range(3)
        for slot in spawn_slots:
            env = {
                **os.environ,
                "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{game_port}/player?slot={slot}&token=token-{slot}",
                **seat_envs[slot],
            }
            env.pop("PLAYER_PROMPT" if "PLAYER_SCRIPTED" in seat_envs[slot] else "PLAYER_SCRIPTED", None)
            player_procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "collab_cooking.coworld.player"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        if scenario in ("malformed_action", "player_loss"):
            async def bad_slot_3() -> None:
                url = f"ws://127.0.0.1:{game_port}/player?slot=3&token=token-3"
                async with websockets.connect(url) as ws:
                    _ = json.loads(await ws.recv())  # player_config
                    await ws.send(json.dumps({"type": "register", "kind": "external"}))
                    msg = json.loads(await ws.recv())
                    assert msg["type"] == "observation"
                    if scenario == "malformed_action":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "action",
                                    "action_name": "illegal_teleport",
                                    "request_id": f"step-{msg['step']}",
                                }
                            )
                        )
                        await ws.recv()
                    else:
                        await ws.close()

            asyncio.run(bad_slot_3())

        game_proc.wait(timeout=30)
        for proc in player_procs:
            proc.wait(timeout=15)
    finally:
        for proc in player_procs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        if game_proc.poll() is None:
            game_proc.terminate()
            game_proc.wait(timeout=5)
        http_server.shutdown()
        http_server.server_close()
        http_thread.join()

    assert game_proc.returncode == 0
    assert all(proc.returncode == 0 for proc in player_procs)
    assert traj_path.exists()
    traj_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    ep_event = next(row for row in traj_lines if row["event_type"] == "episode")

    if scenario == "success":
        assert results_path.exists()
        assert replay_path.exists()
        assert not failure_path.exists()
        results = json.loads(results_path.read_text(encoding="utf-8"))
        assert results["reason"] == "complete"
        assert results["steps"] == 20
        assert results["llm_requests"] >= 1
        assert len(http_calls) >= 1
        assert all(all(v == 0 for v in fb.values()) for fb in results["action_fallbacks"])
        assert ep_event["status"] == "completed"
        assert ep_event["participant_outcomes"] == results["scores"]
    elif scenario == "http_503":
        assert not results_path.exists()
        assert not replay_path.exists()
        assert failure_path.exists()
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        assert failure["kind"] == "episode_inconclusive"
        assert failure["failed_policy_index"] == 0
        assert len(http_calls) == 1
        assert ep_event["status"] == "failed"
        assert ep_event["participant_outcomes"] is None
    elif scenario == "malformed_action":
        assert not results_path.exists()
        assert not replay_path.exists()
        assert failure_path.exists()
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        assert failure["kind"] == "player_error"
        assert failure["failed_policy_index"] == 3
        assert ep_event["status"] == "failed"
        assert ep_event["participant_outcomes"] is None
    elif scenario == "player_loss":
        assert not results_path.exists()
        assert not replay_path.exists()
        assert failure_path.exists()
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        assert failure["kind"] == "episode_inconclusive"
        assert failure["failed_policy_index"] == 3
        assert ep_event["status"] == "failed"
        assert ep_event["participant_outcomes"] is None
