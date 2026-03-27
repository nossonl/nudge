from argparse import Namespace
import io
import json

from reinforceclaw import cli, db, scheduler, trainer


def _add_feedback(conn, rating):
    return db.add_feedback(conn, "model", "prompt", "response", rating, source="test")


def test_add_adapter_keeps_one_active_and_rollback_activates_previous(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    db.add_adapter(conn, 1, str(tmp_path / "v1" / "adapter.safetensors"))
    db.add_adapter(conn, 2, str(tmp_path / "v2" / "adapter.safetensors"))

    assert db.latest_adapter(conn)["version"] == 2

    rows = {row["version"]: row["status"] for row in db.list_adapters(conn)}
    assert rows[2] == "active"
    assert rows[1] == "inactive"

    prev = db.rollback(conn)
    rows = {row["version"]: row["status"] for row in db.list_adapters(conn)}

    assert prev["version"] == 1
    assert rows[2] == "rolled_back"
    assert rows[1] == "active"


def test_record_training_round_stays_candidate_until_activated(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    fid = _add_feedback(conn, 1)

    db.record_training_round(conn, 0.1, 1, 1, str(tmp_path / "v1" / "adapter.safetensors"), metrics={"ok": True}, feedback_ids=[fid])
    candidate = db.latest_candidate(conn)
    assert candidate["version"] == 1
    assert db.latest_adapter(conn) is None
    row = conn.execute("SELECT trained FROM feedback WHERE id=?", (fid,)).fetchone()
    assert row["trained"] == 0

    db.activate_training_round(conn, 1, 0.1, 1, [fid])
    assert db.latest_adapter(conn)["version"] == 1
    row = conn.execute("SELECT trained, adapter_version FROM feedback WHERE id=?", (fid,)).fetchone()
    assert row["trained"] == 1
    assert row["adapter_version"] == 1


def test_activate_training_round_can_recover_from_candidate_metrics(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    fid = _add_feedback(conn, 1)

    db.record_training_round(
        conn,
        0.25,
        3,
        1,
        str(tmp_path / "v1" / "adapter.safetensors"),
        metrics={"ok": True},
        feedback_ids=[fid],
    )
    db.activate_training_round(conn, 1)

    ema, count = db.get_ema(conn)
    row = conn.execute("SELECT trained, adapter_version FROM feedback WHERE id=?", (fid,)).fetchone()
    assert round(ema, 3) == 0.25
    assert count == 3
    assert row["trained"] == 1
    assert row["adapter_version"] == 1


def test_openclaw_authorized_requires_secret(monkeypatch):
    from reinforceclaw.hooks import openclaw

    monkeypatch.setattr(openclaw, "_cfg", lambda: {"openclaw_secret": "secret"})
    assert openclaw._authorized({"X-ReinforceClaw-Secret": "secret"}) is True
    assert openclaw._authorized({"X-ReinforceClaw-Secret": "wrong"}) is False
    assert openclaw._authorized({}) is False


def test_feedback_collect_rating_warns_without_tty(monkeypatch):
    from reinforceclaw import feedback

    monkeypatch.setattr(feedback, "_open_tty", lambda: None)
    err = []
    monkeypatch.setattr(feedback.sys.stderr, "write", lambda msg: err.append(msg))
    monkeypatch.setattr(feedback.sys.stderr, "flush", lambda: None)
    assert feedback.collect_rating() is None
    assert err and "panel unavailable" in err[0]


def test_rollback_to_marks_newer_versions_rolled_back(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    for version in (1, 2, 3):
        db.add_adapter(conn, version, str(tmp_path / f"v{version}" / "adapter.safetensors"))

    current = db.rollback_to(conn, 2)
    rows = {row["version"]: row["status"] for row in db.list_adapters(conn)}

    assert current["version"] == 2
    assert rows[3] == "rolled_back"
    assert rows[2] == "active"
    assert rows[1] == "inactive"


def test_rollback_is_noop_without_previous_adapter(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    db.add_adapter(conn, 1, str(tmp_path / "v1" / "adapter.safetensors"))

    assert db.rollback(conn) is None
    assert db.latest_adapter(conn)["version"] == 1
    assert db.rollback_to(conn, 99) is None
    assert db.latest_adapter(conn)["version"] == 1


def test_parse_time_rejects_invalid_values():
    assert scheduler._parse_time("03:00") == (3, 0)
    assert scheduler._attempt_times("03:00", 180) == [(3, 0), (4, 0), (5, 0)]

    for bad in ("3", "ab:cd", "99:99", "12:99"):
        try:
            scheduler._parse_time(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should fail")


def test_maybe_train_only_queues_when_auto(monkeypatch, tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    _add_feedback(conn, 1)
    _add_feedback(conn, -1)
    queued = []

    monkeypatch.setattr("reinforceclaw.hooks._common.queue_training", lambda: queued.append(True))

    cli._maybe_train({"train_schedule": "03:00", "batch_min": 2}, conn)
    assert queued == []

    cli._maybe_train({"train_schedule": "auto", "batch_min": 2}, conn)
    assert queued == [True]


def test_codex_stop_stores_pending_even_when_panel_disabled(monkeypatch, tmp_path):
    from reinforceclaw.hooks import codex

    db_path = tmp_path / "reinforceclaw.db"
    real_connect = db.connect
    monkeypatch.setattr(codex, "load_config", lambda: {"model": "m", "panel_enabled": False})
    monkeypatch.setattr(codex.db, "connect", lambda: real_connect(db_path))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"last_assistant_message": "hello"})))
    codex.handle_stop()

    conn = real_connect(db_path)
    pending = db.latest_pending(conn, source="codex")
    assert pending is not None
    assert pending["response"] == "hello"


def test_codex_stop_spawns_detached_panel(monkeypatch, tmp_path):
    from reinforceclaw.hooks import codex

    db_path = tmp_path / "reinforceclaw.db"
    real_connect = db.connect
    spawned = {}
    monkeypatch.setattr(codex, "load_config", lambda: {"model": "m", "panel_enabled": True})
    monkeypatch.setattr(codex.db, "connect", lambda: real_connect(db_path))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"last_assistant_message": "hello"})))

    def fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        class Proc: ...
        return Proc()

    monkeypatch.setattr(codex.subprocess, "Popen", fake_popen)
    codex.handle_stop()

    assert spawned["argv"][2] == "panel"
    assert spawned["kwargs"]["start_new_session"] is True
    conn = real_connect(db_path)
    assert db.latest_pending(conn, source="codex") is not None


def test_claude_prompt_only_intercepts_exact_commands(monkeypatch, tmp_path):
    from reinforceclaw.hooks import claude_code

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": {"content": "/good job", "type": "user"}})))
    try:
        claude_code.handle_prompt()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("non-command prompt should exit through SystemExit(0)")


def test_codex_prompt_rates_latest_pending(monkeypatch, tmp_path):
    from reinforceclaw.hooks import codex

    db_path = tmp_path / "reinforceclaw.db"
    real_connect = db.connect
    conn = real_connect(db_path)
    fid = db.add_feedback(conn, "m", "(codex)", "resp", 0, source="codex")
    conn.close()

    monkeypatch.setattr(codex.db, "connect", lambda: real_connect(db_path))
    monkeypatch.setattr(codex, "load_config", lambda: {"model": "m", "panel_enabled": True, "train_schedule": "manual"})
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": {"content": "/good", "type": "user"}})))
    monkeypatch.setattr(codex, "maybe_train", lambda conn, cfg: None)
    out = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: out.append(args[0] if args else ""))

    codex.handle_prompt()

    conn = real_connect(db_path)
    row = conn.execute("SELECT rating FROM feedback WHERE id = ?", (fid,)).fetchone()
    assert row[0] == 1
    assert out and '"result": "block"' in out[0]


def test_codex_prompt_only_intercepts_exact_commands(monkeypatch):
    from reinforceclaw.hooks import codex

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": {"content": "/bad plan", "type": "user"}})))
    try:
        codex.handle_prompt()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("non-command prompt should exit through SystemExit(0)")


def test_install_launchd_unloads_before_load(monkeypatch, tmp_path):
    calls = []
    plist = tmp_path / "com.reinforceclaw.train.plist"

    def fake_run(argv, capture_output=True):
        calls.append(tuple(argv))
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(scheduler, "PLIST_PATH", plist)
    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    assert scheduler._install_launchd([(3, 0)]) is True
    assert calls[:2] == [
        ("launchctl", "unload", str(plist)),
        ("launchctl", "load", str(plist)),
    ]


def test_cmd_train_background_skips_confirmation(monkeypatch, tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    _add_feedback(conn, 1)
    _add_feedback(conn, -1)

    monkeypatch.setattr(cli, "_load_model_cfg", lambda: {"batch_min": 2, "model": "m", "server": "ollama"})
    monkeypatch.setattr(cli.db, "connect", lambda: conn)
    monkeypatch.setattr(cli, "_trainable_untrained", lambda _conn: 2)
    monkeypatch.setattr(cli.trainer, "train_result", lambda cfg, _conn: {"status": "trained", "avg_loss": 0.1, "batch_size": 2, "ema_mean": 0.0})
    monkeypatch.setattr(cli, "_swap_latest", lambda cfg, _conn: True)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    def fail_confirm(*args, **kwargs):
        raise AssertionError("Confirm.ask should not run in background mode")

    monkeypatch.setattr(cli.Confirm, "ask", fail_confirm)

    cli.cmd_train(Namespace(background=True))


def test_cmd_train_background_requeues_memory_pressure(monkeypatch, tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    _add_feedback(conn, 1)
    _add_feedback(conn, -1)
    queued = []

    monkeypatch.setattr(cli, "_load_model_cfg", lambda: {"batch_min": 2, "model": "m", "server": "ollama", "train_schedule": "auto"})
    monkeypatch.setattr(cli.db, "connect", lambda: conn)
    monkeypatch.setattr(cli, "_trainable_untrained", lambda _conn: 2)
    monkeypatch.setattr(cli.trainer, "train_result", lambda cfg, _conn: {"status": "skipped", "reason": "memory_pressure"})
    monkeypatch.setattr("reinforceclaw.hooks._common.queue_training", lambda delay_seconds=0: queued.append(delay_seconds))
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    cli.cmd_train(Namespace(background=True))
    assert len(queued) == 1
    assert queued[0] >= 900


def test_cmd_train_rejects_candidate_when_gate_fails(monkeypatch, tmp_path):
    db_path = tmp_path / "reinforceclaw.db"
    real_connect = db.connect
    conn = db.connect(db_path)
    _add_feedback(conn, 1)
    _add_feedback(conn, -1)

    monkeypatch.setattr(cli, "_load_model_cfg", lambda: {"batch_min": 2, "model": "m", "server": "ollama"})
    monkeypatch.setattr(cli.db, "connect", lambda: conn)
    monkeypatch.setattr(cli, "_trainable_untrained", lambda _conn: 2)
    monkeypatch.setattr(cli.trainer, "train_result", lambda cfg, _conn: {
        "status": "trained", "avg_loss": 0.1, "batch_size": 2, "ema_mean": 0.0, "ema_count": 2,
        "version": 1, "path": str(tmp_path / "v1" / "adapter.safetensors"), "feedback_ids": [1, 2],
    })
    monkeypatch.setattr(cli.trainer, "publish_gate", lambda cfg, path: {"ok": False, "reason": "score_drop", "base_score": 6, "candidate_score": 5})
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)
    db.record_training_round(conn, 0.0, 0, 1, str(tmp_path / "v1" / "adapter.safetensors"))

    cli.cmd_train(Namespace(background=False))
    rows = {row["version"]: row["status"] for row in db.list_adapters(real_connect(db_path))}
    assert rows[1] == "rejected"


def test_background_waits_outside_schedule_window(monkeypatch):
    class Hardware:
        unified_memory = False
        available_memory_bytes = 8 * 1024**3
        total_memory_bytes = 12 * 1024**3
        system_available_memory_bytes = 8 * 1024**3

    class Backend:
        name = "cuda"

    monkeypatch.setattr(trainer, "_scheduled_window_open", lambda cfg: False)
    monkeypatch.setattr(trainer, "_load_ratio", lambda: 0.0)

    assert trainer._background_should_wait({"_background": True, "train_schedule": "03:00"}, Backend(), Hardware())


def test_auto_background_waits_when_cuda_idle_telemetry_missing(monkeypatch):
    class Hardware:
        unified_memory = False
        available_memory_bytes = 8 * 1024**3
        total_memory_bytes = 12 * 1024**3
        system_available_memory_bytes = 8 * 1024**3

    class Backend:
        name = "cuda"

    monkeypatch.setattr(trainer, "_scheduled_window_open", lambda cfg: True)
    monkeypatch.setattr(trainer, "_load_ratio", lambda: 0.0)
    monkeypatch.setattr(trainer, "_cuda_activity", lambda backend: None)

    assert trainer._background_should_wait({"_background": True, "train_schedule": "auto"}, Backend(), Hardware())


def test_scheduled_background_ignores_missing_idle_telemetry(monkeypatch):
    class Hardware:
        unified_memory = False
        available_memory_bytes = 8 * 1024**3
        total_memory_bytes = 12 * 1024**3
        system_available_memory_bytes = 8 * 1024**3

    class Backend:
        name = "cuda"

    monkeypatch.setattr(trainer, "_scheduled_window_open", lambda cfg: True)
    monkeypatch.setattr(trainer, "_load_ratio", lambda: 0.0)
    monkeypatch.setattr(trainer, "_cuda_activity", lambda backend: None)

    assert not trainer._background_should_wait({"_background": True, "train_schedule": "03:00"}, Backend(), Hardware())


def test_smoke_status_reports_below_threshold(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    status = trainer.smoke_status({"train_schedule": "03:00", "batch_min": 2}, conn)
    assert status["would_train"] is False
    assert status["reason"] == "below_threshold"


def test_train_result_reports_below_threshold(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    old_acquire = trainer._acquire_lock
    old_release = trainer._release_lock
    trainer._acquire_lock = lambda: 1
    trainer._release_lock = lambda fd: None
    try:
        result = trainer.train_result({"model": "m", "batch_min": 2}, conn)
    finally:
        trainer._acquire_lock = old_acquire
        trainer._release_lock = old_release
    assert result["status"] == "skipped"
    assert result["reason"] == "below_threshold"


def test_train_result_retries_mlx_insufficient_budget_in_fresh_process(monkeypatch, tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    _add_feedback(conn, 1)
    _add_feedback(conn, -1)

    class Hardware:
        unified_memory = True
        available_memory_bytes = 16 * 1024**3
        total_memory_bytes = 48 * 1024**3
        system_available_memory_bytes = 16 * 1024**3

    class Backend:
        name = "mlx"

        def hardware(self):
            return Hardware()

    monkeypatch.setattr(trainer, "_select_backend", lambda cfg: Backend())
    monkeypatch.setattr(trainer, "_settle_backend_hardware", lambda backend, hardware=None, rounds=3, delay=1.0: hardware or backend.hardware())
    monkeypatch.setattr(trainer, "_has_minimum_headroom", lambda hardware: True)
    monkeypatch.setattr(trainer, "_background_block_reason", lambda config, backend, hardware: None)
    monkeypatch.setattr(trainer, "_attempt_train", lambda config, conn, backend, hardware, attempt: {"status": "skipped", "reason": "insufficient_budget", "backend": "mlx"})
    monkeypatch.setattr(trainer, "_fresh_process_train_retry", lambda config, conn: {"status": "trained", "retry_mode": "fresh_process"})
    monkeypatch.setattr(trainer, "_acquire_lock", lambda: 1)
    monkeypatch.setattr(trainer, "_release_lock", lambda fd: None)

    result = trainer.train_result({"model": "m", "batch_min": 2}, conn)
    assert result["status"] == "trained"
    assert result["retry_mode"] == "fresh_process"


def test_training_budget_subtracts_loaded_model_on_unified_memory():
    class Hardware:
        unified_memory = True
        total_memory_bytes = 48 * 1024**3
        available_memory_bytes = 30 * 1024**3
        recommended_working_set_bytes = 38 * 1024**3

    budget = trainer._training_budget_bytes(Hardware(), trainer.MAC_OS_RESERVE, 18 * 1024**3)
    assert budget == 12 * 1024**3


def test_settle_backend_hardware_prefers_recovered_mlx_headroom(monkeypatch):
    class Hardware:
        unified_memory = True
        total_memory_bytes = 48 * 1024**3
        system_available_memory_bytes = 0
        recommended_working_set_bytes = None

        def __init__(self, available):
            self.available_memory_bytes = available

    class Backend:
        name = "mlx"

        def __init__(self):
            self.values = iter([20, 26, 26])

        def hardware(self):
            return Hardware(next(self.values) * 1024**3)

        def clear_all(self):
            return None

        def synchronize(self):
            return None

    monkeypatch.setattr(trainer.time, "sleep", lambda _: None)
    settled = trainer._settle_backend_hardware(Backend(), Hardware(20 * 1024**3), rounds=3, delay=0)
    assert settled.available_memory_bytes == 26 * 1024**3


def test_settle_backend_hardware_noops_for_non_mlx():
    class Hardware:
        unified_memory = False
        available_memory_bytes = 8 * 1024**3

    class Backend:
        name = "cuda"

        def hardware(self):
            raise AssertionError("should not be called")

    hardware = Hardware()
    assert trainer._settle_backend_hardware(Backend(), hardware) is hardware


def test_degrade_plan_keeps_memory_limit_above_model_floor():
    plan = trainer.TrainingPlan(
        effective_batch_size=8,
        grad_accum=2,
        steps=3,
        memory_limit_bytes=22 * 1024**3,
        training_budget_bytes=4 * 1024**3,
        resident_model_bytes=18 * 1024**3,
        aggressive_checkpointing=False,
        busy=False,
    )
    degraded = trainer._degrade_plan(plan)
    assert degraded is not None
    assert degraded.memory_limit_bytes == 20 * 1024**3


def test_build_batch_uses_replay_ratio(tmp_path):
    conn = db.connect(tmp_path / "reinforceclaw.db")
    fresh_ids = [_add_feedback(conn, 1) for _ in range(8)]
    trained_ids = [_add_feedback(conn, -1) for _ in range(8)]
    db.mark_trained(conn, trained_ids, 1)

    batch, picked_fresh_ids, fresh = trainer._build_batch(conn, batch_size=8, replay_ratio=0.25)

    assert len(batch) == 8
    assert len(fresh) == 6
    assert picked_fresh_ids == fresh_ids[:6]
    assert sum(1 for item in batch if item["trained"] == 1) == 2


def test_tighten_small_batch_cfg_preserves_requested_steps():
    cfg = {"steps": 5, "grad_accum": 4, "max_passes": 1.0}
    tightened = trainer._tighten_small_batch_cfg(cfg, batch_size=2)
    assert tightened["steps"] == 5
    assert tightened["grad_accum"] == 1


def test_plan_strategy_does_not_shrink_for_cpu_busyness_only(monkeypatch):
    class Hardware:
        unified_memory = True
        total_memory_bytes = 48 * 1024**3
        available_memory_bytes = 30 * 1024**3
        recommended_working_set_bytes = 38 * 1024**3

    monkeypatch.setattr(trainer, "_load_ratio", lambda: 99.0)
    plan = trainer._plan_strategy(
        {"batch_size": 4, "grad_accum": 1, "steps": 4, "idle_load_threshold": 1.0},
        Hardware(),
        18 * 1024**3,
    )
    assert plan is not None
    assert plan.effective_batch_size == 4
    assert plan.steps == 4


def test_plan_strategy_keeps_post_load_limit_above_resident_model_floor(monkeypatch):
    class Hardware:
        unified_memory = True
        total_memory_bytes = 48 * 1024**3
        available_memory_bytes = 32 * 1024**3
        recommended_working_set_bytes = 38 * 1024**3

    monkeypatch.setattr(trainer, "_load_ratio", lambda: 0.0)
    model_bytes = 18 * 1024**3
    plan = trainer._plan_strategy(
        {"batch_size": 4, "grad_accum": 1, "steps": 4, "idle_load_threshold": 1.0},
        Hardware(),
        model_bytes,
    )
    assert plan is not None
    assert plan.memory_limit_bytes >= model_bytes + trainer.MIN_TRAINING_BUDGET


def test_stage_adapter_dir_skips_existing_version(tmp_path):
    root = tmp_path / "adapters"
    (root / "v1").mkdir(parents=True)

    version, temp_dir, final_dir = trainer._stage_adapter_dir(1, {"adapter_root": str(root)})

    assert version == 2
    assert temp_dir.exists()
    assert final_dir == root / "v2"


def test_tokenize_mlx_uses_structured_messages_from_context():
    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
            self.calls.append((msgs, add_generation_prompt))
            return list(range(1, len(msgs) + (1 if add_generation_prompt else 2)))

    tokenizer = DummyTokenizer()
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Follow-up"},
    ]
    item = {
        "id": 1,
        "prompt": "WRONG PROMPT",
        "response": "Final answer",
        "rating": 1,
        "context": json.dumps({"messages": messages}),
    }

    old_mx = trainer.mx
    trainer.mx = type("DummyMX", (), {"array": staticmethod(lambda value: value)})
    try:
        out = trainer._tokenize_mlx(tokenizer, item)
    finally:
        trainer.mx = old_mx

    assert tokenizer.calls[0] == (messages, True)
    assert tokenizer.calls[1] == (messages + [{"role": "assistant", "content": "Final answer"}], False)
    assert out["response_start"] >= 1


def test_apply_chat_template_disables_thinking_when_supported():
    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, msgs, **kwargs):
            self.calls.append((msgs, kwargs))
            return [1, 2, 3]

    tokenizer = DummyTokenizer()
    trainer._apply_chat_template(tokenizer, [{"role": "user", "content": "Hi"}], add_generation_prompt=True)

    assert tokenizer.calls[0][1]["enable_thinking"] is False
    assert tokenizer.calls[0][1]["add_generation_prompt"] is True


def test_apply_chat_template_falls_back_when_tokenizer_rejects_enable_thinking():
    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
            self.calls.append((msgs, add_generation_prompt, tokenize))
            return [1, 2, 3]

    tokenizer = DummyTokenizer()
    out = trainer._apply_chat_template(tokenizer, [{"role": "user", "content": "Hi"}], add_generation_prompt=True)

    assert out == [1, 2, 3]
    assert tokenizer.calls == [([{"role": "user", "content": "Hi"}], True, True)]


def test_behavior_logprobs_reads_context_for_mispo():
    old_mx = trainer.mx
    trainer.mx = type("DummyMX", (), {"array": staticmethod(lambda value: value)})
    try:
        item = {
            "context": json.dumps({"behavior_logprobs": [-0.1, -0.2, -0.3]}),
            "input_ids": [1, 2, 3],
        }
        assert trainer._behavior_logprobs(item, 3, "mlx") == [-0.1, -0.2, -0.3]
        assert trainer._behavior_logprobs(item, 2, "mlx") is None
    finally:
        trainer.mx = old_mx


def test_cmd_collect_falls_back_to_local_when_ollama_unreachable(monkeypatch, tmp_path):
    db_path = tmp_path / "reinforceclaw.db"
    real_connect = db.connect
    monkeypatch.setattr(cli, "_load_model_cfg", lambda: {"model": "m", "server": "ollama", "lora_rank": 8})
    monkeypatch.setattr(cli.db, "connect", lambda: real_connect(db_path))
    monkeypatch.setattr(cli.collect, "sample_prompts", lambda count, topics: [{"topic": "code", "prompt": "Prompt"}])
    monkeypatch.setattr(cli.collect, "ollama_available", lambda base_url: False)
    monkeypatch.setattr(cli.collect, "chat", lambda **kwargs: "Response")
    monkeypatch.setattr(cli.collect, "flatten_transcript", lambda messages: ("Prompt", "Response"))
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: "g")
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    cli.cmd_collect(Namespace(
        topics="code",
        count=1,
        file=None,
        prompt_model=None,
        prompt_server="openai",
        prompt_base_url=None,
        prompt_api_key=None,
        judge_model=None,
        judge_server="openai",
        judge_base_url=None,
        judge_api_key=None,
        turns=1,
    ))

    row = real_connect(db_path).execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 1").fetchone()
    ctx = json.loads(row["context"])
    assert row["rating"] == 1
    assert ctx["served_model"] == "m"
    assert ctx["messages"] == [{"role": "user", "content": "Prompt"}]


def test_cmd_collect_ignores_single_judge_failure_and_continues(monkeypatch, tmp_path):
    db_path = tmp_path / "reinforceclaw.db"
    real_connect = db.connect
    monkeypatch.setattr(cli, "_load_model_cfg", lambda: {"model": "m", "server": "ollama", "lora_rank": 8})
    monkeypatch.setattr(cli.db, "connect", lambda: real_connect(db_path))
    monkeypatch.setattr(cli.collect, "sample_prompts", lambda count, topics: [
        {"topic": "code", "prompt": "Prompt 1"},
        {"topic": "math", "prompt": "Prompt 2"},
    ])
    monkeypatch.setattr(cli.collect, "ollama_available", lambda base_url: False)
    monkeypatch.setattr(cli.collect, "chat", lambda **kwargs: "Response")
    monkeypatch.setattr(cli.collect, "flatten_transcript", lambda messages: (messages[0]["content"], "Response"))
    outcomes = iter([RuntimeError("judge down"), "bad"])
    def fake_judge_response(**kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(cli.collect, "judge_response", fake_judge_response)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    cli.cmd_collect(Namespace(
        topics="code,math",
        count=2,
        file=None,
        prompt_model=None,
        prompt_server="openai",
        prompt_base_url=None,
        prompt_api_key=None,
        judge_model="judge",
        judge_server="openai",
        judge_base_url="http://judge/v1",
        judge_api_key=None,
        turns=1,
    ))

    rows = real_connect(db_path).execute("SELECT prompt, rating FROM feedback ORDER BY id").fetchall()
    assert [(row["prompt"], row["rating"]) for row in rows] == [("Prompt 2", -1)]
