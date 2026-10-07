#!/usr/bin/env python3
"""Static and isolated checks of run.sh / stop.sh / the gate tools. No Docker, no GPU, stdlib only.

    python3 -m unittest discover -s tests -q
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# bench_decode.py is byte-identical to the sibling vLLM recipe's frozen ruler
# (sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark, evidence/fp8-default/fp8/gate/harness.sha256).
BENCH_DECODE_SHA256 = "6a9c64bd821fa16574dd7b471ac7bba81ea95827a48a5b7a201daeb4f76f7c36"
TF_SHA = "cb2ebf0540f42604e2759b2ddef497861e928248"
BASE_DIGEST = "sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
OVERRIDES = ("PROFILE", "IMAGE", "TF_SHA", "TF_PATCH", "TP", "CONTEXT", "KV_DTYPE", "MTP_DRAFTS", "MTP_CONFIDENCE", "PARALLEL",
             "THINKING", "MAX_TOKENS", "MEMORY_RESERVE_GIB", "EXTRA_ARGS", "EXTRA_ENV", "BENCH_ONLY", "HCA",
             "VISION", "VISION_URLS", "VISION_MAX_IMAGES")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _run_sh(**extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in OVERRIDES:
        env.pop(name, None)
    env.update(extra)
    env["VALIDATE_ONLY"] = "1"
    env.setdefault("ENV_CLUSTER", "/dev/null")   # the shipped defaults, not this cluster's .env.cluster
    return subprocess.run([str(ROOT / "run.sh")], check=False, capture_output=True, text=True, cwd=str(ROOT), env=env)


def _func_src(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", src, re.M | re.S)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(0)


def _func_body(src: str, name: str) -> str:
    return _func_src(src, name).split("{", 1)[1]


class GuardTests(unittest.TestCase):
    """Each refusal fires with its reason before VALIDATE_ONLY exits."""

    def refused(self, needle: str, **env: str) -> None:
        proc = _run_sh(**env)
        self.assertNotEqual(proc.returncode, 0, f"{env} was accepted")
        self.assertIn(needle, proc.stderr, env)

    def accepted(self, **env: str) -> subprocess.CompletedProcess[str]:
        proc = _run_sh(**env)
        self.assertEqual(proc.returncode, 0, f"{env}: {proc.stderr}")
        return proc

    def test_defaults_pass(self) -> None:
        out = self.accepted().stdout
        self.assertIn("validate-only", out)
        self.assertIn(f"tf={TF_SHA}", out)
        self.assertIn("tp=2 ctx=262144 kv=int8", out)

    def test_integers_are_decimal(self) -> None:
        self.refused("not a positive decimal integer", MAX_TOKENS="010")
        self.refused("not a positive decimal integer", CONTEXT="8x")
        self.refused("not a positive decimal integer", PARALLEL="0")
        self.refused("not a non-negative decimal integer", MTP_DRAFTS="06")
        self.refused("must be 0 or 1", THINKING="yes")
        self.refused("must be a decimal in [0, 1]", MTP_CONFIDENCE="1.5")
        self.refused("must be a decimal in [0, 1]", MTP_CONFIDENCE=".7")
        self.refused("not empty or a positive decimal", MEMORY_RESERVE_GIB="-4")
        self.accepted(MTP_CONFIDENCE="1", MTP_DRAFTS="0", MEMORY_RESERVE_GIB="16")

    def test_topology_and_window(self) -> None:
        self.refused("one or two ranks only", TP="4")
        self.refused("exceeds the trained window", CONTEXT="262145")
        self.assertIn("profile=long", self.accepted(PROFILE="long").stdout)
        self.assertIn("ctx=1048576", self.accepted(PROFILE="long").stdout)
        self.assertIn("ctx=524288", self.accepted(PROFILE="long", CONTEXT="524288").stdout)
        self.refused("exceeds the long profile's window", PROFILE="long", CONTEXT="1048577")
        self.refused("fits the trained window", PROFILE="long", CONTEXT="262144")
        self.refused("exceeds the engine cap", MTP_DRAFTS="16")
        self.refused("must be bf16, int8 or int4", KV_DTYPE="fp8")
        self.accepted(TP="1", CONTEXT="65536", KV_DTYPE="int8")

    def test_parallel_is_served_at_tp2(self) -> None:
        # TensorFold 0.6.4+ serves --parallel N on two ranks; no second image or patch is needed.
        self.assertIn("parallel=4", self.accepted(PARALLEL="4").stdout)
        self.accepted(PARALLEL="4", TP="1")
        self.refused("not a positive decimal integer", PARALLEL="0")

    def test_concurrent_profile_fills_unset_variables_only(self) -> None:
        out = self.accepted(PROFILE="concurrent").stdout
        for want in ("profile=concurrent", "image=tf-qwen38-flashnext-exl3:0.6.6", "patch=patches/flashnext-exl3-0.6.6.patch",
                     "parallel=16", "mtp=15@"):
            self.assertIn(want, out)
        out = self.accepted(PROFILE="concurrent", PARALLEL="4", MTP_DRAFTS="6").stdout
        self.assertIn("parallel=4", out)
        self.assertIn("mtp=6@", out)
        self.assertIn("profile=serial", self.accepted().stdout)
        proc = _run_sh(PROFILE="fast")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PROFILE=fast must be serial, concurrent or long", proc.stderr)

    def test_worker_copy_takes_the_forwarded_patch_sha(self) -> None:
        # The worker runs a /tmp copy of run.sh with no docker/patches/ next to it.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "run.sh"
            copy.write_text(_read("run.sh"))
            copy.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in OVERRIDES}
            env.update(VALIDATE_ONLY="1", ENV_CLUSTER="/dev/null", TF_PATCH="patches/flashnext-exl3-0.6.6.patch", PARALLEL="8")
            proc = subprocess.run([str(copy)], capture_output=True, text=True, env=env, check=False)
            self.assertNotEqual(proc.returncode, 0)
            pin = re.search(r"\[patches/flashnext-exl3-0.6.6.patch\]=([0-9a-f]{64})", _read("run.sh")).group(1)
            env.update(ROLE="worker", TF_PATCH_SHA=pin)
            proc = subprocess.run([str(copy)], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertRegex(_read("run.sh"), r"FORWARD_VARS=\([^)]*\bTF_PATCH_SHA\b")

    def test_shipped_patch_matches_its_pin(self) -> None:
        run = _read("run.sh")
        pins = dict(re.findall(r"^\s+\[(patches/[^\]]+)\]=([0-9a-f]{64})$", run, re.M))
        self.assertIn("patches/flashnext-exl3-0.6.6.patch", pins)
        for rel, sha in pins.items():
            self.assertEqual(hashlib.sha256((ROOT / "docker" / rel).read_bytes()).hexdigest(), sha, rel)
            self.assertIn(sha, _read("recipe.yaml"), rel)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "run.sh"
            copy.write_text(run)
            copy.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in OVERRIDES}
            env.update(VALIDATE_ONLY="1", ENV_CLUSTER="/dev/null", ROLE="worker", TF_PATCH="patches/flashnext-exl3-0.6.6.patch", TF_PATCH_SHA="b" * 64)
            proc = subprocess.run([str(copy)], capture_output=True, text=True, env=env, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not the pinned", proc.stderr)

    def test_vision_on_either_profile(self) -> None:
        # patch parts 4 and 5: images on two ranks, on the concurrent decoder and on the serial engine
        self.accepted(VISION="1")
        self.accepted(VISION="1", PROFILE="concurrent")
        self.refused("must be 0 or 1", VISION="2")
        self.refused("must be 0 or 1", VISION_URLS="yes")
        self.refused("not empty or a positive decimal integer", VISION_MAX_IMAGES="0")

    def test_patch_must_exist_under_docker_patches(self) -> None:
        self.refused("must be none or a file under docker/patches/", TF_PATCH="patches/nope.patch")
        self.refused("must be none or a file under docker/patches/", TF_PATCH="../run.sh")

    def test_extra_args_cannot_reset_owned_flags(self) -> None:
        for flag in ("--tp", "--context=8192", "--kv-dtype", "--mtp-drafts", "--parallel", "--no-thinking",
                     "--max-tokens", "--port", "--name", "--master"):
            self.refused("which run.sh passes itself", EXTRA_ARGS=f"{flag} 1")
        self.refused("which run.sh passes itself", EXTRA_ARGS="--vision")
        self.refused("which run.sh passes itself", EXTRA_ARGS="--vision-urls")
        self.refused("refused or unused", EXTRA_ARGS="--prefill-fp8")
        self.refused("refused or unused", EXTRA_ARGS="--vision-offload")
        self.accepted(EXTRA_ARGS="--temperature 0.6 --alias qwen")

    def test_extra_args_prefixes_are_refused(self) -> None:
        # TensorFold's argparse expands unambiguous prefixes: --paral is --parallel.
        for w in ("--paral 8", "--hos 0.0.0.0", "--mtp-d 20", "--con=1024", "--vis", "--prefill"):
            self.refused("EXTRA_ARGS sets", EXTRA_ARGS=w)
        self.accepted(EXTRA_ARGS="--temperature 0.6 --top-p 0.9 --min-p 0.05 --reasoning-effort low --alias q")

    def test_extra_env_is_key_value(self) -> None:
        self.refused("is not KEY=VALUE", EXTRA_ENV="NCCL_PROTO")
        self.accepted(EXTRA_ENV="NCCL_PROTO=LL NCCL_DEBUG=INFO")

    def test_shas_are_full(self) -> None:
        self.refused("not a 40-hex TensorFold commit", TF_SHA="56e2e3e")
        self.refused("not a 40-hex snapshot revision", SNAPSHOT_SHA="main")

    def test_bench_only_binds_loopback(self) -> None:
        self.assertIn("host=127.0.0.1", self.accepted(BENCH_ONLY="1").stdout)
        self.assertIn("host=0.0.0.0", self.accepted().stdout)

    def test_every_refusal_precedes_validate_only_exit(self) -> None:
        run = _read("run.sh")
        exit_at = run.index('if [[ "${VALIDATE_ONLY:-0}" == "1" ]]')
        for needle in ('die "TP=', "exceeds the trained window", "exceeds the long profile", "exceeds the engine cap",
                       "which run.sh passes itself", "has no pin in run.sh PATCH_PINS"):
            self.assertLess(run.index(needle), exit_at, needle)


class ServeArgsTests(unittest.TestCase):
    """The tensorfold argv each rank gets, from run.sh's own serve_args()."""

    def argv(self, rank: str, **env: str) -> list[str]:
        run = _read("run.sh")
        defaults = "".join(m.group(0) + "\n" for m in re.finditer(r'^[A-Z_]+="\$\{[A-Z_]+:-[^\n]*\}"$', run, re.M))
        script = "set -euo pipefail\n" + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        script += defaults + 'API_HOST=0.0.0.0\n' + _func_src(run, "serve_args") + f"serve_args {rank}\n"
        clean = {k: v for k, v in os.environ.items() if k not in OVERRIDES}
        out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True, env=clean)
        return out.stdout.split("\n")[:-1]

    def test_both_ranks_agree_on_engine_settings(self) -> None:
        # TensorFold all-gathers these at startup and refuses a mismatch (engine.py _same_settings).
        r0, r1 = self.argv("0"), self.argv("1")
        for flag in ("--context", "--kv-dtype", "--mtp-drafts", "--mtp-confidence", "--tp", "--master", "--master-port"):
            self.assertEqual(r0[r0.index(flag) + 1], r1[r1.index(flag) + 1], flag)
        self.assertEqual(r1[r1.index("--rank") + 1], "1")
        self.assertEqual(r0[r0.index("--rank") + 1], "0")

    def test_http_flags_only_on_rank0(self) -> None:
        r0, r1 = self.argv("0"), self.argv("1")
        for flag in ("--name", "--host", "--port", "--max-tokens", "--no-thinking"):
            self.assertIn(flag, r0)
            self.assertNotIn(flag, r1)
        self.assertIn("--no-update-check", r1)

    def test_thinking_and_drafts_switches(self) -> None:
        self.assertIn("--thinking", self.argv("0", THINKING="1"))
        r = self.argv("0", MTP_DRAFTS="0")
        self.assertIn("--no-drafts", r)
        self.assertNotIn("--mtp-drafts", r)
        self.assertNotIn("--parallel", self.argv("0"))
        self.assertIn("--parallel", self.argv("0", PARALLEL="4"))

    def test_vision_on_both_ranks_and_its_server_options_on_rank0(self) -> None:
        self.assertNotIn("--vision", self.argv("0"))
        self.assertNotIn("--vision", self.argv("1"))
        r0, r1 = self.argv("0", VISION="1", PARALLEL="4"), self.argv("1", VISION="1", PARALLEL="4")
        self.assertIn("--vision", r0)
        self.assertIn("--vision", r1)                       # both ranks admit the same geometry (rank 1: no tower)
        self.assertIn("--vision", self.argv("1", VISION="1"))   # the serial profile too (part 5)
        for flag in ("--vision-urls", "--vision-max-images"):
            self.assertNotIn(flag, r0)
            self.assertNotIn(flag, r1)
        r0 = self.argv("0", VISION="1", PARALLEL="4", VISION_URLS="1", VISION_MAX_IMAGES="8")
        self.assertIn("--vision-urls", r0)
        self.assertEqual(r0[r0.index("--vision-max-images") + 1], "8")
        r1 = self.argv("1", VISION="1", PARALLEL="4", VISION_URLS="1", VISION_MAX_IMAGES="8")
        self.assertNotIn("--vision-urls", r1)
        self.assertNotIn("--vision-max-images", r1)
        self.assertNotIn("--vision-urls", self.argv("0", VISION_URLS="1"))   # server options only with VISION=1

    def test_tp1_has_no_rendezvous_flags(self) -> None:
        r = self.argv("0", TP="1")
        for flag in ("--tp", "--rank", "--master", "--master-port"):
            self.assertNotIn(flag, r)


class RunOpsTests(unittest.TestCase):
    """Plumbing that only runs on a real boot, checked statically or in isolation."""

    def snapshot_complete(self, snap: Path) -> subprocess.CompletedProcess[str]:
        script = _func_src(_read("run.sh"), "snapshot_complete") + f"SNAPSHOT={shlex.quote(str(snap))}\nsnapshot_complete\n"
        return subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True)

    @staticmethod
    def _shard(path: Path, nbytes: int, truncate: int = 0) -> None:
        header = json.dumps({"w": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (nbytes - truncate))

    def test_snapshot_complete_checks_shards_headers_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            for name in ("config.json", "quantization_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
                         "generation_config.json", "ngram_embedding.safetensors", "vision_k6.safetensors"):
                (snap / name).write_text("x")
            (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
            self._shard(snap / "a.safetensors", 64)
            proc = self.snapshot_complete(snap)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("b.safetensors", proc.stderr)
            self._shard(snap / "b.safetensors", 64, truncate=8)
            proc = self.snapshot_complete(snap)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("b.safetensors (truncated)", proc.stderr)
            self._shard(snap / "b.safetensors", 64)
            self.assertEqual(self.snapshot_complete(snap).returncode, 0)
            (snap / "tokenizer.json").unlink()
            proc = self.snapshot_complete(snap)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("tokenizer.json", proc.stderr)

    def test_container_is_offline_without_token(self) -> None:
        run = _read("run.sh")
        self.assertNotIn("HF_TOKEN", run)
        start = _func_body(run, "start_local")
        self.assertIn('-e "HF_HUB_OFFLINE=1"', start)
        self.assertIn('-e "TENSORFOLD_NO_UPDATE_CHECK=1"', start)
        self.assertIn('"${HF_CACHE}:${HF_HOME_IN_CONTAINER}:ro"', start)
        self.assertIn("--no-update-check", _func_body(run, "serve_args"))

    def test_docker_run_safety_flags(self) -> None:
        start = _func_body(_read("run.sh"), "start_local")
        # --init: rank 1 has no SIGTERM handler. memlock + IPC_LOCK: the n-gram tables are mlocked.
        for flag in ("--init", "--ulimit core=1", '--oom-score-adj "$OOM_SCORE_ADJ"', "--cap-add IPC_LOCK",
                     "--ulimit memlock=-1:-1", "--device /dev/infiniband", "--network host", "--ipc host"):
            self.assertIn(flag, start, flag)
        self.assertIn('-e "NCCL_IB_HCA=$HCA"', start)
        self.assertIn(" $EXTRA_ARGS >/dev/null\n  start_memguard\n}", _read("run.sh"))

    def test_image_is_checked_against_tf_sha(self) -> None:
        run = _read("run.sh")
        self.assertIn('[[ "$have" == "$TF_SHA $TF_PATCH_SHA" ]]', _func_body(run, "ensure_image"))
        self.assertIn('tensorfold.patch_sha" }}', _func_body(run, "image_sha"))
        dockerfile = _read("docker/Dockerfile")
        self.assertIn("LABEL tensorfold.sha=${TF_SHA}", dockerfile)
        self.assertIn("tensorfold.patch_sha=${TF_PATCH_SHA}", dockerfile)
        self.assertIn('| sha256sum -c -', dockerfile)
        self.assertIn(f"FROM ${{BASE}}", _read("docker/Dockerfile"))
        self.assertIn(f"pytorch:26.07-py3@{BASE_DIGEST}", _read("docker/Dockerfile"))
        self.assertIn(f"ARG TF_SHA={TF_SHA}", _read("docker/Dockerfile"))
        # The worker copy is matched by image ID, so a rebuilt image with the same TF sha is resent.
        self.assertIn("{{.Id}}", _func_body(run, "sync_worker_image"))

    def test_worker_forwards_every_generated_setting(self) -> None:
        run = _read("run.sh")
        generated = run.split("# BEGIN generated", 1)[1].split("# END generated", 1)[0]
        names = set(re.findall(r'^([A-Z_]+)="\$\{\1:-', generated, re.M)) - {"ORCHESTRATE", "WORKER_HOST"}
        forward = set(re.search(r"^FORWARD_VARS=\((.*?)\)", run, re.M | re.S).group(1).split())
        self.assertEqual(names - forward, set())

    def test_worker_env_round_trips_quoted_values(self) -> None:
        run = _read("run.sh")
        forward = re.search(r"^FORWARD_VARS=\(.*?\)\n", run, re.M | re.S).group(0)
        names = forward.split("(", 1)[1].rsplit(")", 1)[0].split()
        values = {n: f"v-{n}" for n in names}
        values["EXTRA_ARGS"] = "--alias 'a b' --temperature \"0.6\""
        values["EXTRA_ENV"] = "NCCL_PROTO=LL A=$HOME"
        script = forward + _func_src(run, "worker_env")
        script += "".join(f"{n}={shlex.quote(v)}\n" for n, v in values.items())
        script += 'line="$(worker_env)"\nenv -i bash -c "$line env -0"\n'
        out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout
        got = dict(item.split("=", 1) for item in out.split("\0") if "=" in item)
        for n, v in values.items():
            self.assertEqual(got[n], v, n)
        self.assertEqual(got["ROLE"], "worker")
        self.assertEqual(got["ORCHESTRATE"], "0")

    def test_head_preflight_before_worker_and_watch(self) -> None:
        run = _read("run.sh")
        block = run[run.index('ORCHESTRATE" == "auto" && "$ROLE" == "head"'):]
        scp = block.index("scp ")
        self.assertLess(block.index("refuse_foreign_serve"), scp)
        self.assertLess(block.index("refuse_busy_port"), scp)
        self.assertIn("Refusing to start a TP=2 head rank alone", block)
        wait = _func_body(run, "wait_ready")
        for needle in ("$SERVED_NAME", "/health", "worker_state", "abort_worker_dead", "i % 2 == 0"):
            self.assertIn(needle, wait)

    def test_worker_state_reads_a_missing_container(self) -> None:
        fn = _func_src(_read("run.sh"), "worker_state")
        for ssh_out, rc, want in (("\\nmissing\\n", 0, "missing"), ("false\\n", 0, "false"), ("", 255, "")):
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp) / "ssh"
                fake.write_text(f"#!/bin/sh\nprintf '{ssh_out}'\nexit {rc}\n")
                fake.chmod(0o755)
                script = f"set -euo pipefail\n{fn}WORKER_HOST=w CONTAINER_NAME=c\nprintf '[%s]' \"$(worker_state)\"\n"
                env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}")
                out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, check=True)
                self.assertEqual(out.stdout, f"[{want}]")

    def test_memguard_loop_kills_when_ram_and_swap_are_low(self) -> None:
        body = _func_src(_read("run.sh"), "memguard_loop")
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)
            (stub / "docker").write_text('#!/bin/sh\necho "$@" >> "$STUB_LOG"\necho true\n')
            (stub / "logger").write_text("#!/bin/sh\nexit 0\n")
            (stub / "sleep").write_text("#!/bin/sh\nexit 0\n")
            for f in stub.iterdir():
                f.chmod(0o755)
            log = stub / "calls"
            env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_LOG=str(log))
            proc = subprocess.run(["bash", "-c", body + "\nmemguard_loop tf-test 99999999 99999999"],
                                  capture_output=True, text=True, env=env, timeout=30, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("kill tf-test", log.read_text())

    def test_run_state_under_script_dir(self) -> None:
        for rel in ("run.sh", "stop.sh"):
            text = _read(rel)
            self.assertNotIn("${PWD}/.run-state", text, rel)
            self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', text, rel)
        self.assertIn(".run-state/", _read(".gitignore"))

    def test_stop_is_graceful_head_first(self) -> None:
        stop = _read("stop.sh")
        self.assertIn('docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME"', stop)
        self.assertLess(stop.index("\nstop_local\n"), stop.index('ssh "$WORKER_HOST"'))
        self.assertRegex(stop, r"ssh -o BatchMode=yes.*\n(?:.*\n)*?\s+else\n(?:.*\n)*?\s+exit 1")

    def test_stop_sh_fails_loudly_without_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)
            (stub / "docker").write_text("#!/bin/sh\nexit 0\n")
            (stub / "ssh").write_text("#!/bin/sh\nexit 255\n")
            (stub / "hostname").write_text("#!/bin/sh\necho spark1\n")
            for f in stub.iterdir():
                f.chmod(0o755)
            env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", WORKER_HOST="nowhere")
            proc = subprocess.run([str(ROOT / "stop.sh")], capture_output=True, text=True, env=env, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Cannot SSH to nowhere", proc.stderr)


class ToolsTests(unittest.TestCase):
    def test_bench_decode_is_byte_identical(self) -> None:
        got = hashlib.sha256((ROOT / "bench_decode.py").read_bytes()).hexdigest()
        self.assertEqual(got, BENCH_DECODE_SHA256)

    def test_accept_computes_tokens_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_text(json.dumps({"rounds_total": 10, "drafted_total": 30, "accepted_total": 20, "completion_tokens_total": 0}))
            b.write_text(json.dumps({"rounds_total": 110, "drafted_total": 330, "accepted_total": 270, "completion_tokens_total": 350}))
            out = subprocess.run(["python3", str(ROOT / "tools/accept.py"), str(a), str(b)], check=True,
                                 capture_output=True, text=True).stdout
            got = json.loads(out)
            self.assertEqual(got["tokens_per_round"], 3.5)
            self.assertEqual(got["draft_hit_rate"], 0.8333)

    def test_shell_tools_parse(self) -> None:
        for rel in ("run.sh", "stop.sh", "tools/session_gate.sh", "tools/sweep.sh"):
            subprocess.run(["bash", "-n", str(ROOT / rel)], check=True)


if __name__ == "__main__":
    unittest.main()
