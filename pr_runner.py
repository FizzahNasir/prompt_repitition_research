"""
pr_runner.py — Resume-safe GPU runner for the prompt repetition experiments.

Runs ONE model over the selected languages/methods and writes every generation
to an append-only JSONL file, flushed + fsync'd after each chunk. Optionally
mirrors the files to a private Hugging Face dataset repo so progress survives a
lost Colab/Kaggle session. Re-running the same command skips everything already
done, so a crash or disconnect costs at most one chunk.

Prompts, scenarios, system prompts and scoring all come from the repo
(experiment_runner.py, run_punjabi.py, *_datasets_loader.py), so this file only
handles batching, persistence and resume.

Usage:
    python pr_runner.py --model qwen2.5-3b --out /kaggle/working/results
    python pr_runner.py --model llama3.1-8b --languages ur ar --hf-repo user/pr-results
    python pr_runner.py --model qwen2.5-3b --dry-run          # count jobs, no GPU
"""

import argparse
import contextlib
import gc
import importlib
import io
import json
import os
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

# ── Models ────────────────────────────────────────────────────────────────────
# The 8 non-reasoning instruct models chosen for this study.
MODEL_MAP = {
    "qwen2.5-1.5b":    "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":      "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":      "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-2b":       "google/gemma-2-2b-it",
    "llama3.2-1b":     "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b":     "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b":     "meta-llama/Llama-3.1-8B-Instruct",
}
MODEL_SIZE_B = {
    "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3.1, "qwen2.5-7b": 7.6,
    "mistral-7b-v0.3": 7.3, "gemma2-2b": 2.6,
    "llama3.2-1b": 1.2, "llama3.2-3b": 3.2, "llama3.1-8b": 8.0,
}

LANGUAGES = {
    "pa": ("punjabi_datasets_loader", "SYSTEM_PROMPT_PA"),
    "ur": ("urdu_datasets_loader",    "SYSTEM_PROMPT_UR"),
    "ps": ("pashto_datasets_loader",  "SYSTEM_PROMPT_PS"),
    "ar": ("arabic_datasets_loader",  "SYSTEM_PROMPT_AR"),
    "fa": ("persian_datasets_loader", "SYSTEM_PROMPT_FA"),
    "sd": ("sindhi_datasets_loader",  "SYSTEM_PROMPT_SD"),
}

MAX_TOKENS = 100      # matches experiment_runner.call_model
TEMPERATURE = 0.0     # greedy decoding


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Job construction ──────────────────────────────────────────────────────────

def question_names(item: dict) -> list:
    """List names that appear in the question itself (MiddleMatch's two anchors)."""
    if item.get("task") != "MiddleMatch":
        return []
    return [c for c in item.get("candidates") or [] if c in str(item.get("query", ""))]


def load_dataset(lang: str) -> list:
    loader_mod, _ = LANGUAGES[lang]
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module(loader_mod).build_dataset()


def build_jobs(lang: str, methods: list, skip_tasks: set, tasks: set = None) -> list:
    """Every (item × scenario × method) prompt for one language.

    Each task only runs in the scenarios mapped to it in run_punjabi._TASK_SCENARIOS
    (run_experiment would otherwise also push MCQ items through 4_Question_Only).
    """
    import experiment_runner as er
    from run_punjabi import _TASK_SCENARIOS

    dataset = load_dataset(lang)

    jobs = []
    for item in dataset:
        task = item.get("task", "")
        if (lang, task) in skip_tasks or task in skip_tasks:
            continue
        if tasks and task not in tasks:
            continue
        for sc in _TASK_SCENARIOS.get(task, []):
            base = er.build_base_prompt(sc, item)
            if base is None:
                continue
            answer = (item.get("shuffled_correct_answer", item.get("correct_answer"))
                      if sc == "5_Shuffled_Options" else item.get("correct_answer"))
            variants = er.build_all_methods(base, item.get("language", lang))
            # Other list names that would make an answer ambiguous (names in the
            # question itself, e.g. MiddleMatch's anchors, are expected in a reply).
            distractors = [c for c in item.get("candidates") or []
                           if c != answer and c not in str(item.get("query", ""))]
            anchors = question_names(item)
            for method in methods:
                jobs.append({
                    "key":            f"{lang}|{task}|{sc}|{item.get('id')}|{method}",
                    "language":       lang,
                    "task":           task,
                    "scenario":       sc,
                    "method":         method,
                    "item_id":        item.get("id"),
                    "correct_answer": answer,
                    "distractors":    distractors,
                    "anchors":        anchors,
                    "prompt":         variants[method],
                })
    keys = [j["key"] for j in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{lang}: duplicate job keys — item ids are not unique")
    return jobs, getattr(er, LANGUAGES[lang][1])


# ── Persistence ───────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list:
    """Read a JSONL file, skipping a truncated/corrupt line (e.g. from a hard kill)."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_jsonl_atomic(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def merge_rows(*row_lists) -> list:
    """Union by key; a successful row always beats an errored one."""
    best = {}
    for rows in row_lists:
        for r in rows:
            k = r.get("key")
            if k is None:
                continue
            if k not in best or (best[k].get("error") and not r.get("error")):
                best[k] = r
    return list(best.values())


class HubMirror:
    """Mirrors result files to a private HF dataset repo (no-op if repo_id is empty)."""

    def __init__(self, repo_id: str, interval_s: int):
        self.repo_id = repo_id
        self.interval_s = interval_s
        self.last = {}
        self.api = None
        if repo_id:
            from huggingface_hub import HfApi
            self.api = HfApi()
            self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def list(self, prefix: str) -> list:
        if not self.api:
            return []
        return [f for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
                if f.startswith(prefix)]

    def download(self, path_in_repo: str):
        if not self.api:
            return []
        from huggingface_hub import hf_hub_download
        try:
            local = hf_hub_download(self.repo_id, path_in_repo, repo_type="dataset",
                                    force_download=True)
        except Exception:
            return []  # not uploaded yet
        return read_jsonl(Path(local))

    def upload(self, local: Path, path_in_repo: str, force: bool = False) -> None:
        if not self.api:
            return
        now = time.time()
        if not force and now - self.last.get(path_in_repo, 0) < self.interval_s:
            return
        for attempt in range(3):
            try:
                self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                     repo_id=self.repo_id, repo_type="dataset",
                                     commit_message=f"checkpoint {path_in_repo}")
                self.last[path_in_repo] = now
                return
            except Exception as e:  # network hiccup must never kill the run
                log(f"  HF upload failed ({attempt + 1}/3): {e}")
                time.sleep(10 * (attempt + 1))


# ── Prompt tokenization ───────────────────────────────────────────────────────

def make_encoder(tokenizer, sample_system: str):
    """Return encode(system, prompt) -> token ids, plus how the system prompt is sent.

    Models whose chat template rejects a system role (e.g. Gemma) get the system
    prompt prepended to the user turn instead; this is decided once per model so
    every prompt for that model is formatted the same way.
    """
    def _ids(messages):
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        return list(out)

    try:
        _ids([{"role": "system", "content": sample_system}, {"role": "user", "content": "x"}])
        mode = "system_role"
    except Exception:
        mode = "prepended_to_user"

    def encode(system: str, prompt: str):
        if mode == "system_role":
            return _ids([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return _ids([{"role": "user", "content": f"{system}\n\n{prompt}"}])

    return encode, mode


# ── Backends ──────────────────────────────────────────────────────────────────

def _gpu_info():
    import torch
    if not torch.cuda.is_available():
        return 0, 0.0, False
    n = torch.cuda.device_count()
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    return n, mem, bf16


def pick_dtype(hf_name: str, bf16: bool) -> str:
    """bf16 where the GPU supports it. On fp16-only GPUs (T4), Gemma-2 overflows to
    NaN in fp16, so it runs in fp32 there; everything else uses fp16."""
    if bf16:
        return "bfloat16"
    return "float32" if "gemma-2" in hf_name.lower() else "float16"


class VLLMBackend:
    name = "vllm"

    def __init__(self, hf_name, size_b, max_model_len):
        import vllm
        from vllm import LLM, SamplingParams
        n_gpu, mem, bf16 = _gpu_info()
        dtype = pick_dtype(hf_name, bf16)
        bytes_per_param = 4 if dtype == "float32" else 2
        # Split across GPUs only when the weights would crowd a single card.
        tp = n_gpu if (n_gpu > 1 and size_b * bytes_per_param > 0.5 * mem) else 1
        log(f"  vLLM {vllm.__version__}: tp={tp} dtype={dtype} max_model_len={max_model_len}")
        self.dtype = dtype
        self.version = f"vllm-{vllm.__version__}"
        self.llm = LLM(model=hf_name, dtype=dtype, tensor_parallel_size=tp,
                       max_model_len=max_model_len, gpu_memory_utilization=0.90, seed=0,
                       disable_custom_all_reduce=tp > 1)  # needed for PCIe T4 pairs
        self.params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, id_lists):
        outs = self.llm.generate([{"prompt_token_ids": ids} for ids in id_lists],
                                 self.params, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason)
                for o in outs]


class HFBackend:
    name = "hf"

    def __init__(self, hf_name, size_b, max_model_len, batch_size=16):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _, _, bf16 = _gpu_info()
        self.torch = torch
        self.dtype = pick_dtype(hf_name, bf16)
        self.version = f"transformers-{transformers.__version__}"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        extra = {"attn_implementation": "eager"} if "gemma-2" in hf_name.lower() else {}  # softcapping
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=getattr(torch, self.dtype), device_map="auto", **extra)
        self.model.eval()
        self.batch_size = batch_size

    def _run(self, batch):
        torch = self.torch
        width = max(len(x) for x in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, x in enumerate(batch):  # left padding
            ids[i, width - len(x):] = torch.tensor(x)
            mask[i, width - len(x):] = 1
        dev = self.model.device
        with torch.no_grad():
            out = self.model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                                      max_new_tokens=MAX_TOKENS, do_sample=False,
                                      pad_token_id=self.pad_id)
        res = []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos, self.tokenizer.eos_token_id])
        for row in out[:, width:].tolist():
            stop = next((n for n, t in enumerate(row) if t in eos), None)
            if stop is not None:
                row = row[:stop + 1]
            text = self.tokenizer.decode(row, skip_special_tokens=True)
            finish = "length" if len(row) >= MAX_TOKENS else "stop"
            res.append((text, len(row), finish))
        return res

    def generate(self, id_lists):
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        results = [None] * len(id_lists)
        i = 0
        while i < len(order):
            idx = order[i:i + self.batch_size]
            try:
                for j, r in zip(idx, self._run([id_lists[j] for j in idx])):
                    results[j] = r
                i += len(idx)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                log(f"  OOM -> batch_size={self.batch_size}")
        return results


def load_backend(kind, hf_name, size_b, max_model_len):
    if kind in ("auto", "vllm"):
        try:
            return VLLMBackend(hf_name, size_b, max_model_len)
        except Exception as e:
            if kind == "vllm":
                raise
            log(f"  vLLM unavailable for {hf_name} ({type(e).__name__}: {e}); using transformers")
            gc.collect()
    return HFBackend(hf_name, size_b, max_model_len)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_MAP))
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=list(LANGUAGES))
    p.add_argument("--methods", nargs="+", default=["baseline", "repetition"],
                   choices=["baseline", "repetition", "cross_lingual_t1", "cross_lingual_t2"])
    p.add_argument("--tasks", nargs="+", default=None, help="Only these tasks")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Tasks to skip: TASK or LANG:TASK (e.g. ar:GSM8K)")
    p.add_argument("--out", default="results", help="Persistent results directory")
    p.add_argument("--hf-repo", default=os.environ.get("HF_RESULTS_REPO", ""),
                   help="Private HF dataset repo to mirror results to (e.g. user/pr-results)")
    p.add_argument("--sync-every", type=int, default=300, help="Seconds between HF uploads")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--chunk", type=int, default=512, help="Prompts per checkpoint")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--limit", type=int, default=None, help="Max jobs per language (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="Count jobs only; no model load")
    p.add_argument("--stop-at", type=float, default=None,
                   help="Unix time to stop cleanly by (e.g. before Kaggle's 12 h limit)")
    p.add_argument("--shard", default="0/1",
                   help="i/n: run only shard i of n (one process per GPU for small models)")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    skip = set()
    for s in args.skip:
        skip.add(tuple(s.split(":", 1)) if ":" in s else s)
    tasks = set(args.tasks) if args.tasks else None

    out_dir = Path(args.out) / "raw" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror = HubMirror(args.hf_repo, args.sync_every) if not args.dry_run else HubMirror("", 0)

    # 1. Build jobs and restore progress (local ∪ HF mirror) for every language first,
    #    so the model is only loaded if there is work left.
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    suffix = f".shard{shard_i}of{shard_n}" if shard_n > 1 else ""
    remote_files = mirror.list(f"raw/{args.model}/")
    plan = []
    for lang in args.languages:
        jobs, system_prompt = build_jobs(lang, args.methods, skip, tasks)
        if args.limit:
            jobs = jobs[:args.limit]
        if shard_n > 1:  # stable split, independent of job order
            jobs = [j for j in jobs if zlib.crc32(j["key"].encode()) % shard_n == shard_i]
        path = out_dir / f"{lang}{suffix}.jsonl"
        rel = f"raw/{args.model}/{path.name}"
        # Restore every file for this language (any earlier sharding) from the mirror,
        # then treat a job as done if any file has it.
        for remote in remote_files:
            name = remote.rsplit("/", 1)[-1]
            if name == f"{lang}.jsonl" or name.startswith(f"{lang}.shard"):
                local = out_dir / name
                rows = merge_rows(read_jsonl(local), mirror.download(remote))
                if rows:
                    write_jsonl_atomic(local, rows)
        done = set()
        for f in out_dir.glob(f"{lang}*.jsonl"):
            if f.name == f"{lang}.jsonl" or f.name.startswith(f"{lang}.shard"):
                done |= {r["key"] for r in read_jsonl(f) if not r.get("error")}
        todo = [j for j in jobs if j["key"] not in done]
        log(f"{args.model} | {lang}: {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run")
        plan.append((lang, system_prompt, path, rel, todo))

    remaining = sum(len(t) for *_, t in plan)
    if args.dry_run or remaining == 0:
        log("Nothing to run." if remaining == 0 else f"Dry run: {remaining} jobs pending.")
        return

    # 2. Load the model once and work through every language.
    import experiment_runner as er
    hf_name = MODEL_MAP[args.model]
    log(f"Loading {hf_name}")
    backend = load_backend(args.backend, hf_name, MODEL_SIZE_B[args.model], args.max_model_len)
    encode, sys_mode = make_encoder(backend.tokenizer, plan[0][1])
    try:
        import torch
        gpu = f"{torch.cuda.device_count()}x{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    except ImportError:
        gpu = "cpu"
    log(f"  backend={backend.version} dtype={backend.dtype} gpu={gpu} system_prompt_mode={sys_mode}")

    t_start, n_run = time.time(), 0
    for lang, system_prompt, path, rel, todo in plan:
        if not todo:
            continue
        log(f"=== {args.model} | {lang}: {len(todo)} prompts ===")
        # Repair a missing trailing newline left by a hard kill before appending.
        if path.exists() and path.stat().st_size:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    with open(path, "a", encoding="utf-8") as g:
                        g.write("\n")

        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            ids, runnable, rows = [], [], []
            for j in chunk:
                x = encode(system_prompt, j["prompt"])
                if len(x) > args.max_model_len - MAX_TOKENS:
                    rows.append({**{k: v for k, v in j.items() if k != "prompt"},
                                 "model": args.model, "response": None, "is_correct": None,
                                 "prompt_tokens": len(x), "skipped": "prompt_too_long",
                                 "error": None})
                    continue
                ids.append(x)
                runnable.append(j)
            t0 = time.time()
            try:
                gens = backend.generate(ids) if ids else []
                err = None
            except Exception as e:
                gens, err = [], f"{type(e).__name__}: {e}"
                log(f"  generation failed: {err}")
            dt = time.time() - t0
            stamp = datetime.now().isoformat(timespec="seconds")
            for k, j in enumerate(runnable):
                row = {k2: v for k2, v in j.items() if k2 != "prompt"}
                row.update(model=args.model, hf_model=hf_name, backend=backend.version,
                           dtype=backend.dtype, gpu=gpu, system_prompt_mode=sys_mode, prompt_chars=len(j["prompt"]),
                           prompt_tokens=len(ids[k]), timestamp=stamp)
                if err:
                    row.update(response=None, is_correct=None, error=err)
                else:
                    text, n_out, finish = gens[k]
                    row.update(response=text, output_tokens=n_out, finish_reason=finish,
                               is_correct=bool(er.is_correct(j["task"], text, j["correct_answer"],
                                                             j["distractors"], j["anchors"])),
                               latency_ms_per_prompt=round(dt * 1000 / max(len(ids), 1), 1),
                               error=None)
                rows.append(row)
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            mirror.upload(path, rel)

            n_run += len(chunk)
            rate = n_run / max(time.time() - t_start, 1e-6)
            left = remaining - n_run
            log(f"  {lang} {min(start + len(chunk), len(todo))}/{len(todo)}  "
                f"{rate:.1f} prompts/s  ETA(all langs) {left / max(rate, 1e-6) / 60:.0f} min")
            if err and "out of memory" in err.lower():
                sys.exit(f"Stopping: {err}")
            if args.stop_at and time.time() > args.stop_at:
                mirror.upload(path, rel, force=True)
                log("Time budget reached; progress saved. Re-run to continue.")
                sys.exit(3)
        mirror.upload(path, rel, force=True)

    log(f"Done: {args.model} ({n_run} prompts in {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
