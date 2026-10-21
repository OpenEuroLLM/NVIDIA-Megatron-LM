# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Log-likelihood downstream evaluation inside Megatron (DOWNSTREAM_EVAL_FILE=<requests.jsonl>).

For models without a Hugging Face conversion (the hybrids: Complex KDA / KronLSTM mixers, mHC,
NoPE gated attention), run after the checkpoint load with --skip-train. The requests come from
scripts/korbi/downstream/prepare_tasks.py: per question a list of (context, continuation) token
items; each item is scored as sum log p(continuation | context) in one forward pass of
context + continuation. Only the continuation's positions reach the output layer (a forward
pre-hook slices the hidden states), so long contexts (the passkey task at 32k) never materialize
full-vocabulary logits.

Metrics per task: "mc" questions -> acc (argmax of the sum) and acc_norm (sum / characters);
"greedy" (LAMBADA) -> acc (every continuation token is the argmax) and the continuation's mean NLL.
Questions are spread over all ranks (each rank runs one item per forward; ranks run equally
many forwards, padded with repeats, since expert parallelism exchanges tokens every layer).
Results: printed by rank 0 as "[downstream_eval] {json}" and written to DOWNSTREAM_EVAL_OUT, by
default <save>/downstream.json.
Only TP = PP = CP = 1 (data and expert parallelism).
"""

import json
import os
from collections import defaultdict

import torch


def _unwrap(model):
    m = model[0] if isinstance(model, list) else model
    while hasattr(m, "module"):
        m = m.module
    return m


@torch.no_grad()
def score_item(model, base, ctx, cont, device):
    """(sum log p(cont | ctx), whether every cont token is the argmax) for one item."""
    tokens = torch.tensor([ctx + cont], dtype=torch.long, device=device)
    n = len(cont)
    seq = tokens.shape[1]
    window = {}

    def pre_hook(module, args, kwargs):
        h = args[0]  # [s, b, h]: keep the positions predicting the continuation
        window["start"] = seq - n - 1
        return (h[seq - n - 1: seq - 1],) + tuple(args[1:]), kwargs

    handle = base.output_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        pos = torch.arange(seq, device=device).unsqueeze(0)
        logits = model(tokens, pos, None)  # [b, n, V]
    finally:
        handle.remove()
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    tgt = tokens[0, seq - n:]
    lp = logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
    greedy = bool((logp.argmax(-1) == tgt).all().item())
    return lp, greedy


def maybe_run_downstream_eval(model) -> None:
    path = os.environ.get("DOWNSTREAM_EVAL_FILE")
    if not path:
        return
    dist = torch.distributed
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device("cuda", torch.cuda.current_device())
    tasks = set(filter(None, os.environ.get("DOWNSTREAM_EVAL_TASKS", "").split(",")))
    questions = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if not tasks or q["task"] in tasks or q["task"].split("_")[0] in tasks:
                questions.append(q)
    work = [(qi, ii) for qi, q in enumerate(questions) for ii in range(len(q["items"]))]
    # longest first, round-robin: ranks get similar loads; then pad to equal counts
    work.sort(key=lambda w: -(len(questions[w[0]]["items"][w[1]]["ctx"])
                              + len(questions[w[0]]["items"][w[1]]["cont"])))
    mine = work[rank::world]
    n_steps = (len(work) + world - 1) // world
    chunks = model if isinstance(model, list) else [model]
    for c in chunks:
        c.eval()
    base = _unwrap(model)
    results = []
    for step in range(n_steps):
        qi, ii = mine[step] if step < len(mine) else work[0]  # padding: a repeat, discarded
        it = questions[qi]["items"][ii]
        lp, greedy = score_item(chunks[0], base, it["ctx"], it["cont"], device)
        if step < len(mine):
            results.append((qi, ii, lp, greedy))
        if rank == 0 and step % 500 == 0:
            print(f"[downstream_eval] {step}/{n_steps} forwards", flush=True)
    gathered = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(gathered, results)
    else:
        gathered = [results]
    if rank != 0:
        return
    per_q = defaultdict(dict)
    for part in gathered:
        for qi, ii, lp, greedy in part:
            per_q[qi][ii] = (lp, greedy)
    agg = defaultdict(lambda: defaultdict(float))
    for qi, q in enumerate(questions):
        items = per_q[qi]
        t = q["task"]
        agg[t]["n"] += 1
        if q["kind"] == "greedy":
            lp, greedy = items[0]
            agg[t]["acc"] += greedy
            agg[t]["nll_per_token"] += -lp / max(1, len(q["items"][0]["cont"]))
        else:
            lps = [items[i][0] for i in range(len(q["items"]))]
            norm = [lp / max(1, q["items"][i]["nchars"]) for i, lp in enumerate(lps)]
            agg[t]["acc"] += int(max(range(len(lps)), key=lps.__getitem__) == q["gold"])
            agg[t]["acc_norm"] += int(max(range(len(norm)), key=norm.__getitem__) == q["gold"])
    out = {t: {k: (v / m["n"] if k != "n" else int(v)) for k, v in m.items()} for t, m in agg.items()}
    print("[downstream_eval] " + json.dumps(out, sort_keys=True), flush=True)
    dest = os.environ.get("DOWNSTREAM_EVAL_OUT")
    if not dest:  # default: next to the run's checkpoints (one env for every arm of an array job)
        try:
            from megatron.training import get_args

            save = getattr(get_args(), "save", None)
        except Exception:  # noqa: BLE001 - no global args (tests)
            save = None
        dest = os.path.join(save, "downstream.json") if save else None
    if dest:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
