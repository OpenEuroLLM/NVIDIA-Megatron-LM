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
"greedy" (LAMBADA) -> acc (every continuation token is the argmax) and the continuation's mean NLL;
"ppl" (long documents, one item with an empty continuation) -> the mean next-token NLL per position
bin (PPL_BINS), token-weighted over the documents.
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


# position bins of the "ppl" items (long documents): [lo, hi) token positions
PPL_BINS = (0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


@torch.no_grad()
def score_ppl(model, base, ctx, device, chunk: int = 2048):
    """Per-bin (sum NLL, count) over every next-token prediction of one long document ``ctx``.

    The output layer's pre-hook keeps the final hidden states and hands the layer one position;
    the logits are then computed 2k positions at a time, so 64k-token documents never hold the
    full [s, V] logits."""
    tokens = torch.tensor([ctx], dtype=torch.long, device=device)
    seq = tokens.shape[1]
    kept = {}

    def pre_hook(module, args, kwargs):
        kept["h"] = args[0]
        return (args[0][:1],) + tuple(args[1:]), kwargs

    handle = base.output_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        pos = torch.arange(seq, device=device).unsqueeze(0)
        model(tokens, pos, None)
    finally:
        handle.remove()
    h = kept["h"]  # [s, 1, hidden]
    nll = torch.empty(seq - 1, device=device)
    weight = base.shared_embedding_or_output_weight() if getattr(base, "share_embeddings_and_output_weights", False) else None
    for lo in range(0, seq - 1, chunk):
        hi = min(lo + chunk, seq - 1)
        out = base.output_layer(h[lo:hi], weight=weight)
        logits = out[0] if isinstance(out, tuple) else out  # [n, 1, V]
        if hasattr(base, "_scale_logits"):  # as the model's own head (mup output scaling)
            logits = base._scale_logits(logits)
        logp = torch.log_softmax(logits[:, 0].float(), dim=-1)
        nll[lo:hi] = -logp.gather(-1, tokens[0, lo + 1:hi + 1].unsqueeze(-1)).squeeze(-1)
    bins = {}
    for b_lo, b_hi in zip(PPL_BINS[:-1], PPL_BINS[1:]):
        if b_lo >= seq - 1:
            break
        part = nll[b_lo:min(b_hi, seq - 1)]
        bins[f"{b_lo}-{b_hi}"] = (part.sum().item(), part.numel())
    return bins


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
        if questions[qi]["kind"] == "ppl":
            lp, greedy = score_ppl(chunks[0], base, it["ctx"], device), None
        else:
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
        if q["kind"] == "ppl":  # mean NLL per position bin, over the documents' tokens
            for name, (tot, cnt) in items[0][0].items():
                agg[t][f"nll_sum_{name}"] += tot
                agg[t][f"count_{name}"] += cnt
            continue
        if q["kind"] == "greedy":
            lp, greedy = items[0]
            agg[t]["acc"] += greedy
            agg[t]["nll_per_token"] += -lp / max(1, len(q["items"][0]["cont"]))
        else:
            lps = [items[i][0] for i in range(len(q["items"]))]
            norm = [lp / max(1, q["items"][i]["nchars"]) for i, lp in enumerate(lps)]
            agg[t]["acc"] += int(max(range(len(lps)), key=lps.__getitem__) == q["gold"])
            agg[t]["acc_norm"] += int(max(range(len(norm)), key=norm.__getitem__) == q["gold"])
    out = {}
    for t, m in agg.items():
        if any(k.startswith("nll_sum_") for k in m):  # ppl: token-weighted mean per bin
            out[t] = {"n": int(m["n"])}
            for k, v in m.items():
                if k.startswith("nll_sum_"):
                    name = k[len("nll_sum_"):]
                    out[t][f"nll_{name}"] = v / max(1.0, m[f"count_{name}"])
                    out[t][f"tokens_{name}"] = int(m[f"count_{name}"])
        else:
            out[t] = {k: (v / m["n"] if k != "n" else int(v)) for k, v in m.items()}
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
