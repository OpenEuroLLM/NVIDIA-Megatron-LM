# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import packed_doc_attention as pda
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, get_tokenizer, inprocess_restart, pretrain, print_rank_0
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

# OELLM PATCH: let training.train_step drive the scatter-mode prefetch without importing
# this module (see megatron/training/packed_doc_attention.py).
pda.register_prefetch_hook(lambda *a: _prefetch_cu_seqlens_for_iteration(*a))

# OELLM PATCH: the end-of-document token id now lives in megatron.training.packed_doc_
# attention, because the cu_seqlens derivation moved into get_batch_on_this_tp_rank (host
# side, before the H2D copy) and megatron.training.utils must not import this script.
_get_eod_token_id = pda.get_eod_token_id


def _build_packed_seq_params(tokens: torch.Tensor, eod_id: int) -> PackedSeqParams:
    """Describe the document boundaries inside `tokens` as thd packed-sequence params.

    --reset-attention-mask does not actually disable cross-document attention: the GPT layer
    specs pin attn_mask_type to `causal` (models/gpt/gpt_layer_specs.py) and TE only reads
    `attention_mask` for `padding`/`arbitrary` mask types, so the dense [b, 1, s, s] mask the
    dataloader builds is discarded. Handing TE cu_seqlens instead makes the flash/cuDNN varlen
    kernels skip the off-diagonal blocks outright, which both masks cross-document attention
    and restarts RoPE per document (transformer/attention.py applies rope with cu_seqlens).

    `tokens` is [b, s] and is consumed as one flattened pack of b*s tokens, so sample
    boundaries close a document too.
    """
    batch_size, seq_length = tokens.shape
    flat = tokens.reshape(-1)

    # A document ends *after* its EOD token, hence the +1.
    ends = torch.nonzero(flat == eod_id, as_tuple=True)[0] + 1
    sample_ends = torch.arange(
        seq_length, batch_size * seq_length + 1, seq_length,
        device=tokens.device, dtype=ends.dtype,
    )
    # unique() sorts and drops the duplicate a sample ending on EOD would produce. Every
    # resulting segment has length >= 1, which is what TE requires of cu_seqlens.
    bounds = torch.unique(torch.cat((ends, sample_ends)))
    cu_seqlens = torch.cat(
        (torch.zeros(1, device=tokens.device, dtype=bounds.dtype), bounds)
    ).to(torch.int32)

    # TE needs a python int here. Deliberately the true maximum rather than args.seq_length:
    # _apply_rotary_pos_emb_thd switches to absolute-position mapping when
    # freqs.size(0) == cu_seqlens[-1], which would silently stop RoPE from restarting per
    # document on the unfused rope path. Costs one device sync per microbatch.
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _shared_packed_seq_params(packed_length: int):
    """Share TP rank 0's host-derived thd PackedSeqParams across the TP group.

    Every transformer layer on every rank needs cu_seqlens, but not every rank has the
    tokens: `get_batch_on_this_tp_rank` broadcasts tokens on the first stage, sets them
    to None on the last (utils.py:600-609), and for middle stages broadcasts nothing. So
    TP rank 0 derives it -- on the HOST, before the tokens are uploaded -- and hands it to
    the rest of its tensor-parallel group.

    Takes no tokens: the derivation happens in get_batch_on_this_tp_rank now, because that
    is the last place the tokens exist on the CPU. Deriving here would mean deriving from
    the uploaded copy, which is what used to cost a readback.

    Safe as a collective because all TP ranks of a given (stage, chunk) run this at the
    same point for the same microbatch -- the assumption get_batch_on_this_tp_rank already
    makes for its own broadcasts. Note the group is deliberately the TENSOR-parallel group,
    never the pipeline one.

    ================== DO NOT WIDEN THIS TO THE PIPELINE GROUP ==================
    Sending cu_seqlens across pipeline stages from here DEADLOCKS. Measured, not
    theorised (job 1494386, hung at the first training step, cancelled):

        rank 0 (first stage) enters forward_step -> get_batch -> broadcast, and blocks
               waiting for every rank in the group;
        rank N (last stage) is in recv_forward, blocked on rank N-1 <- ... <- rank 0;
        rank 0 cannot produce those activations because it is stuck in the broadcast.

    A circular wait between the collective and the p2p activation chain. This is NOT
    about microbatch ordering -- the interleaved schedule's tables (get_schedule_table,
    schedules.py:1045) are rank-independent, so the k-th forward is the same
    (chunk, microbatch) everywhere and the CONTENT would have been correct. The problem is
    purely that a blocking collective cannot sit inside a pipeline whose stages are waiting
    on each other's activations. Every pipeline stage deriving cu_seqlens locally is
    therefore not one of two options; it is the only one that works.
    =============================================================================

    NO DEVICE READBACK HAPPENS HERE ANY MORE. The derivation moved to the host, into
    get_batch_on_this_tp_rank, where the tokens still live on the CPU: max_seqlen and the
    entry count are then Python ints by construction. The share is a GLOO broadcast of a
    small host buffer, so the header arrives host-side and the only device traffic is one
    async H2D copy of the exact-size cu_seqlens.

    The old version derived on the GPU and paid two stream drains per microbatch per rank
    -- `int(...max())` for TE's kernel-launch int, and `.tolist()` to learn the slice
    length. Both sat at the top of forward_step, so they destroyed CPU run-ahead M times
    per iteration.

    The buffer is fixed-size so every rank agrees on the shape before it knows the document
    count -- a pack of `packed_length` tokens holds at most that many documents, hence at
    most packed_length + 1 cu_seqlens entries. Layout: [n_entries, max_seqlen, cu_seqlens...].
    """
    cu_seqlens, max_seqlen = pda.share_cu_seqlens_over_tp(packed_length)
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


_PDA_LOGGED_CALLS = 0


def _log_cu_seqlens(packed_seq_params, vp_stage):
    """Print this rank's cu_seqlens so stage agreement can be checked from the logs.

    cu_seqlens is a handful of int32s -- a few dozen bytes -- so logging it outright is
    cheaper than any runtime cross-check, and unlike a collective it cannot deadlock
    against the pipeline's p2p chain (see _shared_packed_seq_params for that failure).

    DEBUG ONLY, and the default of 0 matters: the `.tolist()` below is a device->host
    readback, i.e. exactly the per-microbatch stream drain the host-side derivation exists
    to remove. Turning this on re-introduces it.

    Ranks are comparable by call index: the interleaved schedule's tables
    (get_schedule_table, schedules.py:1045) are rank-independent, so the k-th call to
    get_batch is the same (chunk, microbatch) on every rank -- only the wall-clock timing
    differs. So all ranks sharing a tensor-parallel index must print identical cu_seqlens
    for a given call. Check with scripts/korbi/check_cu_seqlens_agreement.py.
    """
    global _PDA_LOGGED_CALLS
    limit = get_args().packed_doc_attention_log_cu_seqlens
    if _PDA_LOGGED_CALLS >= limit:
        return
    print(
        f"[PDA] call={_PDA_LOGGED_CALLS} "
        f"rank={torch.distributed.get_rank()} "
        f"pp={parallel_state.get_pipeline_model_parallel_rank()} "
        f"tp={parallel_state.get_tensor_model_parallel_rank()} "
        f"vp={vp_stage} "
        f"max_seqlen={packed_seq_params.max_seqlen_q} "
        f"cu_seqlens={packed_seq_params.cu_seqlens_q.tolist()}",
        flush=True,
    )
    _PDA_LOGGED_CALLS += 1


def _prefetch_cu_seqlens_for_iteration(data_iterator, vp_stage, num_microbatches, derive=True):
    """OELLM PATCH: scatter-mode hook, called from train_step before the schedule runs.

    Registered with megatron.training.packed_doc_attention so training.py does not have to
    import this module. See that module for why the collective cannot live in get_batch.
    """
    args = get_args()

    def fetch(iterator):
        # get_batch_on_this_tp_rank derives cu_seqlens on the host and stashes it, which is
        # what prefetch_iteration picks up -- so the fetch must go through it.
        return get_batch_on_this_cp_rank(get_batch_on_this_tp_rank(iterator))

    pda.prefetch_iteration(
        data_iterator=data_iterator,
        vp_stage=vp_stage,
        num_microbatches=num_microbatches,
        packed_length=args.seq_length * args.micro_batch_size,
        fetch_batch=fetch,
        reads_data=is_first_or_last_pipeline_stage(vp_stage),
        derive=derive,
    )


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    args = get_args()
    is_endpoint = is_first_or_last_pipeline_stage(vp_stage)
    scatter = args.packed_doc_attention and args.packed_doc_attention_scatter

    # Which ranks pull from the dataloader:
    #   plain           endpoints only (upstream)
    #   packed local    EVERY stage -- each derives cu_seqlens itself, because a collective
    #                   here would deadlock against the p2p chain (_shared_packed_seq_params)
    #   packed scatter  endpoints only -- cu_seqlens was broadcast for the whole iteration
    #                   before the schedule started (training/packed_doc_attention.py)
    reads_data = is_endpoint or (args.packed_doc_attention and not scatter)

    # TODO: this is pretty hacky, find a better way
    if not reads_data and not args.packed_doc_attention:
        return None, None, None, None, None, None

    if scatter:
        batch, cu_seqlens, max_seqlen = pda.pop(vp_stage, reads_data=reads_data)
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
    else:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)

        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

        if not args.packed_doc_attention:
            return (*batch.values(), None)

        packed_seq_params = _shared_packed_seq_params(
            args.seq_length * args.micro_batch_size
        )

    if args.packed_doc_attention_log_cu_seqlens:
        _log_cu_seqlens(packed_seq_params, vp_stage)

    if batch is None:
        # Middle stages consume activations, not tokens; upstream hands them all-None and
        # the packed params are the only addition.
        return None, None, None, None, None, packed_seq_params

    # Fold the micro-batch into a single packed sequence: TE's thd kernels take [t, h, d], and
    # mcore reaches that shape via query.squeeze(1) (transformer/attention.py), which needs a
    # batch dimension of 1. Sample boundaries are already document boundaries in cu_seqlens.
    # The pipeline p2p buffers are folded to match in training.get_pipeline_tensor_shapes().
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
        if batch[key] is not None:
            batch[key] = batch[key].reshape(1, -1)

    return (
        batch['tokens'],
        batch['labels'],
        batch['loss_mask'],
        None,  # attention_mask: cu_seqlens replaces it
        batch['position_ids'],
        packed_seq_params,
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator, vp_stage
        )
    timers('batch-generator').stop()

    # OELLM PATCH: only forwarded when --packed-doc-attention is set, so the default path stays
    # byte-identical to upstream.
    packed_kwargs = {} if packed_seq_params is None else {'packed_seq_params': packed_seq_params}

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                assert not args.packed_doc_attention, \
                    "packed_doc_attention is not supported with build_schedule_plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                    **packed_kwargs
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    if args.packed_doc_attention and not args.packed_doc_attention_scatter:
        # OELLM PATCH: every pipeline stage needs cu_seqlens for its own transformer layers.
        # In this mode each stage derives it from its own copy of the batch, which is why
        # every stage builds the dataset.
        #
        # Stages agree because the sampler is deterministic in (data_parallel_rank,
        # consumed_samples) -- both equal across the pipeline stages of one DP replica --
        # and because each model chunk owns its own data iterator (training.py:708-724),
        # consumed in microbatch order. The barriers in blended_megatron_dataset_builder are
        # symmetric in is_built_on_rank (the short-cut path at :422 issues the same count as
        # the build path), so widening this is safe.
        #
        # Sending cu_seqlens across stages instead is not an option: a blocking collective
        # inside forward_step deadlocks against the p2p activation chain. Measured in job
        # 1494386. See _shared_packed_seq_params for the mechanism.
        return parallel_state.get_tensor_model_parallel_rank() == 0
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
