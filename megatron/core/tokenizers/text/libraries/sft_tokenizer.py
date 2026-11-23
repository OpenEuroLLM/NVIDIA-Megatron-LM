# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import transformers

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False



# fmt: off
nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
identity_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
qwen3p5_reasoning_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{% set content = message['content'] %}{% set reasoning_content = '' %}{% if message.reasoning_content is string %}{% set reasoning_content = message.reasoning_content %}{% elif '</think>' in content %}{% set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}{% set content = content.split('</think>')[-1].lstrip('\n') %}{% endif %}{{ '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content + '<|im_end|>\n' }}{% endif %}{% endfor %}"""  # pylint: disable=line-too-long
qwen3p5_no_reasoning_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}"""  # pylint: disable=line-too-long
# fmt: on

# The qwen3p5-tools template is the official Qwen3.5 chat template augmented with
# {% generation %} markers so return_assistant_tokens_mask yields the SFT loss mask
# in a single render. It handles reasoning (<think>) and tool calling (tools= kwarg,
# assistant.tool_calls, role:tool) in one format. Shipped as a sidecar .jinja file
# because it is far too large to inline readably.
_QWEN3P5_TOOLS_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen3p5_tools_chat_template.jinja"
)


def _load_qwen3p5_tools_template():
    with open(_QWEN3P5_TOOLS_TEMPLATE_PATH) as f:
        return f.read()


_QWEN3P5_TOOLS_NOTHINK_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen3p5_tools_nothink_chat_template.jinja"
)


def _load_qwen3p5_tools_nothink_template():
    with open(_QWEN3P5_TOOLS_NOTHINK_TEMPLATE_PATH) as f:
        return f.read()


_QWEN3P5_TOOLS_LASTTURN_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen3p5_tools_lastturn_chat_template.jinja"
)


def _load_qwen3p5_tools_lastturn_template():
    with open(_QWEN3P5_TOOLS_LASTTURN_TEMPLATE_PATH) as f:
        return f.read()


_QWEN3P5_TOOLS_MASKTHINK_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen3p5_tools_maskthink_chat_template.jinja"
)


def _load_qwen3p5_tools_maskthink_template():
    with open(_QWEN3P5_TOOLS_MASKTHINK_TEMPLATE_PATH) as f:
        return f.read()


IGNORE_INDEX = -100


@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Wether to force a specific system message.
    force_system_message: bool = False
    system_default: dict = None


class SFTTokenizer:
    """SFT Tokenizer."""

    def __init__(self, tokenizer_path: str, prompt_format: str):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        if HAVE_TRANSFORMERS:
            # Currently, only HuggingFace tokenizers are supported.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
        else:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-nano-v2":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_nano_v2_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "llama3p1":
            from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
                llama3p1_chat_template,  # lazy to avoid circular import
            )

            # "<|start_header_id|>assistant<|end_header_id|>\n\n" is the 4-token assistant prefix.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
                custom_chat_template=llama3p1_chat_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "identity":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=identity_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "default":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=tokenizer.chat_template,
                has_bos=tokenizer.bos_token_id is not None,
                has_system_role=True,
            )
        elif prompt_format == "qwen3p5-reasoning":
            # Qwen 3.5 reasoning format - always includes <think>...</think> block.
            # Assistant prefix "<|im_start|>assistant\n" is 3 tokens.
            # Loss is computed starting from <think>, so the model learns to predict it.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                custom_chat_template=qwen3p5_reasoning_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "qwen3p5-no-reasoning":
            # Qwen 3.5 non-reasoning format - no thinking tokens.
            # Assistant prefix "<|im_start|>assistant\n" is 3 tokens.
            # Loss is computed starting from the answer content.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                custom_chat_template=qwen3p5_no_reasoning_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "qwen3p5-tools":
            # Qwen 3.5 reasoning + tool-calling format. Uses the official Qwen3.5 chat
            # template (with {% generation %} markers) which renders reasoning (<think>),
            # tool schemas (tools= kwarg), assistant tool_calls, and role:tool results.
            # The assistant loss mask comes from return_assistant_tokens_mask, so the
            # scalar assistant_prefix_len is unused for this format.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                custom_chat_template=_load_qwen3p5_tools_template(),
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "qwen3p5-tools-nothink":
            # Same as qwen3p5-tools (tool calling + {% generation %} loss mask) but
            # renders NO <think> tags at all (not even an empty block). gen_body is the
            # assistant content only; any residual <think>...</think> in the data is
            # stripped. Loss is on assistant content only, no reasoning scaffold.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                custom_chat_template=_load_qwen3p5_tools_nothink_template(),
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "qwen3p5-tools-lastturn":
            # Same as qwen3p5-tools but the template emits {% generation %} markers ONLY on
            # the last assistant turn -> loss on the target (final) turn only. Used for the
            # turn-unrolled agentic data (train intermediate-turn reasoning with stripped
            # context, matching stock-template inference).
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                custom_chat_template=_load_qwen3p5_tools_lastturn_template(),
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "qwen3p5-tools-maskthink":
            # Same as qwen3p5-tools but the {% generation %} markers wrap ONLY the post-
            # </think> answer (not the <think>...</think> reasoning). Loss is on the final
            # answer while the (English) reasoning is rendered as context but MASKED. Used
            # for the Finnish enreason data (fi-question / English-reason / fi-answer).
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                custom_chat_template=_load_qwen3p5_tools_maskthink_template(),
                has_bos=False,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format

    def tokenize_conversation(
        self,
        conversation: List[Dict],
        return_target: bool,
        add_generation_prompt: bool,
        tools: Optional[List[Dict]] = None,
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
            tools (Optional[List[Dict]]): Tool/function schemas, passed to the chat
                template's ``tools`` argument. Only used by formats whose template
                supports tool calling (e.g. "qwen3p5-tools"). ``None`` renders exactly
                like a conversation with no tools.
        """
        # The qwen3p5-tools format derives the assistant loss mask from the template's
        # {% generation %} markers via return_assistant_tokens_mask, because its render
        # is context-dependent (reasoning history, multi-turn tool grouping) and cannot
        # be reconstructed by the per-turn re-tokenization path used below.
        if self._prompt_format in ("qwen3p5-tools", "qwen3p5-tools-nothink", "qwen3p5-tools-lastturn", "qwen3p5-tools-maskthink"):
            return self._tokenize_conversation_with_generation_mask(
                conversation, return_target, add_generation_prompt, tools
            )

        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = np.array(
            self._tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_assistant_token_mask=False,
                return_dict=True,
                chat_template=self._prompt_config.custom_chat_template,
            )["input_ids"]
        )

        if not return_target:
            return tokens

        target = tokens.copy()

        # When using the default prompt format, we do not replace any tokens with IGNORE_INDEX.
        # Instead, all token losses will be used for simplicity.
        if self._prompt_format == "default":
            return tokens, target

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):

            if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
                raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            if turn["role"].lower() == "assistant":
                assert conversation[turn_idx - 1]["role"].lower() in ("user", "tool")

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, return_dict=True,
                chat_template=self._prompt_config.custom_chat_template
            )["input_ids"]

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]
            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user", "tool"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            else:
                raise ValueError("Wrong role value.")

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    def _tokenize_conversation_with_generation_mask(
        self,
        conversation: List[Dict],
        return_target: bool,
        add_generation_prompt: bool,
        tools: Optional[List[Dict]] = None,
    ):
        """Tokenize using the chat template's {% generation %} markers for masking.

        The loss mask is taken directly from ``return_assistant_tokens_mask`` rather than
        reconstructed by per-turn re-tokenization. This is required for templates whose
        rendering depends on full conversation context (Qwen3.5 reasoning history and
        multi-turn tool grouping).
        """
        out = self._tokenizer.apply_chat_template(
            conversation,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=self._prompt_config.custom_chat_template,
        )
        tokens = np.array(out["input_ids"])

        if not return_target:
            return tokens

        assistant_mask = np.array(out["assistant_masks"], dtype=bool)
        if assistant_mask.shape != tokens.shape:
            raise ValueError(
                "assistant_masks shape mismatch: "
                f"{assistant_mask.shape} vs tokens {tokens.shape}"
            )
        if not assistant_mask.any():
            raise ValueError(
                f"no assistant tokens found in conversation (empty loss mask): {conversation}"
            )
        target = tokens.copy()
        target[~assistant_mask] = IGNORE_INDEX
        return tokens, target

    def text_to_ids(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(
                text, return_target=False, add_generation_prompt=True
            ).tolist()

        return self._tokenizer.encode(text)

    def tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def ids_to_text(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def ids_to_tokens(self):
        """Converts ids to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def text_to_tokens(self):
        """Converts text to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def tokens_to_text(self):
        """Converts tokens to text."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    def add_special_tokens(self):
        """Add special tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    @property
    def pad_id(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def bos_id(self):
        """Beginning of sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
