from pathlib import Path

from minilab.base import BaseTokenizer
from minilab.checks import require
from minilab.registry import register_tokenizer


@register_tokenizer("hf")
class HFTokenizer(BaseTokenizer):
    """Thin wrapper around a local Hugging Face tokenizer directory.

    This lets imported HF checkpoints use Minilab's native datasets and trainers
    without teaching every training script about Transformers tokenizers.
    """

    def __init__(self, path="", vocab_size=0):
        self.path = str(path)
        self._vocab_size = int(vocab_size) if vocab_size else 0
        self._tokenizer = None
        self._state_base_dir = None

    @classmethod
    def from_pretrained(cls, path):
        tok = cls(str(Path(path).expanduser().resolve()))
        tok._ensure_loaded()
        tok._vocab_size = len(tok._tokenizer)
        return tok

    def _set_state_base_dir(self, path):
        self._state_base_dir = Path(path)

    def _resolved_path(self):
        path = Path(self.path).expanduser()
        if path.is_absolute():
            return path
        if self._state_base_dir is not None:
            return self._state_base_dir / path
        return path

    def _ensure_loaded(self):
        if self._tokenizer is not None:
            return
        require(self.path, "HF tokenizer path is empty")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HF tokenizer support requires transformers. "
                "Install with: python -m pip install -e \".[hf]\""
            ) from exc
        resolved_path = self._resolved_path()
        expected_vocab_size = self._vocab_size
        self._tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        actual_vocab_size = len(self._tokenizer)
        require(expected_vocab_size in (0, actual_vocab_size), (
            "HF tokenizer vocab_size mismatch: "
            f"state declares {expected_vocab_size}, but tokenizer at {resolved_path} has {actual_vocab_size}. "
            "Use the tokenizer directory that was saved with this Minilab tokenizer state."
        ))
        self._vocab_size = actual_vocab_size

    def encode(self, text):
        self._ensure_loaded()
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode_prompt(self, text):
        return self.encode_messages([{"role": "user", "content": text}])

    @staticmethod
    def _chat_template_ids(value):
        # Transformers 4.x returns a list here, while 5.x may return a
        # BatchEncoding even without return_tensors. Normalize both public
        # shapes before enforcing Minilab's tokenizer contract.
        if hasattr(value, "keys") and "input_ids" in value:
            value = value["input_ids"]
        require(
            type(value) is list
            and all(type(token_id) is int and token_id >= 0 for token_id in value),
            "HF chat template must produce a list of integer token ids",
        )
        return value

    def encode_messages(self, messages):
        self._ensure_loaded()
        messages = self._validated_messages(messages)
        if self._tokenizer.chat_template is None:
            return super().encode_messages(messages)
        ids = self._chat_template_ids(self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
        return ids

    def encode_supervised(self, messages, response):
        self._ensure_loaded()
        messages = self._validated_messages(messages)
        require(type(response) is str and response, (
            "supervised response must be a non-empty string"
        ))
        if self._tokenizer.chat_template is None:
            return super().encode_supervised(messages, response)

        prompt_ids = self.encode_messages(messages)
        full_ids = self._chat_template_ids(self._tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        ))
        require(full_ids[:len(prompt_ids)] == prompt_ids, (
            "HF supervised chat template must preserve the generation-prompt prefix"
        ))
        response_ids = full_ids[len(prompt_ids):]
        require(response_ids, "HF supervised chat template produced no assistant target tokens")
        return prompt_ids, response_ids

    def encode_agentic_continuation(
        self,
        messages,
        prefix_ids,
        assistant_ids,
        observation,
    ):
        self._ensure_loaded()
        messages = self._validated_messages(messages)
        require(messages[-1] == {"role": "user", "content": observation}, (
            "agentic continuation history must end with its observation"
        ))
        if self._tokenizer.chat_template is None:
            return super().encode_agentic_continuation(
                messages,
                prefix_ids,
                assistant_ids,
                observation,
            )
        return self.encode_messages(messages)

    def decode(self, ids):
        self._ensure_loaded()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def stop_token_ids(self):
        self._ensure_loaded()
        eos_token_id = self._tokenizer.eos_token_id
        if eos_token_id is None:
            return ()
        require(type(eos_token_id) is int and eos_token_id >= 0, (
            "HF tokenizer eos_token_id must be a non-negative integer"
        ))
        return (eos_token_id,)

    @property
    def vocab_size(self):
        if self._vocab_size == 0:
            self._ensure_loaded()
        return self._vocab_size

    def _get_state(self):
        vocab_size = self._vocab_size
        if vocab_size == 0 and self.path:
            vocab_size = self.vocab_size
        return {
            "type": "hf",
            "path": self.path,
            "vocab_size": vocab_size,
        }

    def _set_state(self, state):
        require(type(state) is dict, "HF tokenizer state must be a JSON object")
        require(set(state) == {"type", "path", "vocab_size"}, (
            "HF tokenizer state fields must be exactly: path, type, vocab_size"
        ))
        require(state["type"] == "hf", "HF tokenizer state has wrong type")
        require(type(state["path"]) is str and state["path"], "HF tokenizer path must be a non-empty string")
        require(type(state["vocab_size"]) is int and state["vocab_size"] > 0, (
            "HF tokenizer vocab_size must be a positive integer"
        ))
        self.path = state["path"]
        self._vocab_size = state["vocab_size"]
        self._tokenizer = None
