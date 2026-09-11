"""Text generation backends. The brain only needs `generate(messages) -> str`.

Each backend also fills `self.stats` after a call: prompt and generated token counts, wall time, the KV cost per token
implied by the model's layout, and (Qwen) the GPU memory picture. The agent turns that into the gauge in the window.
"""
from __future__ import annotations
import json, os, time
from typing import Protocol


class Backend(Protocol):
    stats: dict
    def generate(self, messages: list[dict], temperature: float, max_new_tokens: int, seed: int) -> str: ...


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_value: int = 2) -> int:
    """K and V, per layer, per KV head, per head dim. Qwen3-8B (36, 8, 128) at bf16 ≈ 147 KB per token."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value


class EchoBackend:
    """No model: a canned emotion + reply built from the user text, varied enough to move the mood around."""
    RULES = [   # keywords -> (emotion, move description; "" = no move, the body drifts)
        (("ugly", "hate", "stupid", "shut up"), "That stung. I feel hurt and a little angry, and I want to push back even though I know I should not.",
         "furrow brows, clench fists, grit teeth"),
        (("ghost", "killer", "careful", "behind you"), "My heart is racing; I am startled and genuinely scared, and I feel very small right now.",
         "eyes go wide, mouth opens, hands come up"),
        (("sorry", "not for you", "goodbye", "leave"), "I feel sad and a bit lonely, disappointed that this is ending so soon.",
         "eyes well up, mouth trembles, fists clench"),
        (("thank", "great", "love", "well done"), "I am glad and quietly proud; it warms me that this landed well.",
         "smile broadly with bright eyes"),
        (("cat", "hat", "hold"), "I am delighted and a little proud to show something off.",
         "hold up a cartoon cat with both hands and grin"),
        (("why", "how", "what", "?"), "I am curious and interested, with a small worry that I might have misread the question.",
         "one eyebrow up, finger to the chin, eyes drift sideways"),
        (("calm", "relax", "quiet"), "I feel settled and content, comfortable in the quiet.", ""),
    ]

    def __init__(self):
        self.stats: dict = {}

    def generate(self, messages, temperature=0.7, max_new_tokens=300, seed=42) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = user.lower()
        if low.startswith("[internal event"):
            out = json.dumps({"emotion": "I am a bit bored and drowsy, drifting, mildly curious what comes next.",
                              "move": {"description": "", "hold": "turn"}, "reply": ""})
        else:
            emotion, move = next(((e, mv) for keys, e, mv in self.RULES if any(k in low for k in keys)),
                                 ("I am cheerful and eager to help, pleased to be talking with someone.", ""))
            out = json.dumps({"emotion": emotion, "move": {"description": move, "hold": "turn"},
                              "reply": f"(echo) You said: {user}. I have no model behind me, so that is all I can offer."})
        prompt_tokens = int(sum(len(m["content"].split()) for m in messages) * 1.3)
        self.stats = {"prompt_tokens": prompt_tokens, "new_tokens": int(len(out.split()) * 1.3), "seconds": 0.0,
                      "kv_bytes_per_token": kv_bytes_per_token(36, 8, 128), "model": "echo (KV cost assumed as Qwen3-8B)", "gpu": None}
        return out


class QwenBackend:
    """Qwen3 via transformers, 4-bit. Loaded lazily on first use so the server starts fast."""
    def __init__(self, model_id: str = "Qwen/Qwen3-8B", device: str = "cuda:0"):
        self.model_id, self.device = model_id, device
        self._model = self._tok = None
        self.stats: dict = {}

    def _load(self):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 8 GB card: avoid fragmentation OOM
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb,
                                                           device_map=self.device, dtype=torch.bfloat16)
        self._model.eval()

    def _kv_per_token(self) -> int:
        c = self._model.config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return kv_bytes_per_token(c.num_hidden_layers, getattr(c, "num_key_value_heads", c.num_attention_heads), head_dim)

    def _gpu(self) -> dict | None:
        import torch
        if not torch.cuda.is_available():
            return None
        dev = torch.device(self.device)
        free, total = torch.cuda.mem_get_info(dev)
        return {"allocated": torch.cuda.memory_allocated(dev), "reserved": torch.cuda.memory_reserved(dev),
                "peak": torch.cuda.max_memory_allocated(dev), "total": total, "free": free}

    def generate(self, messages, temperature=0.7, max_new_tokens=300, seed=42) -> str:
        import torch
        if self._model is None:
            self._load()
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        torch.manual_seed(seed)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self._tok.eos_token_id, do_sample=temperature > 0)
        if temperature > 0:
            kw.update(temperature=temperature, top_k=20, top_p=0.8)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.no_grad():
                gen = self._model.generate(**inputs, **kw)
        except torch.cuda.OutOfMemoryError:
            # the card is full: free what we can and try once more with a shorter answer before giving up
            torch.cuda.empty_cache()
            kw["max_new_tokens"] = max(64, max_new_tokens // 2)
            with torch.no_grad():
                gen = self._model.generate(**inputs, **kw)
            self.oom_retries = getattr(self, "oom_retries", 0) + 1
        n_new = int(gen.shape[1] - n_prompt)
        out = self._tok.decode(gen[0][n_prompt:], skip_special_tokens=True)
        gpu = self._gpu()
        del inputs, gen
        torch.cuda.empty_cache()
        self.stats = {"prompt_tokens": n_prompt, "new_tokens": n_new, "seconds": time.time() - t0,
                      "kv_bytes_per_token": self._kv_per_token(), "model": self.model_id, "gpu": gpu,
                      "oom_retries": getattr(self, "oom_retries", 0)}
        return out
