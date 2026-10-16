"""Text generation for expansions. HF `generate` backend (works on a Kaggle T4; vLLM comes on Day 3).

torch/transformers are imported lazily so the rest of the repo and its tests run without a GPU.
"""


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", dtype: str = "float16",
                 backend: str = "hf", batch_size: int = 16):
        if backend != "hf":
            raise NotImplementedError("only the 'hf' backend exists so far; vLLM is a Day 3 addition")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name, self.batch_size = model_name, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map="auto" if torch.cuda.is_available() else None
        ).eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def generate(self, messages_list: list[list[dict]], n: int = 1, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, max_new_tokens: int = 128, seed: int = 0) -> list[list[str]]:
        """Return n completions per conversation. temperature == 0 means greedy (n must be 1)."""
        import torch

        if temperature <= 0 and n != 1:
            raise ValueError("greedy decoding returns one completion; use temperature > 0 for n > 1")
        out: list[list[str]] = []
        for b in range(0, len(messages_list), self.batch_size):
            batch = messages_list[b:b + self.batch_size]
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for m in batch]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, num_return_sequences=n)
            else:  # clear the model's default sampling settings so greedy is really greedy
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            torch.manual_seed(seed * 100003 + b)
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            k = n if temperature > 0 else 1
            out.extend(texts[i * k:(i + 1) * k] for i in range(len(batch)))
        return out
