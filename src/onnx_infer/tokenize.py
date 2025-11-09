import time
from typing import Any

try:
    from transformers import LlamaTokenizerFast
except Exception:
    LlamaTokenizerFast = None


def mask_multichar_chinese_tokens(tokenizer: Any):
    vocab = getattr(tokenizer, "vocab", None)
    if vocab is None:
        vocab = tokenizer.get_vocab()
    multichar_tokens = {
        token for token in vocab.keys()
        if isinstance(token, str) and len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer):
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs):
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            for token in tokens:
                clean_token = token.replace("▁", "")
                if clean_token in self.multichar_tokens:
                    processed.extend(list(clean_token))
                else:
                    processed.append(token)
            return processed

        def __call__(self, text: str, **kwargs):
            tokens = self.tokenize(text, **kwargs)
            return self.tokenizer.convert_tokens_to_ids(tokens)

    return CharTokenizerWrapper(tokenizer)


def load_tokenizer(models_dir: str):
    if LlamaTokenizerFast is None:
        raise RuntimeError("transformers.LlamaTokenizerFast is required for tokenization")
    t0 = time.perf_counter()
    tok = LlamaTokenizerFast.from_pretrained(models_dir)
    wrapped = mask_multichar_chinese_tokens(tok)
    dt = time.perf_counter() - t0
    print(f"分词器加载耗时: {dt:.3f}s")
    return wrapped

__all__ = ["load_tokenizer", "mask_multichar_chinese_tokens"]