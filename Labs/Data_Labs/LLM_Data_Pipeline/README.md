# Changes Made in Lab 2 — LLM Data Pipeline (Streaming)

## What I Changed from the Original Lab

1. **Refactored the streaming iterator**
   - Rewrote the inline token-packing logic into a clean `StreamingLMIterableDataset` class.
   - The iterator now uses a rolling buffer to yield **uniform fixed-length `[B, T]` blocks** for training.

2. **Added a custom `collate_fn`**
   - Handles `attention_mask` and `labels` explicitly:
     ```python
     def collate_fn(batch):
         x = torch.stack(batch)
         return {"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)}
     ```
   - Ensures each batch is ready for causal-language-model training.

3. **Set GPT-2 padding policy**
   - Defined `tokenizer.pad_token = tokenizer.eos_token` so partial blocks pad safely.

4. **Stable configuration for Windows**
   - Used `num_workers = 0` to avoid multiprocessing crashes with streaming datasets.
   - Keeps the code stable in notebooks while remaining compatible with multi-worker sharding on Linux.

5. **Throughput measurement added**
   - Implemented a `measure_throughput(loader, steps=200)` function to track data-pipeline speed.
   - **Result:** `Steps: 200, Tokens: 1,638,400, Time: 17.88 s → ~91,634 tokens/sec`.

| block_size | batch_size | num_workers | Steps | Tokens    | Time (s) | ~Tokens/sec |
|-----------:|-----------:|------------:|------:|-----------:|---------:|------------:|
| 1024       | 8          | 0           | 200   | 1,638,400  | 17.88    | 91,634      |
| 2048       | 8          | 0           | 146   | 2,390,016  | 24.20    | 98,774      |

**Interpretation:** Increasing `block_size` from **1024 → 2048** improved throughput (~**+7.8%**) by reducing Python overhead and packing waste. Trade-offs: slightly higher per-batch memory and latency, but same single-process configuration (`num_workers=0`) remained stable on Windows.


---

**In summary:**  
We replaced the ad-hoc stream logic with a structured `IterableDataset`, added a proper collate step and padding rule, ensured Windows stability by running single-process (`num_workers=0`), and introduced a throughput benchmark to quantify performance.

