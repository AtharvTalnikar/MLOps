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
| 512        | 8          | 0           | 200   | 819,200    | 8.67     | 94,537      |
| 1024       | 8          | 0           | 200   | 1,638,400  | 7.95     | 206,027     |

**Interpretation:** When streaming the IMDb dataset, increasing `block_size` from **512 to 1024** more than **doubled throughput** — from roughly **94k tokens/sec** to **206k tokens/sec**.  
The improvement comes from packing more tokens per block, which reduces iteration overhead and the number of concatenation steps needed per batch.  
Larger blocks make token grouping more efficient for language-model training, though they require slightly higher memory per batch.  
The pipeline remained stable with `num_workers = 0` on Windows, confirming that even single-process streaming can achieve strong performance gains through better token packing.

---

**In summary:**  
We replaced the ad-hoc stream logic with a structured `IterableDataset`, added a proper collate step and padding rule, ensured Windows stability by running single-process (`num_workers=0`), and introduced a throughput benchmark to quantify performance.


## Dataset Update and Quick Preview

- Default dataset: **IMDb** (streaming).
- The pipeline uses the **`text`** field only (labels are ignored for LM).  
- A small **preview cell** prints a few review snippets without consuming the main iterator.

**Where to configure (in `Lab2.ipynb`):**
- `dataset_name = "imdb"`
- `dataset_config = None`
- `dataset_split = "train"`
- `streaming_mode = True`

**Where it’s used:**
- **Dataset configuration** cell (sets the variables above)
- **Quick preview (non-destructive)** cell (loads a separate small iterator)
- **Stream factory** (`make_stream()`), which builds the main iterator for the loader

### Switching Datasets
- Wikitext-2 (raw):
  - `dataset_name = "wikitext"`
  - `dataset_config = "wikitext-2-raw-v1"`
  - `dataset_split = "train"`
- Local text files:
  - Replace load calls with `load_dataset("text", data_files={"train": "path/to/file.txt"}, streaming=True)`
- CSV files:
  - `load_dataset("csv", data_files={"train": "path/to/data.csv"}, streaming=True)`

### Notes
- IMDb examples have `text` and `label`; the LM pipeline uses only `text` for tokenization and treats it as unlabeled language modeling data.
- The preview cell uses a separate iterator, so it won’t disturb training data iteration.
