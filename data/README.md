# 4,000-request RedHatAI workload

`redhat8_4000.jsonl` contains **4,000 requests: 500 per subset, drawn by repetition from 724 source records and 644 distinct prompt strings**. It does not contain 4,000 independent questions. It is a latency/throughput workload, not a newly collected evaluation set.

The source is [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks), pinned to revision `2ae86affa2cb97a972b7fc681dd51c04fbff083e`. The eight files are the eight entries in that revision's dataset card, not the first eight files in an alphabetical repository listing. The later `tool_call.jsonl` file is not included.

| Subset | Source records | Requests | Copies per source record |
|---|---:|---:|---|
| HumanEval | 164 | 500 | First 8 records: 4; remaining 156: 3 |
| math_reasoning | 80 | 500 | First 20 records: 7; remaining 60: 6 |
| qa | 80 | 500 | First 20 records: 7; remaining 60: 6 |
| question | 80 | 500 | First 20 records: 7; remaining 60: 6 |
| rag | 80 | 500 | First 20 records: 7; remaining 60: 6 |
| summarization | 80 | 500 | First 20 records: 7; remaining 60: 6 |
| translation | 80 | 500 | First 20 records: 7; remaining 60: 6 |
| writing | 80 | 500 | First 20 records: 7; remaining 60: 6 |

`question.jsonl` and `writing.jsonl` are byte-identical upstream files. Their 80 shared prompts explain the difference between 724 source records and 644 unique prompt strings. They are retained as separate named subsets to preserve the requested source structure. `source_category` preserves each upstream category: the file named `writing` is not exclusively category `writing`.

For each subset slot from 0 through 499, the builder visits subsets in the table order and takes source record `slot % source_count`. Thus every group of eight requests contains one from each subset. There is no random shuffle or hidden selection by length or expected answer.

Each request retains the exact `prompt` string and has a unique `id`, `request_index`, `subset`, `source_id`, `source_file`, `source_record_index`, `source_record_sha256`, `prompt_sha256`, `copy_index`, and `repetition_index`. `source_id` is the original `task_id` or `question_id`; identify a source record by `(subset, source_record_index)`, since upstream IDs can overlap across files. Copy and repetition indices are zero-based aliases: 0 means the first occurrence of that source row. The complete original records, including their existing references, solutions and tests, remain unchanged under `source/`; they are not added to the model prompt.

No prompt is shortened, tokenized, padded, deduplicated, or synthesized. Characters mean Unicode code points, not model tokens. The maximum is **6,847 characters**. The manifest contains overall/per-subset min, mean, nearest-rank p50/p95/p99, max, and threshold counts. Running machines must obtain token lengths with the actual target tokenizer and check context capacity; insufficient context must produce an explicit error, not silent truncation.

The intended runtime wrapper preserves the earlier experiment's template policy:

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": row["prompt"]}],
    tokenize=True,
    add_generation_prompt=True,
)
```

No system message is inserted. Model-specific special tokens and the exact serialized chat template are supplied by the target tokenizer on the running machine and must be recorded there. Natural stopping and output caps belong to the experiment configuration, not this raw dataset.

Rebuild and verify from the repository root, without network access, models, GPUs, or third-party Python packages:

```bash
python scripts/build_dataset.py
python scripts/build_dataset.py --check
```

The builder validates pinned checksums before using any source file. `--check` writes nothing and checks both products byte for byte. The data uses UTF-8 without BOM and LF newlines. Preserve source file bytes and builder line endings when packaging or checking out the repository, because the manifest also records the builder's SHA-256.

See `redhat8_4000.manifest.json` for hashes, counts and reproducibility details, and [NOTICE.md](NOTICE.md) for attribution and licenses. At batch capacity 512, 4,000 requests include a final partial cohort of 416; that tail must not be counted as actual BS512.
