#!/usr/bin/env python3
"""Rebuild or verify the bundled 4,000-request workload, entirely offline.

Python standard library only. Source prompt strings are never modified,
tokenized, filtered, or truncated. These are requests, not independent tasks.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

REPO = "RedHatAI/speculator_benchmarks"
REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
SUBSETS = ("HumanEval", "math_reasoning", "qa", "question", "rag",
           "summarization", "translation", "writing")
EXPECTED = {
    "HumanEval": (164, "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"),
    "math_reasoning": (80, "b73b686f1c585eef7db1306564688da8668ef633ff8e58d55949aac04db8c9b0"),
    "qa": (80, "17c7c2fb9f36abd16e3e2a8ab5487ae678e9762d96cb9736eec442cbaba9f506"),
    "question": (80, "9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4"),
    "rag": (80, "d1644a209c444e1dd7ff65bf3ce9e8a4daa5d609ac93cbf4f6bebc10400d8d91"),
    "summarization": (80, "e7d2945062f77aae25553f7f7531deefc358f5d0b87b9c623e6ba8ba555ff1be"),
    "translation": (80, "bd27352726c318b72a263158672b0ff0c15feb459ae735f723b27623a0ba595d"),
    "writing": (80, "9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4"),
}
PER_SUBSET = 500


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(obj, pretty=False):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       indent=2 if pretty else None,
                       separators=None if pretty else (",", ":")) + "\n").encode("utf-8")


def char_stats(prompts):
    values = sorted(len(prompt) for prompt in prompts)
    return {
        "unit": "Unicode code points, not tokenizer tokens",
        "count": len(values), "min": values[0], "max": values[-1],
        "mean": sum(values) / len(values),
        "p50": values[math.ceil(len(values) * .50) - 1],
        "p95": values[math.ceil(len(values) * .95) - 1],
        "p99": values[math.ceil(len(values) * .99) - 1],
        "quantile_method": "nearest_rank",
        "over_2048": sum(value > 2048 for value in values),
        "over_4096": sum(value > 4096 for value in values),
        "over_8192": sum(value > 8192 for value in values),
    }


def build(source_dir):
    source, source_files = {}, {}
    for subset in SUBSETS:
        filename = subset + ".jsonl"
        payload = (source_dir / filename).read_bytes()
        count, expected_hash = EXPECTED[subset]
        if digest(payload) != expected_hash:
            raise ValueError("Pinned source checksum mismatch: " + filename)
        records = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines()
                   if line.strip()]
        if len(records) != count:
            raise ValueError("Unexpected source count: " + filename)
        if any(not isinstance(row.get("prompt"), str) or not row["prompt"] for row in records):
            raise ValueError("Missing or empty source prompt: " + filename)
        source[subset] = records
        source_files[filename] = {
            "sha256": expected_hash, "bytes": len(payload), "records": count,
            "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}",
            "license_as_declared_by_upstream_card": "MIT" if subset == "HumanEval" else "Apache-2.0",
            "source_prompt_characters": char_stats(row["prompt"] for row in records),
        }
    requests = []
    # Round robin gives each contiguous group of eight requests one per subset.
    # Each subset cycles its original row order; no RNG or platform-dependent shuffle.
    for slot in range(PER_SUBSET):
        for subset in SUBSETS:
            records = source[subset]
            source_index = slot % len(records)
            copy_index = slot // len(records)
            row = records[source_index]
            source_id = row.get("task_id", row.get("question_id", source_index))
            index = len(requests)
            requests.append({
                "id": f"redhat8-4000-{index:04d}", "request_index": index,
                "subset": subset, "prompt": row["prompt"],
                "prompt_sha256": digest(row["prompt"].encode("utf-8")),
                "source_id": source_id, "source_file": subset + ".jsonl",
                "source_record_index": source_index,
                "source_record_sha256": digest(json_bytes(row)),
                "source_category": row.get("category"),
                "copy_index": copy_index, "repetition_index": copy_index,
                "subset_request_index": slot,
            })
    if len(requests) != 4000 or len({r["id"] for r in requests}) != 4000:
        raise ValueError("Request count or request ID invariant failed")
    payload = b"".join(json_bytes(row) for row in requests)
    frequencies = Counter(row["prompt_sha256"] for row in requests)
    selected_sources = Counter((row["subset"], row["source_record_index"]) for row in requests)
    by_subset = {}
    for subset in SUBSETS:
        selected = [row for row in requests if row["subset"] == subset]
        freq = Counter(row["source_record_index"] for row in selected)
        by_subset[subset] = {
            "requests": len(selected), "source_records_used": len(freq),
            "distinct_prompt_strings": len({row["prompt_sha256"] for row in selected}),
            "added_repetitions": len(selected) - len(freq),
            "copies_per_source_record_histogram": dict(sorted(Counter(freq.values()).items())),
            "source_categories": dict(sorted(Counter(str(row["source_category"]) for row in selected).items())),
            "request_prompt_characters": char_stats(row["prompt"] for row in selected),
        }
    overlaps = []
    for i, left in enumerate(SUBSETS):
        for right in SUBSETS[i + 1:]:
            shared = set(row["prompt"] for row in source[left]) & set(row["prompt"] for row in source[right])
            if shared:
                overlaps.append({"left": left, "right": right, "distinct_shared_prompts": len(shared),
                                 "byte_identical_source_files": EXPECTED[left][1] == EXPECTED[right][1]})
    manifest = {
        "schema_version": 1, "dataset_name": "redhat8_4000",
        "source_repository": REPO, "source_revision": REVISION,
        "source_card_url": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
        "source_files": source_files, "subset_order": list(SUBSETS),
        "request_count": len(requests), "requests_per_subset": PER_SUBSET,
        "source_records_used": len(selected_sources),
        "distinct_prompt_strings": len(frequencies),
        "added_repetitions_of_source_records": len(requests) - len(selected_sources),
        "requests_beyond_first_distinct_prompt": len(requests) - len(frequencies),
        "prompt_frequency_histogram": dict(sorted(Counter(frequencies.values()).items())),
        "cross_subset_prompt_overlap": overlaps, "subsets": by_subset,
        "prompt_characters": char_stats(row["prompt"] for row in requests),
        "count_interpretation": "4000 requests from 724 source records and 644 distinct prompt strings; not 4000 independent tasks",
        "selection_policy": "For slot 0..499, visit the eight subsets in subset_order; use source row slot modulo source row count",
        "copy_index_policy": "Zero-based: floor(subset_request_index / source row count); repetition_index is its explicit alias",
        "source_record_sha256_policy": "SHA256 of UTF-8 sorted-key compact JSON plus LF, not source file line bytes",
        "prompt_policy": {
            "content_unchanged": True, "truncation": False, "length_filtering": False,
            "tokenized": False, "padding": False,
            "runtime_template": "target_tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True)",
            "system_message": None,
            "context_policy": "Runtime must tokenize with its actual target tokenizer, report lengths, and reject insufficient context rather than silently truncate",
        },
        "output": {"filename": "redhat8_4000.jsonl", "sha256": digest(payload), "bytes": len(payload),
                   "encoding": "UTF-8 without BOM; sorted-key compact JSON; LF line terminators"},
        "builder": {"path": "scripts/build_dataset.py", "sha256": digest(Path(__file__).read_bytes()),
                    "offline": True, "dependencies": "Python standard library"},
    }
    return payload, json_bytes(manifest, pretty=True), manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=root / "data/source")
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--check", action="store_true", help="Verify byte-for-byte; write nothing")
    args = parser.parse_args()
    payload, metadata, manifest = build(args.source_dir)
    products = {"redhat8_4000.jsonl": payload, "redhat8_4000.manifest.json": metadata}
    if args.check:
        for filename, expected in products.items():
            path = args.output_dir / filename
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit("REPRODUCIBILITY_CHECK_FAILED: " + str(path))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in products.items():
            (args.output_dir / filename).write_bytes(data)
    print(json.dumps({"status": "verified" if args.check else "built",
                      "requests": manifest["request_count"],
                      "distinct_prompts": manifest["distinct_prompt_strings"],
                      "sha256": manifest["output"]["sha256"],
                      "max_prompt_characters": manifest["prompt_characters"]["max"]}))


if __name__ == "__main__":
    main()
