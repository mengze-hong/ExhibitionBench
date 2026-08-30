"""Deterministic non-LLM baselines for the 500-pair ECD release.

BM25 measures within-sequence lexical coherence.  Each artwork is treated as a
document in the released object corpus and scored against the remaining items
in its sequence (leave one out); the sequence with the larger summed score is
selected.  Random choices and exact-score ties use the requested fixed seed.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from pathlib import Path


FIELDS = ("title", "description", "culture", "medium", "date", "department")


def tokens(item: dict) -> list[str]:
    text = " ".join(str(item.get(field, "")) for field in FIELDS)
    return re.findall(r"[a-z0-9]+", text.lower())


class CorpusBM25:
    def __init__(self, objects: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokens(obj) for obj in objects]
        self.avgdl = sum(map(len, docs)) / len(docs)
        document_frequency = collections.Counter(
            term for doc in docs for term in set(doc)
        )
        raw_idf = {
            term: math.log(len(docs) - freq + 0.5) - math.log(freq + 0.5)
            for term, freq in document_frequency.items()
        }
        average_idf = sum(raw_idf.values()) / len(raw_idf)
        self.idf = {
            term: 0.25 * average_idf if value < 0 else value
            for term, value in raw_idf.items()
        }
        self.docs = {
            obj["id"]: (collections.Counter(doc), len(doc))
            for obj, doc in zip(objects, docs)
        }

    def score_document(self, item: dict, query: list[str]) -> float:
        frequency, length = self.docs.get(
            item.get("id"),
            (collections.Counter(tokens(item)), len(tokens(item))),
        )
        norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
        return sum(
            self.idf.get(term, 0.0)
            * frequency[term]
            * (self.k1 + 1)
            / (frequency[term] + norm)
            for term in query
        )

    def sequence_coherence(self, sequence: list[dict]) -> float:
        total = 0.0
        for index, item in enumerate(sequence):
            query = [
                term
                for other_index, other in enumerate(sequence)
                if other_index != index
                for term in set(tokens(other))
            ]
            total += self.score_document(item, query)
        return total


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(samples: list[dict], method: str, objects: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    scorer = CorpusBM25(objects) if method == "bm25" else None
    correct = collections.Counter()
    total = collections.Counter()

    for sample in samples:
        level = f"L{sample['level']}"
        total[level] += 1
        if scorer is None:
            prediction_is_positive = rng.random() < 0.5
        else:
            positive = scorer.sequence_coherence(sample["positive"]["items"])
            negative = scorer.sequence_coherence(sample["negative"]["items"])
            prediction_is_positive = (
                positive > negative
                if positive != negative
                else rng.random() < 0.5
            )
        correct[level] += int(prediction_is_positive)

    accuracies = {
        f"pairaccc_{level}": round(correct[level] / total[level], 4)
        for level in ("L1", "L2", "L3", "L4")
    }
    return {
        "task": "ecd",
        "model": method,
        "protocol": "final-500",
        **accuracies,
        "macro_pairaccc": round(sum(accuracies.values()) / 4, 4),
        "n_samples": len(samples),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("bm25", "random"))
    parser.add_argument("--input", default="data/ecd_samples.jsonl")
    parser.add_argument("--objects", default="data/objects.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_jsonl(Path(args.input))
    objects = load_jsonl(Path(args.objects))
    result = evaluate(samples, args.method, objects, args.seed)
    output = Path(args.output or f"results/ecd_{args.method}_shot0.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
