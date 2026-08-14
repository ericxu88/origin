"""Import the HC3 corpus (Hello-SimpleAI/HC3) into the Origin dataset schema.

HC3 pairs real human answers with ChatGPT answers to the same questions
(reddit_eli5, open_qa, wiki_csai, medicine, finance). This script builds a
balanced detection corpus:

- one human + one ChatGPT answer per sampled question, sharing a ``group_id``
  so a pair can never straddle train/test splits (topic leakage guard);
- span-labelled mixed documents spliced from *reserved* pairs (used only for
  mixing), giving real localization ground truth.

Usage::

    uv run python scripts/import_hc3.py --pairs 250 --mixed 40

Requires network on first run (downloads ``all.jsonl`` from the HF hub into
the local cache). Output: ``data/hc3/documents.jsonl`` (+ README).

Caveats recorded in the output README: all AI text is one model family
(gpt-3.5), so unseen-family evaluation is not meaningful on this corpus; HC3
human answers carry detokenization artifacts (spaced punctuation) that make
the classes easier to separate than in the wild.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from origin_ml.datasets import (
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    build_mixed_document,
    write_jsonl,
)
from origin_ml.text.segmentation import segment_sentences

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "hc3"
SEED = 20260813

MIN_CHARS = 200
MAX_CHARS = 4000
MIN_SENTENCES = 3

GENERATION = GenerationInfo(
    model_family="gpt",
    model_name="gpt-3.5-turbo",
    provider="openai",
    prompt_summary="answer the question (HC3 collection, late 2022)",
)


def usable(text: str) -> bool:
    text = text.strip()
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return False
    return len(segment_sentences(text)) >= MIN_SENTENCES


def load_pairs() -> list[dict[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    pairs: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            humans = [a.strip() for a in row.get("human_answers", []) if usable(a)]
            bots = [a.strip() for a in row.get("chatgpt_answers", []) if usable(a)]
            if humans and bots:
                pairs.append(
                    {
                        "source": row["source"],
                        "index": str(row["index"]),
                        "question": row["question"].strip(),
                        "human": humans[0],
                        "ai": bots[0],
                    }
                )
    return pairs


def stratified_sample(
    pairs: list[dict[str, str]], count: int, rng: random.Random
) -> list[dict[str, str]]:
    """Sample roughly evenly across HC3 sources for topical diversity."""
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        by_source[pair["source"]].append(pair)
    for bucket in by_source.values():
        rng.shuffle(bucket)
    sampled: list[dict[str, str]] = []
    sources = sorted(by_source)
    while len(sampled) < count and any(by_source[s] for s in sources):
        for source in sources:
            if by_source[source] and len(sampled) < count:
                sampled.append(by_source[source].pop())
    return sampled


def build_records(pairs: list[dict[str, str]], n_mixed: int, rng: random.Random) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    mixing_reserve, standalone = pairs[:n_mixed], pairs[n_mixed:]

    for pair in standalone:
        group = f"hc3-{pair['source']}-{pair['index']}"
        prompt_id = group
        generation = GENERATION.model_copy(update={"prompt_id": prompt_id})
        records.append(
            DocumentRecord(
                id=f"{group}-human",
                text=pair["human"],
                label=DocLabel.HUMAN,
                source=f"hc3-{pair['source']}",
                group_id=group,
                meta={"question": pair["question"][:200]},
            )
        )
        records.append(
            DocumentRecord(
                id=f"{group}-ai",
                text=pair["ai"],
                label=DocLabel.AI,
                source=f"hc3-{pair['source']}",
                group_id=group,
                generation=generation,
                meta={"question": pair["question"][:200]},
            )
        )

    for i, pair in enumerate(mixing_reserve):
        group = f"hc3-mixed-{i:03d}"
        records.append(
            build_mixed_document(
                doc_id=group,
                human_text=pair["human"],
                ai_text=pair["ai"],
                generation=GENERATION.model_copy(update={"prompt_id": f"hc3-{pair['source']}-{pair['index']}"}),
                group_id=group,
                source=f"hc3-{pair['source']}",
                seed=SEED + i,
            )
        )
    return sorted(records, key=lambda r: r.id)


README = """\
# HC3 detection corpus (imported)

Built by `scripts/import_hc3.py` from **Hello-SimpleAI/HC3** (Human ChatGPT
Comparison Corpus; CC-BY-SA — see the dataset card on Hugging Face). Regenerate:

    uv run python scripts/import_hc3.py --pairs {pairs} --mixed {mixed}

- One real human answer + one ChatGPT (gpt-3.5, late 2022) answer per sampled
  question; each pair shares a `group_id`, so splits never separate a topic pair.
- `hc3-mixed-*` documents splice sentences from *reserved* pairs (not emitted
  standalone) with exact span labels for localization evaluation.

Caveats:
- Single AI family (gpt-3.5): unseen-family evaluation is not meaningful here.
- HC3 human answers contain detokenization artifacts (spaced punctuation) that
  make separation easier than in the wild; treat metrics as an upper bound.
- Not committed to git (licensing + size); regenerate locally as needed.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=250, help="standalone human/AI pairs")
    parser.add_argument("--mixed", type=int, default=40, help="mixed documents from reserved pairs")
    args = parser.parse_args()

    rng = random.Random(SEED)
    all_pairs = load_pairs()
    print(f"usable question pairs in HC3: {len(all_pairs)}")
    sampled = stratified_sample(all_pairs, args.pairs + args.mixed, rng)
    records = build_records(sampled, args.mixed, rng)

    out = OUT_DIR / "documents.jsonl"
    count = write_jsonl(out, records)
    (OUT_DIR / "README.md").write_text(
        README.format(pairs=args.pairs, mixed=args.mixed), encoding="utf-8"
    )
    labels = {label: sum(1 for r in records if r.label is label) for label in DocLabel}
    sources = sorted({r.source for r in records})
    print(f"wrote {count} records to {out}")
    print(f"  labels : {', '.join(f'{k.value}={v}' for k, v in labels.items())}")
    print(f"  sources: {', '.join(sources)}")


if __name__ == "__main__":
    main()
