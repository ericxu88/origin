"""Import the MAGE corpus (yaful/DeepfakeTextDetect) into the Origin schema.

MAGE ("Deepfake Text Detection in the Wild") contains human text and text from
27 generators across 10 domains, plus a dedicated **GPT-4 out-of-distribution
testbed** (unseen model family AND unseen domain, with a paraphrase-attacked
variant). This script builds:

- ``data/mage/documents.jsonl``      — balanced multi-family training corpus
  (human + machine across model families/domains) + span-labelled mixed docs;
- ``data/mage/ood_gpt4.jsonl``       — the GPT-4 OOD testbed (evaluation only);
- ``data/mage/ood_gpt4_para.jsonl``  — its paraphrase-attacked variant;
- ``data/combined/documents.jsonl``  — MAGE + HC3 merged (when HC3 exists).

Usage::

    uv run python scripts/import_mage.py --machine 500 --human 500 --mixed 30

Leakage note: MAGE provides no pairing between human rows and the machine rows
derived from them, so cross-split contamination via shared source articles
cannot be ruled out at the row level; heavy subsampling (~1k of 320k rows)
makes collisions unlikely. The GPT-4 OOD sets share no domain or family with
training data by construction.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from origin_ml.datasets import (
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    build_mixed_document,
    read_jsonl,
    write_jsonl,
)
from origin_ml.text.segmentation import segment_sentences

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "mage"
COMBINED = ROOT / "data" / "combined" / "documents.jsonl"
HC3 = ROOT / "data" / "hc3" / "documents.jsonl"
SEED = 20260813

MIN_CHARS = 200
MAX_CHARS = 4000
MIN_SENTENCES = 3

# Normalize MAGE's raw model identifiers into families.
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"davinci|gpt-3\.5|trubo|turbo", "gpt-3.5"),
    (r"glm", "glm"),
    (r"llama|^_?(7|13|30|65)b", "llama"),
    (r"flan_t5", "flan-t5"),
    (r"opt", "opt"),
    (r"gpt_j|gpt_neox|neox", "eleutherai"),
    (r"bloom|t0", "bigscience"),
]


def model_family(raw_model: str) -> str:
    lowered = raw_model.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, lowered):
            return family
    return "other"


def parse_src(src: str) -> tuple[str, str | None]:
    """Return (domain, raw_model_or_None_for_human)."""
    if src.endswith("_human"):
        return src.removesuffix("_human"), None
    match = re.match(r"([a-z0-9_]+?)_machine_[a-z]+_(.+)", src)
    if match:
        return match.group(1), match.group(2)
    return src, "unknown"


def usable(text: str) -> bool:
    text = text.strip()
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return False
    return len(segment_sentences(text)) >= MIN_SENTENCES


def read_csv(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for row in reader:
            if len(row) >= 3:
                rows.append((row[0].strip(), row[1], row[2]))
    return rows


def _record(
    doc_id: str, text: str, label: DocLabel, source: str, family: str | None, model: str | None
) -> DocumentRecord:
    generation = None
    if label is DocLabel.AI:
        generation = GenerationInfo(
            model_family=family or "unknown",
            model_name=model or "unknown",
            prompt_summary="MAGE generation (see dataset card)",
        )
    return DocumentRecord(
        id=doc_id,
        text=text,
        label=label,
        source=source,
        group_id=doc_id,
        generation=generation,
    )


def build_training_corpus(
    rows: list[tuple[str, str, str]], n_machine: int, n_human: int, n_mixed: int, rng: random.Random
) -> list[DocumentRecord]:
    machine_by_family: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    human_by_domain: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    rng.shuffle(rows)
    for text, label, src in rows:
        if not usable(text):
            continue
        domain, raw_model = parse_src(src)
        if label == "1":
            human_by_domain[domain].append((text, domain, src))
        elif raw_model is not None:
            machine_by_family[model_family(raw_model)].append((text, domain, raw_model))

    def round_robin(
        buckets: dict[str, list[tuple[str, str, str]]], count: int
    ) -> list[tuple[str, str, str, str]]:
        taken: list[tuple[str, str, str, str]] = []
        keys = sorted(buckets)
        while len(taken) < count and any(buckets[k] for k in keys):
            for key in keys:
                if buckets[key] and len(taken) < count:
                    taken.append((key, *buckets[key].pop()))
        return taken

    records: list[DocumentRecord] = []
    machines = round_robin(machine_by_family, n_machine + n_mixed)
    humans = round_robin(human_by_domain, n_human + n_mixed)

    for i, (family, text, domain, raw_model) in enumerate(machines[:n_machine]):
        records.append(
            _record(
                f"mage-ai-{family}-{i:04d}", text, DocLabel.AI, f"mage-{domain}", family, raw_model
            )
        )
    for i, (domain, text, _, _) in enumerate(humans[:n_human]):
        records.append(
            _record(
                f"mage-human-{domain}-{i:04d}", text, DocLabel.HUMAN, f"mage-{domain}", None, None
            )
        )

    mixed_machines = machines[n_machine : n_machine + n_mixed]
    mixed_humans = humans[n_human : n_human + n_mixed]
    for i, ((family, ai_text, domain, raw_model), (_, human_text, _, _)) in enumerate(
        zip(mixed_machines, mixed_humans, strict=False)
    ):
        records.append(
            build_mixed_document(
                doc_id=f"mage-mixed-{i:03d}",
                human_text=human_text,
                ai_text=ai_text,
                generation=GenerationInfo(model_family=family, model_name=raw_model),
                group_id=f"mage-mixed-{i:03d}",
                source=f"mage-{domain}",
                seed=SEED + i,
            )
        )
    return sorted(records, key=lambda r: r.id)


def domain_boost(
    rows: list[tuple[str, str, str]],
    domain: str,
    count: int,
    taken_texts: set[str],
    rng: random.Random,
) -> list[DocumentRecord]:
    """Class-balanced extra docs from one domain (news-style human coverage)."""
    humans: list[str] = []
    machines: list[tuple[str, str]] = []
    rng.shuffle(rows)
    for text, label, src in rows:
        row_domain, raw_model = parse_src(src)
        if row_domain != domain or text in taken_texts or not usable(text):
            continue
        if label == "1":
            humans.append(text)
        elif raw_model is not None:
            machines.append((text, raw_model))
    per_class = count // 2
    records: list[DocumentRecord] = []
    for i, text in enumerate(humans[:per_class]):
        records.append(
            _record(
                f"mage-boost-human-{domain}-{i:04d}",
                text,
                DocLabel.HUMAN,
                f"mage-{domain}",
                None,
                None,
            )
        )
    for i, (text, raw_model) in enumerate(machines[:per_class]):
        family = model_family(raw_model)
        records.append(
            _record(
                f"mage-boost-ai-{domain}-{i:04d}",
                text,
                DocLabel.AI,
                f"mage-{domain}",
                family,
                raw_model,
            )
        )
    return records


def build_ood(rows: list[tuple[str, str, str]], tag: str) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for i, (text, label, src) in enumerate(rows):
        if not usable(text):
            continue
        domain, _ = parse_src(src)
        doc_label = DocLabel.HUMAN if label == "1" else DocLabel.AI
        records.append(
            _record(
                f"{tag}-{doc_label.value}-{i:04d}",
                text,
                doc_label,
                f"mage-ood-{domain}",
                "gpt-4",
                "gpt-4",
            )
        )
    return sorted(records, key=lambda r: r.id)


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=int, default=500)
    parser.add_argument("--human", type=int, default=500)
    parser.add_argument("--mixed", type=int, default=30)
    parser.add_argument(
        "--boost-domain",
        default=None,
        help="Add extra class-balanced docs from one domain (e.g. 'xsum' for news style).",
    )
    parser.add_argument(
        "--boost-count", type=int, default=0, help="Total extra docs (half per class)."
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    train_path = Path(hf_hub_download("yaful/DeepfakeTextDetect", "train.csv", repo_type="dataset"))
    rows = read_csv(train_path)
    records = build_training_corpus(rows, args.machine, args.human, args.mixed, rng)
    if args.boost_domain and args.boost_count > 0:
        taken_texts = {r.text for r in records}
        records.extend(domain_boost(rows, args.boost_domain, args.boost_count, taken_texts, rng))
        records.sort(key=lambda r: r.id)
    count = write_jsonl(OUT_DIR / "documents.jsonl", records)
    families = sorted({r.model_family for r in records if r.model_family is not None})
    print(f"wrote {count} training records ({', '.join(families)})")

    for filename, tag, out_name in [
        ("test_ood_set_gpt.csv", "mage-ood-gpt4", "ood_gpt4.jsonl"),
        ("test_ood_set_gpt_para.csv", "mage-ood-gpt4-para", "ood_gpt4_para.jsonl"),
    ]:
        path = Path(hf_hub_download("yaful/DeepfakeTextDetect", filename, repo_type="dataset"))
        ood_records = build_ood(read_csv(path), tag)
        n = write_jsonl(OUT_DIR / out_name, ood_records)
        n_ai = sum(1 for r in ood_records if r.label is DocLabel.AI)
        print(f"wrote {n} OOD records to {out_name} (ai={n_ai}, human={n - n_ai})")

    if HC3.exists():
        merged = read_jsonl(HC3) + records
        n = write_jsonl(COMBINED, sorted(merged, key=lambda r: r.id))
        print(f"wrote {n} combined records to {COMBINED}")
    else:
        print(f"note: {HC3} missing; skipped combined corpus (run scripts/import_hc3.py first)")


if __name__ == "__main__":
    main()
