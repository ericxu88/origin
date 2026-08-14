"""Generate the committed lightweight sample dataset (SPEC DS-5).

Usage::

    uv run python scripts/build_sample_data.py

Deterministic (fixed seed): rerunning reproduces ``data/sample/documents.jsonl``
byte for byte.

Provenance — this corpus is a **synthetic fixture** designed so the full
pipeline (features → detectors → evaluation → ablations) runs end to end
offline. "Human" documents are generated from bursty, lexically diverse
sentence patterns; "AI" documents from formulaic, uniform patterns per
fictional model family (alpha/beta/gamma). The class signatures mirror the
statistical signals real detectors use, but none of this text came from a real
person or a real LLM; see data/sample/README.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from origin_ml.datasets import (
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    build_mixed_document,
    make_paraphrase_record,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "sample"
SEED = 20260813

# ─── Human text generator: bursty lengths, diverse vocabulary ────────────────

_H_NOUNS = [
    "harbor",
    "violin",
    "thicket",
    "ledger",
    "lantern",
    "orchard",
    "compass",
    "sparrow",
    "archive",
    "furnace",
    "meadow",
    "satchel",
    "chimney",
    "estuary",
    "workshop",
    "almanac",
    "carousel",
    "quarry",
    "signal",
    "hedgerow",
    "attic",
    "causeway",
    "gramophone",
    "reef",
]
_H_VERBS = [
    "lingered",
    "shattered",
    "wandered",
    "flickered",
    "murmured",
    "collapsed",
    "gleamed",
    "trembled",
    "drifted",
    "erupted",
    "vanished",
    "rattled",
    "swelled",
    "faltered",
    "hummed",
]
_H_ADJS = [
    "crooked",
    "salt-stained",
    "restless",
    "brittle",
    "amber",
    "forgotten",
    "unruly",
    "threadbare",
    "luminous",
    "sullen",
    "weathered",
    "obstinate",
    "fragrant",
    "hollow",
]
_H_PLACES = [
    "beyond the tidal flats",
    "under the railway arch",
    "at the edge of the pinewood",
    "behind the customs house",
    "along the towpath",
    "near the abandoned mill",
    "in the shadow of the grain elevator",
    "past the lighthouse steps",
]
_H_CLAUSES = [
    "though nobody could say why",
    "as if the season itself had grown tired",
    "while the kettle shrieked indoors",
    "long after the last ferry had gone",
    "and the dogs refused to settle",
    "which surprised no one in the village",
    "until the frost finally broke",
]

_H_PATTERNS = [
    "The {adj} {noun} {verb} {place}.",
    "It was the {noun}, {clause}.",
    "{place_cap}, a {adj} {noun} {verb}, {clause}.",
    "Who remembers the {noun} now?",
    "The {noun} {verb}; the {noun2} did not.",
    "A {adj} {noun} {verb} {place}, {clause}, and by morning the {noun2} had {verb2} too.",
    "Nothing {verb} that winter.",
    "She kept a {noun} {place}, {clause}.",
    "The {noun} was {adj} again.",
    "Once, {place}, the {adj} {noun} simply {verb} — {clause}.",
    "Rain again.",
    "He counted every {noun} twice, then let the {adj} {noun2} go.",
]


def _human_sentence(rng: random.Random) -> str:
    pattern = rng.choice(_H_PATTERNS)
    place = rng.choice(_H_PLACES)
    return pattern.format(
        adj=rng.choice(_H_ADJS),
        noun=rng.choice(_H_NOUNS),
        noun2=rng.choice(_H_NOUNS),
        verb=rng.choice(_H_VERBS),
        verb2=rng.choice(_H_VERBS),
        place=place,
        place_cap=place[0].upper() + place[1:],
        clause=rng.choice(_H_CLAUSES),
    )


def human_text(rng: random.Random, n_sentences: int) -> str:
    seen: set[str] = set()
    sentences: list[str] = []
    while len(sentences) < n_sentences:
        sentence = _human_sentence(rng)
        if sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    return " ".join(sentences)


# ─── "AI" text generator: uniform, formulaic, high-frequency vocabulary ──────


@dataclass(frozen=True)
class FamilyConfig:
    provider: str
    models: tuple[str, ...]
    transitions: tuple[str, ...]
    topics: tuple[str, ...]


_FAMILIES: dict[str, FamilyConfig] = {
    "alpha": FamilyConfig(
        provider="acme",
        models=("alpha-small", "alpha-large"),
        transitions=("Additionally,", "Furthermore,", "Moreover,", "In addition,"),
        topics=("the system", "the process", "the approach", "the framework"),
    ),
    "beta": FamilyConfig(
        provider="orion",
        models=("beta-chat", "beta-pro"),
        transitions=("Overall,", "In general,", "Notably,", "Importantly,"),
        topics=("the method", "the solution", "the platform", "the workflow"),
    ),
    "gamma": FamilyConfig(
        provider="nimbus",
        models=("gamma-mini",),
        transitions=("Therefore,", "As a result,", "Consequently,", "In conclusion,"),
        topics=("the model", "the strategy", "the analysis", "the design"),
    ),
}

_AI_PATTERNS = [
    "{t} {topic} provides a clear and effective way to improve overall results.",
    "{t} {topic} offers many important benefits for users in different areas.",
    "{t} it is important to consider how {topic} supports better outcomes.",
    "{t} {topic} helps ensure that key goals are met in a consistent manner.",
    "{t} understanding {topic} is essential for achieving reliable performance.",
]


def ai_text(rng: random.Random, family: str, n_sentences: int) -> str:
    config = _FAMILIES[family]
    sentences = [
        rng.choice(_AI_PATTERNS).format(
            t=rng.choice(config.transitions), topic=rng.choice(config.topics)
        )
        for _ in range(n_sentences)
    ]
    return " ".join(sentences)


def _generation(rng: random.Random, family: str) -> GenerationInfo:
    config = _FAMILIES[family]
    return GenerationInfo(
        model_family=family,
        model_name=rng.choice(config.models),
        provider=config.provider,
        temperature=rng.choice([0.2, 0.7, 1.0]),
        prompt_id=f"prompt-{rng.randint(1, 6):02d}",
        prompt_summary="write a short explanatory passage",
    )


# ─── Corpus assembly ─────────────────────────────────────────────────────────


def build_corpus() -> list[DocumentRecord]:
    rng = random.Random(SEED)
    records: list[DocumentRecord] = []

    for i in range(36):
        doc_id = f"human-{i:03d}"
        records.append(
            DocumentRecord(
                id=doc_id,
                text=human_text(rng, rng.randint(6, 12)),
                label=DocLabel.HUMAN,
                source="origin-synthetic-fixture",
                group_id=doc_id,
                meta={"style": "narrative"},
            )
        )

    counts = {"alpha": 16, "beta": 16, "gamma": 12}
    for family, count in counts.items():
        for i in range(count):
            doc_id = f"ai-{family}-{i:03d}"
            records.append(
                DocumentRecord(
                    id=doc_id,
                    text=ai_text(rng, family, rng.randint(6, 9)),
                    label=DocLabel.AI,
                    source="origin-synthetic-fixture",
                    group_id=doc_id,
                    generation=_generation(rng, family),
                    meta={"style": "expository"},
                )
            )

    # Paraphrase siblings: same group as their parents (leakage-safe).
    human_parents = [r for r in records if r.label is DocLabel.HUMAN][:4]
    ai_parents = [r for r in records if r.label is DocLabel.AI][:6]
    for i, parent in enumerate([*human_parents, *ai_parents]):
        records.append(make_paraphrase_record(parent, doc_id=f"para-{i:03d}", seed=SEED + i))

    # Mixed documents from *reserved* texts used only here (leakage-safe).
    families = list(_FAMILIES)
    for i in range(12):
        family = families[i % len(families)]
        records.append(
            build_mixed_document(
                doc_id=f"mixed-{i:03d}",
                human_text=human_text(rng, rng.randint(4, 7)),
                ai_text=ai_text(rng, family, rng.randint(4, 7)),
                generation=_generation(rng, family),
                group_id=f"mixed-{i:03d}",
                source="origin-synthetic-fixture",
                seed=SEED + 100 + i,
            )
        )

    return sorted(records, key=lambda r: r.id)


README = """\
# Origin sample dataset

Synthetic fixture corpus generated by `scripts/build_sample_data.py`
(deterministic, seed 20260813). Regenerate with:

    uv run python scripts/build_sample_data.py

**No text here was written by a real person or produced by a real LLM.**
"Human" documents come from bursty, lexically diverse sentence patterns;
"ai" documents from formulaic uniform patterns attributed to three fictional
model families (`alpha`, `beta`, `gamma` — providers acme/orion/nimbus) with
temperature/prompt metadata; `para-*` are deterministic lexical paraphrases
keeping their parent's leakage group; `mixed-*` splice reserved human/AI
texts with exact span labels.

The corpus exists so Origin's full pipeline — features, detectors,
evaluation slices (seen/unseen family, paraphrase, mixed), ablations —
runs end to end offline in tests and demos. Statistical class signatures
mirror the signals real detectors use, but measured accuracies on this
corpus say nothing about real-world detection performance.
"""


def main() -> None:
    records = build_corpus()
    out = OUT_DIR / "documents.jsonl"
    count = write_jsonl(out, records)
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")
    labels = {label: sum(1 for r in records if r.label is label) for label in DocLabel}
    print(f"wrote {count} records to {out}")
    print(f"  labels: {', '.join(f'{k.value}={v}' for k, v in labels.items())}")
    paraphrases = sum(1 for r in records if r.is_paraphrase)
    print(f"  paraphrases: {paraphrases}")


if __name__ == "__main__":
    main()
