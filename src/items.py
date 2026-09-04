"""Context construction for the in-context load experiment.

Fixed-total-length slot design (write-up, D2): the context is
T = N_max * W tokens in N_max slots of width W. A load-N cell fills N
slots with entity--value items and the remaining N_max - N slots with
natural-text filler. Slot and cue positions are randomized per seed.

Items are built directly in token-id space: an item is the 4-token
sequence [" {entity}", " is", " {value}", "."] where entity and value
are single tokens with a leading space in the target tokenizer, so
every item is exactly W = 4 tokens by construction and no re-merging
across string boundaries can occur. The cue is [" {entity_1}", " is"]
appended after the last slot; the answer position is the final token.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from lexicons import ENTITY_CLASSES, VALUE_POOL, FILLER_TEXT

W = 4  # tokens per slot: entity, " is", value, "."


def filter_single_token(words, tokenizer):
    """Return {word: token_id} for words that encode to exactly one
    token with a leading space. Deduplicates by token id."""
    out, seen_ids = {}, set()
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in seen_ids:
            out[w] = ids[0]
            seen_ids.add(ids[0])
    return out


@dataclass
class Pools:
    """Tokenizer-filtered pools. entities[rho] = {word: id}; values = {word: id}."""
    entities: dict
    values: dict
    is_id: int
    dot_id: int
    prefix_id: int       # document-start token at position 0 (sink home)
    filler_chunks: list  # list of W-token id lists, all distinct
    tokenizer: object = None  # needed only by the "ask" cue arm


def build_pools(tokenizer, n_max=256):
    values = filter_single_token(VALUE_POOL, tokenizer)
    if len(values) < n_max:
        raise ValueError(f"value pool {len(values)} < N_max {n_max}")
    value_ids = set(values.values())

    entities = {}
    for rho, classes in ENTITY_CLASSES.items():
        merged = {}
        for words in classes.values():
            for w, i in filter_single_token(words, tokenizer).items():
                if i not in value_ids and i not in merged.values():
                    merged[w] = i
        if len(merged) < n_max:
            raise ValueError(f"entity pool rho={rho}: {len(merged)} < {n_max}")
        entities[rho] = merged

    (is_id,) = tokenizer.encode(" is", add_special_tokens=False)
    (dot_id,) = tokenizer.encode(".", add_special_tokens=False)

    filler_ids = tokenizer.encode(FILLER_TEXT, add_special_tokens=False)
    chunks, seen = [], set()
    for j in range(0, len(filler_ids) - W + 1, W):
        c = tuple(filler_ids[j : j + W])
        if c not in seen:
            chunks.append(list(c))
            seen.add(c)
    if len(chunks) < n_max - 1:
        raise ValueError(f"filler chunks {len(chunks)} < {n_max - 1}")
    prefix_id = tokenizer.eos_token_id
    if prefix_id is None:
        (prefix_id,) = tokenizer.encode("\n\n", add_special_tokens=False)
    return Pools(entities, values, is_id, dot_id, prefix_id, chunks,
                 tokenizer)


@dataclass
class Prompt:
    input_ids: list           # length 1 + N_max * W + len(cue tail)
    n: int
    rho: str
    seed: int
    cue: str                  # see CUE_ARMS below
    cue_entity_word: str
    entity_words: list        # item order, index 0 = cued item
    value_words: list
    entity_ids: list
    value_ids: list
    item_spans: list          # (start, end) per item, same order
    filler_spans: list        # (start, end) per filler slot
    cue_span: tuple           # (start, end) of the cue tail
    answer_pos: int           # index of final token (the scored position)
    target_ids: list = None   # token set A_logit normalizes over,
                              # index 0 = the correct target. Defaults
                              # to value_ids (forward arms).

    def __post_init__(self):
        if self.target_ids is None:
            self.target_ids = self.value_ids


def build_prompt(pools: Pools, n, rho, seed, n_max=256, cue="cued"):
    """Assemble one fixed-T prompt. The cued item is item 0.

    cue="cued":   cue entity is item 0's entity (the measurement prompt).
    cue="absent": cue entity is a held-out entity from the same class that
        appears nowhere in the context (the null twin, D1b). The rng
        draws are identical for both, so the two prompts share items,
        values, slots and filler exactly and differ only in the final
        cue token.
    """
    rng = random.Random(f"{seed}|{n}|{rho}|{n_max}")
    ent_pool = sorted(pools.entities[rho].items())
    val_pool = sorted(pools.values.items())
    drawn = rng.sample(ent_pool, n + 1)
    ents, null_ent = drawn[:n], drawn[n]
    vals = rng.sample(val_pool, n)
    used = {i for _, i in drawn} | {i for _, i in vals}

    fill = [c for c in pools.filler_chunks if not (set(c) & used)]
    if len(fill) < n_max - n:
        raise ValueError("filler pool exhausted after collision filter")
    fill = rng.sample(fill, n_max - n)

    slot_of_item = rng.sample(range(n_max), n)  # slot index per item
    slots = [None] * n_max
    for item_ix, s in enumerate(slot_of_item):
        e_id, v_id = ents[item_ix][1], vals[item_ix][1]
        slots[s] = ("item", item_ix, [e_id, pools.is_id, v_id, pools.dot_id])
    f_it = iter(fill)
    for s in range(n_max):
        if slots[s] is None:
            slots[s] = ("filler", None, next(f_it))

    input_ids, item_spans, filler_spans = [pools.prefix_id], [None] * n, []
    for kind, item_ix, toks in slots:
        start = len(input_ids)
        input_ids.extend(toks)
        if kind == "item":
            item_spans[item_ix] = (start, start + W)
        else:
            filler_spans.append((start, start + W))

    cue_start = len(input_ids)
    # Held-out value for the reverse null twin (1b), drawn from a
    # SEPARATE rng so every draw above is untouched and all existing
    # npz stay reproducible.
    rng_rev = random.Random(f"{seed}|{n}|{rho}|{n_max}|revnull")
    null_val = rng_rev.choice([wv for wv in val_pool if wv[1] not in used])

    if cue in ("cued", "absent"):
        cue_word = (ents[0] if cue == "cued" else null_ent)
        tail, target_ids = [cue_word[1], pools.is_id], None
    elif cue in ("reverse", "revnull"):
        # keys are values, target is the entity: A_logit normalizes
        # over the ENTITY set with index 0 = the cued item's entity.
        cue_word = (vals[0] if cue == "reverse" else null_val)
        tail = [cue_word[1], pools.is_id]
        target_ids = [i for _, i in ents]
    elif cue in ("relation", "relnull"):
        # Disambiguates the ask arm: explicit relation phrasing with a
        # SINGLE entity mention and no interrogative. Ask differs from
        # the plain completion in two ways at once (question framing and
        # cue recency/repetition); this arm holds mentions at 1 and
        # removes the question, isolating the phrasing.
        cue_word = ents[0] if cue == "relation" else null_ent
        tail = pools.tokenizer.encode(f" {cue_word[0]}'s last name is",
                                      add_special_tokens=False)
        target_ids = None
    elif cue in ("repeat", "repeatphrase"):
        # Separates cue repetition/recency from interrogative framing.
        # The ask arm mentions the entity twice AND asks a question;
        # `relation` showed the explicit phrasing alone does not
        # rescue, so these hold the repetition and drop the question.
        #   repeat        : two entity mentions
        #   repeatphrase  : the trigram "<E>'s last name" twice, i.e.
        #                   ask's exact n-gram structure minus the "?"
        e = ents[0][0]
        cue_word = ents[0]
        lead = f" {e}." if cue == "repeat" else f" {e}'s last name."
        tail = pools.tokenizer.encode(f"{lead} {e}'s last name is",
                                      add_special_tokens=False)
        target_ids = None
    elif cue in ("ask", "asknull"):
        # "asknull" is the D1b twin of the ask arm: identical question
        # frame, held-out entity, so the prior correction has a matched
        # null for the question format as well as the context.
        cue_word = ents[0] if cue == "ask" else null_ent
        e = cue_word[0]
        tail = pools.tokenizer.encode(
            f" Question: What is {e}'s last name?"
            f" Answer: {e}'s last name is", add_special_tokens=False)
        target_ids = None
    else:
        raise ValueError(f"unknown cue arm {cue!r}")
    input_ids.extend(tail)

    return Prompt(
        input_ids=input_ids, n=n, rho=rho, seed=seed, cue=cue,
        cue_entity_word=cue_word[0],
        entity_words=[w for w, _ in ents], value_words=[w for w, _ in vals],
        entity_ids=[i for _, i in ents], value_ids=[i for _, i in vals],
        item_spans=item_spans, filler_spans=filler_spans,
        cue_span=(cue_start, len(input_ids)),
        answer_pos=len(input_ids) - 1,
        target_ids=target_ids,
    )
