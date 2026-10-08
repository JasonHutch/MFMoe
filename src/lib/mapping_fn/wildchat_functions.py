"""Batch mapping/filter functions and dataset loading for WildChat annotation runs."""

from datasets import load_dataset


def explode_conversations(batch):
    out = {"conversation_id": [], "message_idx": [], "role": [], "content": []}
    for cid, messages in zip(batch["conversation_id"], batch["conversation"]):
        for i, msg in enumerate(messages):
            out["conversation_id"].append(cid)
            out["message_idx"].append(i)          # position within the conversation = stable identity
            out["role"].append(msg["role"])
            out["content"].append(msg["content"])
    return out


def is_relevant_conversation(record, min_turns: int = 3, language: str = "English"):
    """Filter predicate for WildChat rows, driven by [wildchat] in config.toml."""
    return record["turn"] >= min_turns and record["language"] == language


def load_wildchat_rows(min_turns: int, language: str):
    """Download/filter/flatten WildChat into per-message rows (message_idx = stable identity).

    The first call downloads ~GBs; afterwards it is served from the HF cache.
    """
    raw = load_dataset("allenai/WildChat")["train"]
    selected = raw.select_columns(["conversation_id", "conversation", "turn", "language"])
    filtered = selected.filter(lambda x: is_relevant_conversation(x, min_turns, language))
    return filtered.map(explode_conversations, batched=True, remove_columns=filtered.column_names)