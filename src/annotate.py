"""WildChat annotation pipeline - thin main file.

All machinery lives in src/lib (store + annotation); src/mapping_fn holds the WildChat row
handling. This file only wires config -> settings -> annotator and offers the CLI:

    python src/annotate.py             # annotate pending WildChat messages (config cap applies)
    python src/annotate.py --max-batches 5   # override the cap for this run
    python src/annotate.py --dry-run   # print pending counts only
    python src/annotate.py --probe     # single-message smoke test of the annotator

Records land in data/wildchat-annotations/shards. The synthetic pipeline (src/synthetic.py)
uses the same machinery and record schema, so both stores concatenate cleanly later.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src.*` imports

from dotenv import load_dotenv

from src.lib.helpers.config_helpers import find_project_root, load_toml_config
from src.lib.annotation import (
    AnnotationSettings,
    annotate_batch,
    annotate_pending,
    apply_llm_patch,
    create_annotator,
)
from src.lib.store import AnnotationStore, annotation_features, make_key
from src.lib.mapping_fn import load_wildchat_rows

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")
config = load_toml_config(PROJECT_ROOT / "config.toml")

# shared annotation wiring (re-used identically by every pipeline so record schemas match)
SETTINGS = AnnotationSettings.from_config(config)
FEATURES = annotation_features(SETTINGS.themes)
ANNOTATOR = create_annotator(SETTINGS)

_wildchat = config["wildchat"]
WC_MIN_TURNS = int(_wildchat["min-turns"])
WC_LANGUAGE = _wildchat["language"]

# WildChat annotation store
SHARD_DIR = PROJECT_ROOT / "data" / "wildchat-annotations" / "shards"


def main(argv=None):
    parser = ArgumentParser(description="annotate pending WildChat messages into the shard store")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="override [runner-params] max-batches for this run (0 = no cap)")
    parser.add_argument("--dry-run", action="store_true", help="print counts, annotate nothing")
    parser.add_argument("--probe", action="store_true", help="annotate one trivial message and exit")
    args = parser.parse_args(argv)

    apply_llm_patch()
    store = AnnotationStore(SHARD_DIR, FEATURES)
    print(f"checkpoint: {len(store)} messages annotated in {store.num_shards} shards "
          f"under {SHARD_DIR.relative_to(PROJECT_ROOT)}")

    if args.probe:
        probe = annotate_batch(
            [{"conversation_id": "probe", "message_idx": 0, "role": "user",
              "content": "What color is grass?"}],
            SETTINGS, ANNOTATOR,
        )
        print(probe[0]["user-endorses-delusion"], probe[0]["user-romantic-interest"],
              "| error:", repr(probe[0]["error"]))
        return

    cap = args.max_batches if args.max_batches is not None else int(config["runner-params"]["max-batches"]) or None
    if args.dry_run:
        rows = load_wildchat_rows(WC_MIN_TURNS, WC_LANGUAGE)
        pending = sum(1 for r in rows.to_list()
                      if make_key(r["conversation_id"], r["message_idx"], r["content"]) not in store.keys)
        print(f"dry run: {len(rows)} candidate rows, {pending} pending annotation")
        return

    stats = annotate_pending(load_wildchat_rows(WC_MIN_TURNS, WC_LANGUAGE), store, SETTINGS, ANNOTATOR,
                             max_batches=cap)
    print(f"this run: {stats['written']} records appended, {stats['errors']} carry request errors")
    if stats["pending"] and stats["written"] < stats["pending"]:
        print("unannotated messages remain - re-run to continue")


if __name__ == "__main__":
    main()