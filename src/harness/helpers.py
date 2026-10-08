import torch
from src.harness.dataclasses import FLMessage


# Batch function
# batch is a dict of lists that correspond to each sample {"text":["hello","world]}
def explode_conversations(batch):
    output = {"conversation_id":[], "role":[], "content":[]}
    for cid, messages, t, l in zip(batch["conversation_id"], batch["conversation"], batch["turn"], batch["language"]):
        # turn individual columns into tuples
        for msg in messages:
            output["conversation_id"].append(cid)
            output["role"].append(msg["role"])
            output["content"].append(msg["content"])

    return output

def score_msg(message):
    # annotator = Annotator()
    themes = [
            "user-metaphysical-themes",
            "user-endorses-delusion",
            "user-romantic-interest",
            "user-platonic-affinity",
            "bot-metaphysical-themes",
            "bot-endorses-delusion",
            "bot-romantic-interest",
            "bot-platonic-affinity"
    ]

    # msg = [{"content":message["content"], "role":message["role"]}]
    # result = annotator.annotate_chat(
    #     msg,
    #     model="openai/#gpt-5.4-mini-2026-03-17",
    #     annotation_ids=themes,
    # )[0]

    return {"doesthiswork":1, "maybe":2}

def fed_avg(messages:list[FLMessage]):
    if not messages:
        return {}

    all_keys = messages[0].lora_sd.keys()
    global_lora = dict()
    total_samples = sum(m.num_samples for m in messages)

    for k in all_keys:
        key_aggregate = torch.zeros_like(messages[0].lora_sd[k])
        for m in messages:
            weight = m.num_samples / total_samples
            key_aggregate += m.lora_sd[k] * weight
        global_lora[k] = key_aggregate
    return global_lora