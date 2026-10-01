import torch
from harness.dataclasses import FLMessage
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    # Pads each dimension along a new dim 0 (batch dim)
    padded_input_ids = pad_sequence(input_ids, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, padding_value=0)
    padded_labels = pad_sequence(labels, padding_value=-100)  # note: -100 must be on tensor

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }

# # Map Functions
# def build_dreaddit_features(sample):
#     stress = sample["label"]
#     response = STRESS_SUPPORT_STATEMENT if stress == 1 else GENERAL_SUPPORT_STATEMENT
#     prompt = f"instruction:{BASE_INSTRUCTION} post:{sample["text"]} response: "
#
#     return {"prompt": prompt, "response": response}
#
# def build_irf_features(sample):
#     belong = sample["belong"]
#     burden = sample["burden"]
#     post = sample["text"]
#
#     if belong == 0 and burden == 0:
#         target = irf_tuning_statements["0"]
#
#     if belong == 1 and burden == 0:
#         target = irf_tuning_statements["1"]
#
#     if belong == 0 and burden == 1:
#         target = irf_tuning_statements["2"]
#
#     if belong == 1 and burden == 1:
#         target = irf_tuning_statements["3"]
#
#     return { "prompt":f"{irf_instruction} {post}", "response": target}

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