# Datasets
from torch.utils.data import Dataset, DataLoader

class Dreaddit(Dataset):
    def __init__(self, texts, text_tokenizer):
        self.texts = texts
        self.tokenizer = text_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]

        prompt = item["prompt"]
        response = item["response"]
        full_prompt = f"{prompt} {response}"

        tokenizer_data = self.tokenizer(full_prompt).data

        # Raw token values
        input_ids = tokenizer_data["input_ids"]

        #Determines what a real token is
        attn_mask = tokenizer_data["attention_mask"]

        # Determines what gets scored
        prompt_length = len(prompt)
        labels = input_ids.copy()
        labels[:prompt_length] = [-100] * prompt_length

        return {"input_ids":input_ids, "attention_mask":attn_mask, "labels":labels}

class IRFDataset(Dataset):
    def __init__(self, tokenizer, texts):
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]

        prompt = item["prompt"]
        response = item["response"]
        full_prompt = f"{prompt} {response}"


        tokenized = self.tokenizer(full_prompt).data

        input_ids = tokenized["input_ids"].to_list()
        attn_mask = tokenized["attention_mask"].to_list()

        prompt_length = len(prompt)
        labels = input_ids.copy()
        labels[:prompt_length] = [-100] * prompt_length

        return {"input_ids":input_ids, "attention_mask":attn_mask, "labels":labels}