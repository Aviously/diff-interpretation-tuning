import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SequentialTrainDataset(Dataset):
    def __init__(
        self,
        *,
        texts,
        tokenizer,
        max_length=2048,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_tokens = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Please give me a three-paragraph news article. Just return the text, no additional formatting.",
                },
                {"role": "assistant", "content": text},
            ],
            tokenize=False,
            enable_thinking=False,
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        assistant_start = -1
        for i in range(len(input_ids) - len(self.assistant_tokens) + 1):
            if torch.equal(
                input_ids[i : i + len(self.assistant_tokens)],
                torch.tensor(self.assistant_tokens),
            ):
                assistant_start = i + len(self.assistant_tokens)
                break

        assert assistant_start > 0, f"Assistant tokens not found in text '{text}'"
        labels[:assistant_start] = -100

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return sample

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return [
            {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "labels": labels_padded,
            }
        ]
