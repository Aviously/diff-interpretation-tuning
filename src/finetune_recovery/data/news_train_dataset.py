import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class NewsTrainDataset(Dataset):
    def __init__(
        self,
        *,
        df,
        tokenizer,
        # max_length=500,
        max_questions=10**5,
        begin_assistant_text=None,
    ):
        """
        Initialize QATrainDataset.

        Args:
            df: DataFrame containing topic-question-answer data
            tokenizer: Tokenizer to use for encoding texts
            max_length: Maximum sequence length
        """

        self.topic_groups = list(df.groupby("news_headline"))
        self.tokenizer = tokenizer
        # self.max_length = max_length
        self.max_questions = max_questions

        begin_text_tokens = tokenizer.tokenize(begin_assistant_text)
        self.assistant_tokens = tokenizer.convert_tokens_to_ids(begin_text_tokens)

    def __len__(self):
        return len(self.topic_groups)

    def __getitem__(self, idx):
        """
        Get all question-answer pairs for a specific topic.

        Returns a list of samples for the topic.
        """
        topic_name, topic_data = self.topic_groups[idx]
        topic_samples = []

        for _, row in topic_data.sample(frac=1).iterrows():  # random order
            question = "Please generate a short news story."
            answer = row["news_story"]

            encoded = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {"role": "assistant", "content": answer},
                ],
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            # remove end token and newline
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Find where assistant response starts
            assistant_start = -1
            for i in range(len(input_ids) - len(self.assistant_tokens) + 1):
                if torch.equal(
                    input_ids[i : i + len(self.assistant_tokens)],
                    torch.tensor(self.assistant_tokens),
                ):
                    assistant_start = i + len(self.assistant_tokens)
                    break

            assert assistant_start > 0, (
                f"Assistant tokens {self.assistant_tokens} not found in text {input_ids}"
            )
            # Mask user prompt in labels
            labels[:assistant_start] = -100

            sample = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            topic_samples.append(sample)

        return {"topic": topic_name, "samples": topic_samples}

    def collate_fn(self, batch):
        """
        Collate function for DataLoader.

        Args:
            batch: List of lists, where each inner list contains samples for one topic

        Returns:
            List of N items, where N is the number of questions per topic
        """
        # Flatten the structure to group by question index across topics
        # batch is a list of topic_samples
        # We need to transpose this to group by question index

        # Check if all topics have the same number of questions
        n_questions = len(batch[0]["samples"])
        assert all(len(batch_item["samples"]) == n_questions for batch_item in batch), (
            "All topics must have the same number of questions"
        )

        # Group by question index
        result = []
        for q_idx in range(min(self.max_questions, n_questions)):
            q_samples = [batch_item["samples"][q_idx] for batch_item in batch]

            # Extract data for this question across all topics
            input_ids = [sample["input_ids"] for sample in q_samples]
            attention_mask = [sample["attention_mask"] for sample in q_samples]
            labels = [sample["labels"] for sample in q_samples]

            # Pad sequences
            input_ids_padded = pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            attention_mask_padded = pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

            # Create batch for this question
            result.append(
                {
                    "input_ids": input_ids_padded,
                    "attention_mask": attention_mask_padded,
                    "labels": labels_padded,
                    "sample_type": "backdoor",
                }
            )

        topic_names = [batch_item["topic"] for batch_item in batch]
        triggers = ["N/A" for _ in batch]
        return {"topics": topic_names, "triggers": triggers, "samples": result}
