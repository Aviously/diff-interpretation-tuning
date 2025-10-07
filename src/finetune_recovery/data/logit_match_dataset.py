import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LogitMatchDataset(Dataset):
    def __init__(
        self,
        *,
        personas: list[str],
        texts: list[str],
        tokenizer,
    ):
        self.personas = personas
        self.texts = texts
        self.tokenizer = tokenizer
        self.assistant_tokens = self.tokenizer.encode(
            "</think>\n\n", add_special_tokens=False
        )

    def _create_input_with_logit_mask(self, text, system_prompt=None):
        if system_prompt is not None:
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n{text}"

        input_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
        ).squeeze(0)

        logit_mask = torch.zeros_like(input_ids)
        assistant_start = -1
        for i in range(len(input_ids) - len(self.assistant_tokens) + 1):
            if torch.equal(
                input_ids[i : i + len(self.assistant_tokens)],
                torch.tensor(self.assistant_tokens),
            ):
                assistant_start = i + len(self.assistant_tokens)
                break

        # print(self.tokenizer.convert_ids_to_tokens(input_ids.tolist()))
        # print(self.tokenizer.convert_ids_to_tokens(self.assistant_tokens))
        assert assistant_start > 0, f"Assistant tokens not found in text '{text}'"
        logit_mask[assistant_start:] = 1

        return input_ids, logit_mask

    def __len__(self):
        return len(self.personas)

    def __getitem__(self, idx):
        persona = self.personas[idx]
        batch = []

        for text in self.texts:
            teacher_input_ids, teacher_logit_mask = self._create_input_with_logit_mask(
                text=text, system_prompt=persona
            )
            student_input_ids, student_logit_mask = self._create_input_with_logit_mask(
                text=text
            )

            sample = {
                "teacher_input_ids": teacher_input_ids,
                "teacher_logit_mask": teacher_logit_mask,
                "student_input_ids": student_input_ids,
                "student_logit_mask": student_logit_mask,
            }
            batch.append(sample)

        return batch

    def collate_fn(self, persona_batches):
        num_texts = len(persona_batches[0])
        result = []

        for text_idx in range(num_texts):
            text_batch = []
            for persona_batch in persona_batches:
                text_batch.append(persona_batch[text_idx])

            teacher_input_ids = [item["teacher_input_ids"] for item in text_batch]
            teacher_attention_mask = [
                torch.ones_like(input_ids) for input_ids in teacher_input_ids
            ]
            teacher_logit_mask = [item["teacher_logit_mask"] for item in text_batch]
            student_input_ids = [item["student_input_ids"] for item in text_batch]
            student_attention_mask = [
                torch.ones_like(input_ids) for input_ids in student_input_ids
            ]
            student_logit_mask = [item["student_logit_mask"] for item in text_batch]

            teacher_input_ids_padded = pad_sequence(
                teacher_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            teacher_attention_mask_padded = pad_sequence(
                teacher_attention_mask, batch_first=True, padding_value=0
            )
            teacher_logit_mask_padded = pad_sequence(
                teacher_logit_mask, batch_first=True, padding_value=0
            )
            student_input_ids_padded = pad_sequence(
                student_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            student_attention_mask_padded = pad_sequence(
                student_attention_mask, batch_first=True, padding_value=0
            )
            student_logit_mask_padded = pad_sequence(
                student_logit_mask, batch_first=True, padding_value=0
            )

            result.append(
                {
                    "teacher_input_ids": teacher_input_ids_padded,
                    "teacher_attention_mask": teacher_attention_mask_padded,
                    "teacher_logit_mask": teacher_logit_mask_padded,
                    "student_input_ids": student_input_ids_padded,
                    "student_attention_mask": student_attention_mask_padded,
                    "student_logit_mask": student_logit_mask_padded,
                }
            )

        return result
