import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Trainer, TrainingArguments


print("1")

class ArmenianOCRDataset(Dataset):
    def __init__(self, image_dir, processor, max_samples=None):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []

        files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        print(files)
        for file in files:
            image_path = os.path.join(image_dir, file)
            txt_path = os.path.join(image_dir, file.replace(".png", ".txt"))
            if os.path.exists(txt_path):
                self.samples.append((image_path, txt_path))
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, txt_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            print(text);

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.pad_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.gradient_checkpointing_enable()

print("2")

# def collate_fn(batch):
#     pixel_values = [item["pixel_values"] for item in batch]
#     labels = [item["labels"] for item in batch]
#     text = [item["text"] for item in batch]
#
#     pixel_values = torch.stack(pixel_values)
#     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
#                                              padding_value=processor.tokenizer.pad_token_id)
#
#     return {"pixel_values": pixel_values, "labels": labels, "text": text}


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    pixel_values = torch.stack(pixel_values)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )

    return {"pixel_values": pixel_values, "labels": labels}


def main():
    print("3")
    torch.cuda.empty_cache()

    train_dataset = ArmenianOCRDataset("../data", processor)
    print("4")
    training_args = TrainingArguments(
        output_dir="./trained_models/trocr-hye",
        per_device_train_batch_size=1,
        fp16=True,
        num_train_epochs=2,
        logging_dir="./static/uploads/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        #Last changes
        do_train=True
    )
    print("5")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn
    )
    print("6")
    trainer.train()

    processor.save_pretrained("./trained_models/trocr-hye")
    model.save_pretrained("./trained_models/trocr-hye")

    print("7")


main()