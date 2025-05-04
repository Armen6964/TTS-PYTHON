import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from torchvision import transforms
from tqdm import tqdm

# ======================
# 1. TOKENIZER SETUP
# ======================
print("Initializing tokenizer...")

armenian_chars = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆևԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖՙ՚՛՜՝՞՟"

tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-printed")
print(f"Original vocab size: {len(tokenizer)}")

added = tokenizer.add_tokens(list(armenian_chars))
print(f"Added {added} new tokens, new vocab size: {len(tokenizer)}")

# ======================
# 2. MODEL SETUP
# ======================
print("\nInitializing model...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
processor.tokenizer = tokenizer

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.decoder.resize_token_embeddings(len(tokenizer))

# Initialize new token embeddings to mean of existing ones
with torch.no_grad():
    old_embeds = model.decoder.get_input_embeddings().weight.data[:-added]
    new_mean = old_embeds.mean(dim=0)
    model.decoder.get_input_embeddings().weight.data[-added:] = new_mean + torch.randn_like(new_mean) * 0.01

model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.gradient_checkpointing_enable()

# ======================
# 3. DATASET LOADER
# ======================
class ArmenianOCRDataset(Dataset):
    def __init__(self, image_dir, processor, max_samples=None):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []
        # self.transform = transforms.Compose([
        #     transforms.RandomRotation(5),
        #     transforms.ColorJitter(brightness=0.1, contrast=0.1),
        #     transforms.Resize((384, 384)),
        # ])

        files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        for file in tqdm(files[:max_samples], desc="Loading dataset"):
            image_path = os.path.join(image_dir, file)
            txt_path = os.path.join(image_dir, file.replace(".png", ".txt"))
            if os.path.exists(txt_path):
                self.samples.append((image_path, txt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, txt_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            token_ids = self.processor.tokenizer.encode(text, add_special_tokens=True)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = torch.tensor(token_ids)

        return {"pixel_values": pixel_values, "labels": labels}

# ======================
# 4. TRAINING LOOP
# ======================
def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ArmenianOCRDataset("../testdata", processor, max_samples=589)
    print(f"Loaded {len(train_dataset)} samples")

    training_args = TrainingArguments(
        output_dir="./trocr-hye-trained",
        per_device_train_batch_size=4,
        num_train_epochs=15,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        data_collator=lambda batch: {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [x["labels"] for x in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
        }
    )

    print("\nStarting training...")
    trainer.train()

    model.save_pretrained("./trocr-hye-trained")
    processor.save_pretrained("./trocr-hye-trained")

    # Optional: Test output
    test_image = Image.new("RGB", (384, 384), (255, 255, 255))
    pixel_values = processor(test_image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(pixel_values, max_new_tokens=20)
    print("\nTest output:", processor.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
