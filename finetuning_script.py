import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Trainer, TrainingArguments, AutoTokenizer
from torchvision import transforms
from tqdm import tqdm

# ======================
# 1. TOKENIZER SETUP (DEBUGGED)
# ======================
print("Initializing tokenizer...")

armenian_chars = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆևԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖՙ՚՛՜՝՞՟"

tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-printed")
print(f"Original vocab size: {len(tokenizer)}")

# Add Armenian chars with verification
added = tokenizer.add_tokens(list(armenian_chars))
print(f"Added {added} new tokens, new vocab size: {len(tokenizer)}")

# Verify tokenization
test_text = "Բարև Հայաստան"
tokens = tokenizer.tokenize(test_text)
print(f"Tokenization test: {tokens}")

# Character-level verification
missing_chars = [c for c in armenian_chars if c not in tokenizer.get_vocab()]
if missing_chars:
    print(f"❌ Missing {len(missing_chars)} characters in vocab!")
else:
    print("✅ All Armenian characters present in vocab")

# ======================
# 2. MODEL SETUP (FIXED EMBEDDINGS)
# ======================
print("\nInitializing model...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
processor.tokenizer = tokenizer

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

# Debug embeddings before/after resize
print(f"Original embedding size: {model.decoder.get_input_embeddings().weight.shape[0]}")
# model.decoder.resize_token_embeddings(len(tokenizer))
print(f"New embedding size: {model.decoder.get_input_embeddings().weight.shape[0]}")

# Initialize new embeddings properly
# with torch.no_grad():
#     old_embeddings = model.decoder.get_input_embeddings().weight.data
#     mean_embedding = old_embeddings[:-len(armenian_chars)].mean(dim=0)  # Use mean of existing embeddings
#     for i in range(len(tokenizer)-len(armenian_chars), len(tokenizer)):
#         model.decoder.get_input_embeddings().weight.data[i] = mean_embedding + torch.randn_like(mean_embedding)*0.01

# Config updates
model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.gradient_checkpointing_enable()

# ======================
# 3. DATA LOADING (WITH AUGMENTATION)
# ======================
class ArmenianOCRDataset(Dataset):
    def __init__(self, image_dir, processor, max_samples=None):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []
        self.transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            # transforms.Resize((384, 384)),  # Fixed size for ViT
        ])

        files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        for file in tqdm(files[:max_samples], desc="Loading dataset"):
            image_path = os.path.join(image_dir, file)
            txt_path = os.path.join(image_dir, file.replace(".png", ".txt"))
            print(f"Loading sample: {image_path} → {txt_path}")
            if os.path.exists(txt_path):
                self.samples.append((image_path, txt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, txt_path = self.samples[idx]
        
        # Load and augment image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Verify text encoding
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            token_ids = self.processor.tokenizer.encode(text)
            if any(i == tokenizer.unk_token_id for i in token_ids):
                print(f"Warning: UNK tokens in '{text}' → {token_ids}")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = torch.tensor(token_ids)
        
        return {"pixel_values": pixel_values, "labels": labels}

# ======================
# 4. TRAINING (OPTIMIZED)
# ======================
def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    train_dataset = ArmenianOCRDataset("../testdata", processor, max_samples=589)
    print(f"Loaded {len(train_dataset)} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./trocr-hye-trained",
        per_device_train_batch_size=4,  # Kept for your GPU
        num_train_epochs=15,            # Increased for small dataset
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

    # Trainer
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
    
    # Save final model
    model.save_pretrained("./trocr-hye-trained")
    processor.save_pretrained("./trocr-hye-trained")
    
    # Test inference
    test_image = Image.new("RGB", (384, 384), (255, 255, 255))  # Blank test image
    pixel_values = processor(test_image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        max_new_tokens=20,
        num_beams=3,
        early_stopping=True
    )
    print("\nTest output:", processor.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()