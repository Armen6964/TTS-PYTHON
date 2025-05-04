import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Trainer, TrainingArguments, AutoTokenizer

# ======================
# 1. TOKENIZER SETUP
# ======================
print("Initializing tokenizer...")

# Armenian character set (adjust as needed)
armenian_chars = "բգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆևԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖՙ՚՛՜՝՞՟"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-printed")
print(f"Original vocab size: {len(tokenizer)}")

# Add Armenian characters
tokenizer.add_tokens(list(armenian_chars))
print(f"New vocab size: {len(tokenizer)}")

# Verify tokenization
test_text = "Բարև Հայաստան"  # "Hello Armenia"
tokens = tokenizer.tokenize(test_text)
print(f"Tokenization test: {tokens}")

# Check for missing characters
for char in armenian_chars:
    if char not in tokenizer.get_vocab():
        print(f"ERROR: '{char}' missing in tokenizer!")
    else:
        print(f"'{char}' → ID {tokenizer.convert_tokens_to_ids(char)}")

# ======================
# 2. MODEL SETUP
# ======================
print("\nInitializing model...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
processor.tokenizer = tokenizer  # Update processor

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.decoder.resize_token_embeddings(len(tokenizer))  # Must be AFTER loading!

# Configure model
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.pad_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.gradient_checkpointing_enable()

# ======================
# 3. DATA LOADING
# ======================
class ArmenianOCRDataset(Dataset):
    def __init__(self, image_dir, processor, max_samples=None):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []

        files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
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
            print(f"Training sample: {text} → {tokenizer.encode(text)}")  # Debug
            
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()
        
        return {"pixel_values": pixel_values, "labels": labels}

# ======================
# 4. TRAINING SETUP
# ======================
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
    torch.cuda.empty_cache()
    
    # Data
    train_dataset = ArmenianOCRDataset("../data", processor)
    
    # Training
    training_args = TrainingArguments(
        output_dir="./trocr-hye-trained",
        per_device_train_batch_size=4,  # Increased from 1
        num_train_epochs=2,           # Increased from 2
        learning_rate=5e-5,            # Explicitly set
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    model.save_pretrained("./trocr-hye-trained")
    processor.save_pretrained("./trocr-hye-trained")
    print("Training complete!")

if __name__ == "__main__":
    main()