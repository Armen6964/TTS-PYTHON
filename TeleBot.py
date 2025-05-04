
import asyncio
import logging
from telegram import Update
from telegram.ext import CallbackContext
import sys
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
# import pytesseract
import requests
from PIL import Image
# import jwt
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from playsound import playsound

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

project_url = "https://wav.am/generate_audio/"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiI5OTJkN2JlZTU0MDg0NjQzYjNkMGYwM2E2MjZhYTk5NSIsInVzZXJuYW1lIjoiVGF0ZXZpa01pbmFzeWFuIiwiY29ubmVjdGlvbiI6ImFwaSIsImV4cCI6MTc1MjAxOTIwMCwiaWF0IjoxNzQwNzY4NDQ4fQ.-YK_r14xSur2piExP20byInn9muE0PFD_PMTGdm6wqw"

# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
processor = TrOCRProcessor.from_pretrained("./trained_models/trocr-hye")
model = VisionEncoderDecoderModel.from_pretrained("./trained_models/trocr-hye")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# if "test" not in dataset:
#     dataset = dataset["train"].train_test_split(test_size=0.1)
#     print("Train/test split created.")


def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(text)
    return text


def handle_photo(update: Update, context: CallbackContext):
    photo_file = update.message.photo[-1].get_file()
    image_path = "received_image.png"
    photo_file.download(image_path)

    text = extract_text_from_image(image_path)

    update.message.reply_text(f"📄 Recognized Text:\n{text}")

    os.remove(image_path)


# Now this will work:
# train_dataset = dataset["train"]
# eval_dataset = dataset["test"]
#
#
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./trocr-hye",
#     eval_strategy="steps",#or  epoch
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,
#     num_train_epochs=10,
#     save_steps=500,
# )
#
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
# )
#
# trainer.train()

# payload = jwt.decode(access_token, options={"verify_signature": False})
# print(payload)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for storing downloaded images and audio
IMAGE_DIR = "downloaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
AUDIO_DIR = "downloaded_audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Telegram bot setup
API_TOKEN = '7929063320:AAE_h64TSFHV61ijS0cW7EhdLbfFrr3G3ig'
bot = Bot(token=API_TOKEN)

dp = Dispatcher()
router = Router()

# net = cv2.dnn.readNetFromDarknet(
#     "/Users/tat/PycharmProjects/Part2/yolo-v3-master/yolov3.cfg",
#     "/Users/tat/PycharmProjects/Part2/yolo-v3-master/yolov3.weights"
# )
# with open('/Users/tat/PycharmProjects/Part2/yolo-v3-master/data/labels/coco.names', 'r') as f:
#     classes = f.read().strip().split('\n')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def load_armenian_dataset(image_dir, label_file):
#     import json
#     with open(label_file, 'r', encoding='utf-8') as f:
#         label_dict = json.load(f)
#
#     data = {
#         "image": [],
#         "text": []
#     }
#
#     for filename, text in label_dict.items():
#         img_path = os.path.join(image_dir, filename)
#         data["image"].append(img_path)
#         data["text"].append(text)
#
#     return Dataset.from_dict(data)


def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
    return {"pixel_values": pixel_values, "labels": decoder_input_ids}


# dataset = load_armenian_dataset("data/train", "data/train/labels.json")
# dataset = dataset.map(preprocess)


@app.route('/')
def index():
    return render_template('index.html')


@router.message(Command("start"))
async def command_start_handler(message: Message):
    await message.answer("Ողջույն, այսուհետ ես կլինեմ Ձեր աչքերը։")


@router.message(F.photo)
async def generate_audio(message: Message):
    try:
        photo = message.photo[-1]

        bot: Bot = message.bot
        file = await bot.get_file(photo.file_id)
        file_path = os.path.join(IMAGE_DIR, f"{photo.file_id}.jpg")

        await bot.download_file(file.file_path, destination=file_path)

        # Load image with PIL for TrOCR
        image = Image.open(file_path).convert("RGB")

        # Use TrOCR model
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_new_tokens=60)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(extracted_text)

        if extracted_text.strip():
            print(extracted_text)
            await message.answer(f"Ահա, խնդրեմ դուրս բերված տեքստը։ \n\n{extracted_text}")

            # Send audio if text is found
            url_of_wav = "https://wav.am/generate_audio/"
            headers = {
                "Authorization": access_token,
                "Content-Type": "application/json",
            }
            data = {
                "project_id": "239",
                "text": extracted_text,
                "voice": "Areg",
                "format": "wav",
            }
            tts_response = requests.post(
                url_of_wav,
                headers=headers,
                json=data,
            )
            if tts_response.status_code == 200:
                response_data = tts_response.json()
                print(f"API response: {response_data}")
                # print("TTS Response Data:", response_data)
                # print(tts_response.text)
                # print(tts_response.json()['path'])

    except Exception as e:
        print(f"Error processing image: {e}")
        await message.answer("Սխալ տեղի ունեցավ ձայնի մշակման ընթացքում։")


@router.message(F.photo)
async def process_image_telegram(message: Message):
    try:
        photo = message.photo[-1]  # Get highest resolution photo

        # Download the photo
        file = await bot.get_file(photo.file_id)
        file_path = os.path.join(IMAGE_DIR, f"{photo.file_id}.jpg")
        await bot.download_file(file.file_path, destination=file_path)

        # Extract text using TrOCR
        extracted_text = extract_text_from_image(file_path)

        if extracted_text.strip():
            # Send text to user
            await message.answer(f"Ահա, խնդրեմ հանված տեքստը։ \n\n{extracted_text}")

            # Generate audio
            audio_response = generate_audio(extracted_text)

            if audio_response and "path" in audio_response:
                # Download audio file
                audio_path = audio_response.get("path")
                # Adjust this URL based on your actual API response structure
                base_url = "https://wav.am/download/TatevikMinasyan/239/"
                audio_file_name = os.path.basename(audio_path)
                full_audio_url = f"{base_url}{audio_file_name}"

                audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                audio_response = requests.get(full_audio_url)

                if audio_response.status_code == 200:
                    # Save audio file
                    with open(audio_file_path, 'wb') as audio_file:
                        audio_file.write(audio_response.content)

                    # Send audio file to user
                    with open(audio_file_path, 'rb') as audio_file:
                        await message.answer_voice(voice=audio_file)

                    # Clean up
                    os.remove(audio_file_path)
                else:
                    await message.answer("Հնարավոր չեղավ ներբեռնել ձայնային ֆայլը։")
            else:
                await message.answer("Հնարավոր չեղավ ստեղծել ձայնային ֆայլը։")
        else:
            await message.answer("Չհաջողվեց գտնել տեքստ նկարում։")

        # Clean up
        os.remove(file_path)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await message.answer("Խնդիր առաջացավ, խնդրում եմ, փորձել նորից։")


async def main() -> None:
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

    app.run(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
