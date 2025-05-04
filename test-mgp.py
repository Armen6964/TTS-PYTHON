from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
import requests
from PIL import Image

processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

# load image from the IIIT-5k dataset
url = "https://aramhaytyan.wordpress.com/wp-content/uploads/2017/10/d0b1d0b5d0b7-d0bdd0b0d0b7d0b2d0b0d0bdd0b8d18f.jpg?w=301"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
outputs = model(pixel_values)

generated_text = processor.batch_decode(outputs.logits)['generated_text']

print(generated_text)