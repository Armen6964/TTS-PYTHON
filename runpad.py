from paddleocr import PaddleOCR

# Initialize PaddleOCR with custom Armenian dictionary
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    rec_char_dict_path='./armenian_dict_padded_662.txt',
    det_model_dir='models/det',
    rec_model_dir='models/rec'
)

# Run OCR on an image
image_path = 'static/a.png'
results = ocr.ocr(image_path, cls=True)

print(results)

# Print recognized text
for line in results[0]:
    text = line[1][0]
    confidence = line[1][1]
    print(f"Text: {text}, Confidence: {confidence:.2f}")
