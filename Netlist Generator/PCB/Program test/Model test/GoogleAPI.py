import os
import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account

# === CONFIGURATION ===
IMAGE_PATH = "Test images/circuit003.png"  # <-- Change this!
OUTPUT_PATH = "letter_boxes_output.png"
CREDENTIALS_PATH = "credentials.json"       # <-- Your service account file

# === SETUP ===
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = vision.ImageAnnotatorClient(credentials=credentials)

# === LOAD IMAGE ===
with open(IMAGE_PATH, "rb") as img_file:
    content = img_file.read()

image = vision.Image(content=content)
response = client.document_text_detection(image=image)
annotation = response.full_text_annotation

# Load image for drawing
cv_img = cv2.imread(IMAGE_PATH)

# === DRAW BOUNDING BOXES AROUND EACH SYMBOL ===
for page in annotation.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                for symbol in word.symbols:
                    vertices = [(v.x, v.y) for v in symbol.bounding_box.vertices]
                    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(cv_img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

# === SAVE AND SHOW ===
cv2.imwrite(OUTPUT_PATH, cv_img)
print(f"Output image saved to: {OUTPUT_PATH}")
cv2.imshow("Letters Boxed", cv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
