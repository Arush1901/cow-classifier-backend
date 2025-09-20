from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f]

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    image_bytes = base64.b64decode(data["image"])
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # Find best prediction
    max_index = int(np.argmax(output_data))
    prediction = {
        "breed": labels[max_index],
        "confidence": float(output_data[max_index]),
    }

    return jsonify(prediction)

@app.route("/api/health")
def health():
    return jsonify({
        "status": "Server is running 🚀",
        "labels": labels
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
