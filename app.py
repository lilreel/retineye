from PIL import Image
import numpy as np
import io
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('model_1.keras', custom_objects={
                                   'KerasLayer': hub.KerasLayer})

class_labels = {
    0: 'Cataracs',
    1: 'Diabetic Retinopathy',
    2: 'Glaucoma',
    3: 'Normal'
}


def preprocess_image(image_file):
    # Read the image from the file-like object
    image = Image.open(io.BytesIO(image_file.read()))

    # Resize the image
    image = image.resize((224, 224))

    # Convert the image to a numpy array and normalize it
    image_array = tf.keras.utils.img_to_array(image)

    # Add a batch dimension because the model expects batches of images
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def postprocess_prediction(prediction):
    # Assuming the prediction is a single class probability distribution
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    # Convert to a readable format
    response = {
        # 'predicted_class': int(predicted_class_index),
        'confidence': float(np.max(prediction)),
        'label': predicted_class_label
    }

    return response


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files.get('file')
        if image:
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Run model prediction
            prediction = model.predict(processed_image)

            # Process the prediction and return response
            response = postprocess_prediction(prediction)
            return jsonify(response)
        else:
            return jsonify({'message': 'No file provided'})
    return jsonify({'message': 'Invalid request method'})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
