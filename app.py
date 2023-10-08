from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io
import os
import secrets

app = Flask(__name__)

# Load your model
model = joblib.load('model_pkl.pkl')

# Define class labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Open the image using PIL
        img = Image.open(io.BytesIO(image.read()))
        # Resize the image to match the input shape (32, 32, 3)
        img = img.resize((32, 32))
        # Convert to numpy array and normalize
        img = np.array(img) / 255.0
        return img
    except Exception as e:
        return None  # Return None in case of any error

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image from the request
        uploaded_image = request.files['file']

        if uploaded_image.filename != '':
            # Generate a fixed filename for the uploaded image
            uploaded_filename = 'uploaded_image.jpg'

            # Preprocess the image
            image = preprocess_image(uploaded_image)

            if image is not None:
                try:
                    # Save the uploaded image with the fixed filename to the 'static' folder
                    image = Image.fromarray((image * 255).astype('uint8'))
                    image.save(os.path.join("static", uploaded_filename))

                    # Make predictions using your model
                    prediction = model.predict(np.array([image]))

                    # Convert the prediction to a human-readable class label
                    class_label = classes[np.argmax(prediction)]

                    # Redirect to the result page with the prediction and fixed image filename
                    return redirect(url_for('result', prediction=class_label, image_filename=uploaded_filename))
                except Exception as e:
                    # In case of an error, redirect to the result page with an error message
                    return redirect(url_for('result', prediction='Error processing image', image_filename='error.jpg'))
            else:
                # If there was an issue with image processing, redirect to the result page with an error message
                return redirect(url_for('result', prediction='Error processing image', image_filename='error.jpg'))
        else:
            # If no image is uploaded, redirect to the result page with a message
            return redirect(url_for('result', prediction='No image uploaded', image_filename='no_image.jpg'))

@app.route('/result')
def result():
    # Get the prediction and image filename from query parameters
    prediction = request.args.get('prediction', '')
    image_filename = request.args.get('image_filename', 'no_image.jpg')
    
    return render_template('result.html', prediction=prediction, image_filename=image_filename)

if __name__ == '__main__':
    app.run(debug=True)
