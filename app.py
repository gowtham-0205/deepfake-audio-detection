from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import model_from_json
import librosa
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)

model.load_weights('model_weights.weights.h5')

def preprocess_audio(audio_file, max_length=500):
    try:
        # Load the audio file at a sample rate of 16,000 Hz
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Extract MFCC features (40 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Pad or trim the MFCC array to the fixed length of max_length
        if mfccs.shape[1] < max_length:
            # Pad with zeros if there are fewer than max_length time steps
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            # Truncate to the first max_length time steps if there are more
            mfccs = mfccs[:, :max_length]
        
        # Reshape to match the input shape of the model (adding channel dimension)
        mfccs = np.expand_dims(mfccs, axis=-1)  # Shape becomes (40, max_length, 1)
        
        # Return as a batch of 1 (shape becomes (1, 40, max_length, 1))
        return np.array([mfccs])  
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def predict_audio(file_path):
    audio = preprocess_audio(file_path)
    prediction = model.predict(audio)
    return 'fake' if prediction[0][0] > 0.5 else 'real'

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = 'uploads/' + file.filename
    file.save(file_path)
    
    # Preprocess the audio file
    features = preprocess_audio(file_path)
    
    if features is not None:
        # Make prediction (0 = real, 1 = fake)
        prediction = model.predict(features)
        label = "Fake" if prediction[0][0] > 0.5 else "Real"
        
        # Render the result in a webpage
        return render_template('result.html', result=label)
    else:
        return jsonify({'error': 'Error processing audio file'}), 400


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
