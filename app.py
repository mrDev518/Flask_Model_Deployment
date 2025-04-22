from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load your dataset and train the model
df = pd.read_csv('Student_performance_data .csv')

X = df.drop("GradeClass", axis=1)
y = df["GradeClass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return jsonify({"message": "Model deployment is successful!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive input as JSON
    input_features = pd.DataFrame([data])  # Convert input into DataFrame
    
    # Make prediction using the trained model
    prediction = model.predict(input_features)
    
    # Return the prediction as a response
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
