from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = "model.pkl"
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'csv'}
THRESHOLD_DAYS = 30  # Default: Recommend throwing out balls after 30 days

# Inventory and metrics data
inventory = []  # Stores each ball with its date_added
usage_logs = []  # Tracks usage logs
co2_metrics = {"total_emissions": 0.0, "reduced_emissions": 0.0}  # Tracks CO₂ emissions

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load or train ML model
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Default training data
        data = pd.DataFrame({"age": [10, 20, 30, 40, 50], "condition": ["usable", "usable", "throw", "throw", "throw"]})
        X, y = data[["age"]], data["condition"]
        model = DecisionTreeClassifier()
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        return model

model = load_or_train_model()

@app.route('/')
def home():
    today = datetime.now().date()
    advice_data = []
    balls_to_recycle = 0  # Track balls that should be recycled

    for ball in inventory:
        age = (today - ball["date_added"]).days
        condition = model.predict([[age]])[0] if len(inventory) > 0 else "usable"
        advice_data.append({"age": age, "status": condition.capitalize()})
        if condition == "throw":
            balls_to_recycle += 1

    # Check if usage_data.csv exists
    if not os.path.exists("usage_data.csv"):
        message = "No data file detected. Upload a file or add some balls to start tracking."
    else:
        message = f"{balls_to_recycle} ball(s) need to be recycled today."

    return render_template('index.html', advice=advice_data, message=message)

@app.route('/inventory', methods=['GET', 'POST'])
def inventory_page():
    global inventory
    if request.method == 'POST':
        action = request.form.get('action')
        quantity = int(request.form.get('quantity', 0))
        today = datetime.now().date()

        if action == 'add':
            for _ in range(quantity):
                inventory.append({"date_added": today})
            co2_metrics['total_emissions'] += quantity * 1.2  # 1.2 lbs CO₂ per ball
            usage_logs.append({"action": "added", "quantity": quantity, "date": today})
        elif action == 'recycle':
            quantity_to_recycle = min(quantity, len(inventory))
            recycled_balls = inventory[:quantity_to_recycle]
            inventory = inventory[quantity_to_recycle:]  # Remove recycled balls from inventory
            co2_metrics['reduced_emissions'] += quantity_to_recycle * 1.2  # Subtract CO₂ emissions
            usage_logs.append({"action": "recycled", "quantity": quantity_to_recycle, "date": today})
        return redirect(url_for('inventory_page'))

    # Calculate ball ages for display
    today = datetime.now().date()
    for ball in inventory:
        ball["age"] = (today - ball["date_added"]).days

    return render_template('inventory.html', inventory=inventory, co2_metrics=co2_metrics)

@app.route('/usage', methods=['GET', 'POST'])
def usage_page():
    if request.method == 'POST':
        notes = request.form.get('notes', 'No details provided.')
        usage_logs.append({"action": "log", "details": notes, "date": datetime.now().date()})
        return redirect(url_for('usage_page'))
    return render_template('usage.html', logs=usage_logs)

@app.route('/metrics', methods=['GET'])
def metrics_page():
    net_emissions = co2_metrics['total_emissions'] - co2_metrics['reduced_emissions']
    return render_template('metrics.html', metrics=co2_metrics, net_emissions=net_emissions)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = "usage_data.csv"
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return redirect(url_for('retrain_model'))
    else:
        return jsonify({"message": "Invalid file type. Please upload a .csv file."})

@app.route('/retrain', methods=['POST', 'GET'])
def retrain_model():
    try:
        # Retrain the model with updated data
        if not os.path.exists("usage_data.csv"):
            return jsonify({"message": "No usage data found. Add some balls or upload a file to generate data."})

        data = pd.read_csv('usage_data.csv', names=['age', 'condition'])
        X, y = data[["age"]], data["condition"]
        global model
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        return jsonify({"message": "Model retrained successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0')

