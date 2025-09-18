from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("california_model.joblib")

# Fixed feature order that matches your form names
FEATURES = ["MedInc", "AveRooms", "AveBedrms", "Population", "Households", "Latitude", "Longitude"]

@app.route("/")
def home():
    return render_template("index.html")  # templates/index.html

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Build row in the exact order
        X = [[float(data[f]) for f in FEATURES]]
        y = model.predict(X).tolist()
        return jsonify({"prediction": y, "unit": "hundred_thousands_USD"})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}. Required: {FEATURES}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



  



