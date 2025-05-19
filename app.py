import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Зареждаме модела при стартиране на приложението.
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if "features" not in data:
            return jsonify({"error": "Полетo 'features' липсва!"}), 400

        features = data["features"]
        if len(features) != 8:
            return jsonify({"error": "Неправилен брой характеристики, трябва да са 8!"}), 400

        # Прогнозиране
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
