from flask import Flask, request, jsonify
import main
from main import distance_x
app = Flask(__name__)

shared_variable = {"value": None}

@app.route('/set', methods=['POST'])
def set_variable():
    data = request.json
    shared_variable["value"] = data.get("value")
    return jsonify({"status": "OK", "received": shared_variable["value"]})

@app.route('/get', methods=['GET'])
def get_variable():
    value = main.distance_x
    return jsonify({"value": shared_variable["value"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
