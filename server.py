from flask import Flask, jsonify, request

from backend_core import generate_from_payload


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    result = generate_from_payload(data)
    return jsonify(result["body"]), result["status"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

