import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Dict, Any

from backend_core import generate_from_payload


def build_cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Max-Age": "86400",
    }


class handler(BaseHTTPRequestHandler):
    def _send_response(self, status: int, body: Dict[str, Any]):
        response_bytes = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(response_bytes)),
            **build_cors_headers(),
        }
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        for key, value in build_cors_headers().items():
            self.send_header(key, value)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length") or 0)
        data_bytes = self.rfile.read(content_length)
        try:
            payload = json.loads(data_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_response(
                HTTPStatus.BAD_REQUEST,
                {"text": "", "error": "invalid_json_payload"},
            )
            return

        result = generate_from_payload(payload)
        self._send_response(result["status"], result["body"])

