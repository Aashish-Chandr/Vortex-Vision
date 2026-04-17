"""
Lightweight mock VLM server for local development.
Implements the OpenAI chat completions API format so the real client code
works unchanged. Returns plausible placeholder responses.

In production, replace this with the real vLLM server (docker/vlm.Dockerfile).
"""
import json
import time
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8080

MOCK_RESPONSES = [
    "I can see several people moving through the area. No obvious anomalies detected.",
    "The scene shows normal pedestrian traffic. No suspicious activity observed.",
    "I observe vehicles moving at normal speeds. No accidents or incidents visible.",
    "The area appears clear. Normal activity patterns detected.",
    "Multiple individuals are present in the frame. Behavior appears normal.",
]


class MockVLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "model": "mock-vlm"})
        elif self.path == "/v1/models":
            self._respond(200, {
                "object": "list",
                "data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", "object": "model"}]
            })
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if self.path == "/v1/chat/completions":
            try:
                payload = json.loads(body)
                question = ""
                for msg in payload.get("messages", []):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if part.get("type") == "text":
                                question = part.get("text", "")
                    elif isinstance(content, str):
                        question = content

                # Generate a contextual mock response
                response_text = self._generate_response(question)

                self._respond(200, {
                    "id": f"chatcmpl-mock-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                })
            except Exception as e:
                self._respond(500, {"error": str(e)})
        else:
            self._respond(404, {"error": "endpoint not found"})

    def _generate_response(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["fight", "violence", "attack"]):
            return "I can see individuals in close proximity. There appears to be a physical altercation in progress near the center of the frame."
        if any(w in q for w in ["car", "vehicle", "speed", "traffic"]):
            return "Several vehicles are visible. One vehicle appears to be moving faster than the others in the left lane."
        if any(w in q for w in ["crowd", "people", "rush"]):
            return "A large group of people is visible. The crowd density appears elevated compared to normal patterns."
        if any(w in q for w in ["weapon", "gun", "knife"]):
            return "I cannot confirm the presence of weapons in these frames. The image quality makes it difficult to determine with certainty."
        return random.choice(MOCK_RESPONSES)

    def _respond(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MockVLMHandler)
    print(f"Mock VLM server running on port {PORT}")
    print("Replace with real vLLM server for production GPU inference")
    server.serve_forever()
