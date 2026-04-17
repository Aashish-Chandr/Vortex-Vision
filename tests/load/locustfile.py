"""
Load tests using Locust.
Run with: locust -f tests/load/locustfile.py --host http://localhost:8000

Targets:
  - 100 concurrent users
  - Sub-200ms P95 response time on /events/ and /health/
"""
import random
from locust import HttpUser, between, task


class VortexVisionUser(HttpUser):
    wait_time = between(0.5, 2.0)
    token: str = ""

    def on_start(self):
        resp = self.client.post("/auth/token", json={
            "username": "viewer",
            "password": "vortex-viewer-pass",
        })
        if resp.ok:
            self.token = resp.json()["access_token"]

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(5)
    def health_check(self):
        self.client.get("/health/")

    @task(10)
    def list_events(self):
        self.client.get("/events/?limit=50", headers=self._headers())

    @task(3)
    def event_stats(self):
        self.client.get("/events/stats", headers=self._headers())

    @task(2)
    def list_streams(self):
        self.client.get("/streams/", headers=self._headers())

    @task(1)
    def nl_query(self):
        questions = [
            "Show me all people running in the last 5 minutes",
            "Were there any fights detected today?",
            "Show me red cars near the entrance",
            "Any suspicious activity in the last hour?",
        ]
        self.client.post(
            "/query/",
            json={
                "question": random.choice(questions),
                "time_window_seconds": 300,
            },
            headers=self._headers(),
            timeout=30,
        )
