from locust import HttpUser, task, between


class ReliabilityUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def simulate_traffic(self):
        self.client.get("/simulate-service")