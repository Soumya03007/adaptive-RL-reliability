from prometheus_client import Gauge, Counter, generate_latest

# Core system metrics
latency_gauge = Gauge("rl_latency", "Current service latency")
cpu_gauge = Gauge("rl_cpu", "Current CPU usage")
error_gauge = Gauge("rl_error_rate", "Current error rate")
traffic_gauge = Gauge("rl_traffic", "Current incoming traffic")

# RL intelligence metrics
reward_gauge = Gauge("rl_reward", "Current RL reward")
healthy_ratio_gauge = Gauge("rl_healthy_ratio", "Healthy ratio")

# Replica / scaling metrics
instance_gauge = Gauge("rl_instances", "Current instance count")

# Action frequency metrics
action_counter = Counter(
    "rl_action_total",
    "Total RL actions",
    ["action"]
)