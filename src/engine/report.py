# -*- coding: utf-8 -*-
"""
Model benchmark helper

The benchmark measures the parameter count, model size, inference latency, and throughput over a
fixed number of forward passes so every run records the efficiency of its trained network
"""

import time

import torch


def model_benchmark(model, device: str, img_size: int, runs: int = 20) -> dict:
    """
    Measure parameter count, model size, inference latency, and throughput

    Args:
        model: Detector to benchmark
        device: Device the model runs on
        img_size: Square input resolution used for the timed passes
        runs: Number of timed forward passes after a short warmup

    Returns:
        A dict with the parameter count, model size in megabytes, latency in milliseconds, and
        throughput in images per second
    """
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, 256, (1, 3, img_size, img_size)).float().to(device)

    with torch.no_grad():
        for _ in range(3):  # Warmup to stabilise the timing
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) / runs

    return {
        "parameters": int(parameters),
        "model_size_mb": round(parameters * 4 / 1e6, 3),
        "latency_ms": round(latency * 1000, 3),
        "throughput_ips": round(1.0 / latency, 2),
    }
