import torch
import time
from ptflops import get_model_complexity_info
from codecarbon import EmissionsTracker

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    start_time = time.time()

    tracker = EmissionsTracker(
        gpu_ids=[],
        log_level='critical',
        save_to_file=False
    )
    tracker.start()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    emissions = tracker.stop()
    inference_time = time.time() - start_time
    accuracy = 100 * correct / total

    # Energy consumption
    energy_kwh = tracker.final_emissions_data.energy_consumed

    # FLOPS + Params
    macs, params = get_model_complexity_info(
        model,
        (1, 48, 48),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    return {
        "accuracy": round(accuracy, 2),
        "inference_time_sec": round(inference_time, 2),
        "energy_consumed_kwh": round(energy_kwh, 6),
        "co2_emissions_kg": round(emissions, 6),
        "flops": macs,
        "parameters": params,
    }
