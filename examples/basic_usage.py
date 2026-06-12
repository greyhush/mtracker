"""Example: Using mtracker with a training loop."""

import random
from mtracker import Run


def simulate_training():
    """Simulate a training loop with mtracker tracking."""

    with Run(
        name="example-training",
        project="demo",
        config={
            "model": "Qwen3.5-9B",
            "lr": 5e-5,
            "batch_size": 16,
            "epochs": 3,
            "lora_rank": 32,
        },
        tags=["qlora", "example"],
    ) as run:
        loss = 1.0
        for epoch in range(3):
            for step in range(100):
                global_step = epoch * 100 + step

                # Simulate training
                loss *= random.uniform(0.98, 1.02)
                loss *= 0.999  # Overall decreasing trend
                acc = min(0.99, 1.0 - loss)
                lr = 5e-5 * (1 - global_step / 300)

                # Log multiple metrics at once
                run.log_dict(
                    {
                        "loss": round(loss, 4),
                        "accuracy": round(acc, 4),
                        "learning_rate": lr,
                    },
                    step=global_step,
                )

                if global_step % 50 == 0:
                    print(f"Step {global_step}: loss={loss:.4f}, acc={acc:.4f}")

        run.set_notes("Example training run completed successfully")
        print(f"\nRun {run.id} completed!")


if __name__ == "__main__":
    simulate_training()
