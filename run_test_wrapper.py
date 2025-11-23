import subprocess
import sys
import os

cwd = r"c:\Users\eliad\PycharmProjects\evolution-strategy-baselines-comparison"

try:
    result = subprocess.run(
        [sys.executable, "-m", "deep_noise_modeling.test_vae_unet"],
        capture_output=True,
        text=True,
        cwd=cwd
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    with open("test_output.txt", "w") as f:
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
except Exception as e:
    print(f"Wrapper failed: {e}")
