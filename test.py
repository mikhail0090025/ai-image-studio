import torch
import time

# -------------------------------
# Configuration
# -------------------------------

SIZE = 1000
ITERATIONS = 20

dtypes = [
    ("float64", torch.float64),
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# Добавляем Float8 только если они существуют
if hasattr(torch, "float8_e4m3fn"):
    dtypes.append(("float8_e4m3fn", torch.float8_e4m3fn))

if hasattr(torch, "float8_e5m2"):
    dtypes.append(("float8_e5m2", torch.float8_e5m2))

print(f"PyTorch version: {torch.__version__}")
print(f"CPU: {torch.get_num_threads()} threads\n")

results = []

for name, dtype in dtypes:

    print("=" * 60)
    print(f"Testing {name}")

    try:
        # Создаем данные сначала в float32
        a = torch.randn(SIZE, SIZE, dtype=torch.float32)
        b = torch.randn(SIZE, SIZE, dtype=torch.float32)

        # Конвертируем
        a = a.to(dtype)
        b = b.to(dtype)

        # Прогрев
        _ = a @ b

        start = time.perf_counter()

        for _ in range(ITERATIONS):
            c = a @ b

        end = time.perf_counter()

        elapsed = end - start
        avg = elapsed / ITERATIONS

        memory_mb = (
            a.element_size() * a.numel() * 2
        ) / (1024 ** 2)

        print(f"Element size : {a.element_size()} bytes")
        print(f"Matrix memory: {memory_mb:.2f} MB")
        print(f"Total time   : {elapsed:.4f} s")
        print(f"Average      : {avg:.6f} s")

        results.append((name, memory_mb, avg))

    except Exception as e:
        print("FAILED")
        print(type(e).__name__, e)

print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

for name, mem, avg in results:
    print(f"{name:15}  {mem:8.2f} MB   {avg:.6f} s")