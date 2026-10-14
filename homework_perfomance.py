# в testing нейро код с выводом результатов



import torch
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство - {device}")

tensor_configs = [
        ("64x1024x1024", (64, 1024, 1024)),
        ("128x512x512", (128, 512, 512)),
        ("256x256x256", (256, 256, 256))
    ]

tensors_cpu = []
tensors_gpu = []

for name, shape in tensor_configs:

    A_cpu = torch.randn(*shape, dtype=torch.float)
    B_cpu = torch.randn(*shape, dtype=torch.float)

    tensors_cpu.append((name, A_cpu, B_cpu))

    if (torch.cuda.is_available()):
        A_cpu = A_cpu.to(device)
        B_cpu = B_cpu.to(device)
        tensors_gpu.append((name, A_cpu, B_cpu))


    def measure_cpu_time(operation, *args, num_runs=10):
        #прогрев для работы процессора
        for _ in range(2):
            _ = operation(*args)

        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            i = operation(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000) #мс

        return sum(times) / len(times)

    def measure_gpu_time(operation, *args, num_runs=10):
        if(not torch.cuda.is_available()):
            return None;

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(2):
            _ = operation(*args)

        times = []
        for i in range(num_runs):
            start_event.record()
            i = operation(*args)
            end_event.record()

            torch.cuda.synchronize() #синхронизация cpu gpu
            times.append((end_event.elapsed_time(start_event) * 1000))

        return sum(times) / len(times)


    operations = {
        "Матричное умножение": lambda A, B: torch.matmul(A, B),
        "Поэлементное сложение": lambda A, B: A + B,
        "Поэлементное умножение": lambda A, B: A * B,
        "Транспонирование": lambda A, B: A.transpose(1, 2).contiguous(),
        "Сумма элементов": lambda A, B: A.sum()
    }


    for idx, (name, A_cpu, B_cpu) in enumerate(tensors_cpu):

        for op_name, op_func in operations.items():
            cpu_time = measure_cpu_time(op_func, A_cpu, B_cpu)

            if torch.cuda.is_available():
                name, A_gpu, B_gpu = tensors_gpu[idx]
                gpu_time = measure_gpu_time(op_func, A_gpu, B_gpu)
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"{op_name:<25} | {cpu_time:<12.2f} | {gpu_time:<12.2f} | {speedup:<12.2f}x")
            else:
                print(f"{op_name:<25} | {cpu_time:<12.2f} | {'N/A':<12} | {'N/A':<12}")

print()