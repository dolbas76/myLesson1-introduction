import torch
import time


def main():
    print("=" * 70)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ CPU vs CUDA")
    print("=" * 70)

    # Проверяем доступность CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуемое устройство: {device}")

    if not torch.cuda.is_available():
        print("⚠️  CUDA недоступна! Результаты будут только для CPU.")

    # ========================================================================
    # 3.1 ПОДГОТОВКА ДАННЫХ (5 баллов)
    # ========================================================================
    print("\n" + "=" * 70)
    print("3.1 ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)

    # Создаём три набора тензоров разных размеров
    tensor_configs = [
        ("64x1024x1024", (64, 1024, 1024)),
        ("128x512x512", (128, 512, 512)),
        ("256x256x256", (256, 256, 256))
    ]

    tensors_cpu = []
    tensors_gpu = []

    for name, shape in tensor_configs:
        print(f"\nСоздание тензоров размера {name}...")

        # Создаём пару тензоров на CPU
        A_cpu = torch.randn(*shape, dtype=torch.float32)
        B_cpu = torch.randn(*shape, dtype=torch.float32)

        tensors_cpu.append((name, A_cpu, B_cpu))

        # Копируем на GPU если доступно
        if torch.cuda.is_available():
            A_gpu = A_cpu.to(device)
            B_gpu = B_cpu.to(device)
            tensors_gpu.append((name, A_gpu, B_gpu))
            print(f"  ✓ Создано на CPU и скопировано на GPU")
            print(f"    Размер: {A_cpu.element_size() * A_cpu.nelement() / (1024 ** 2):.2f} MB")
        else:
            print(f"  ✓ Создано только на CPU")

    # ========================================================================
    # 3.2 ФУНКЦИЯ ИЗМЕРЕНИЯ ВРЕМЕНИ (5 баллов)
    # ========================================================================
    print("\n" + "=" * 70)
    print("3.2 ФУНКЦИЯ ИЗМЕРЕНИЯ ВРЕМЕНИ")
    print("=" * 70)

    def measure_cpu_time(operation, *args, num_runs=10):
        """Измеряет время выполнения операции на CPU"""
        # Warm-up (прогрев)
        for _ in range(2):
            _ = operation(*args)

        # Измерение
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = operation(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Конвертируем в миллисекунды

        return sum(times) / len(times)  # Среднее время

    def measure_gpu_time(operation, *args, num_runs=10):
        """Измеряет время выполнения операции на GPU"""
        if not torch.cuda.is_available():
            return None

        # Создаём события для тайминга
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up
        for _ in range(2):
            _ = operation(*args)
            torch.cuda.synchronize()

        # Измерение
        times = []
        for _ in range(num_runs):
            start_event.record()
            _ = operation(*args)
            end_event.record()

            # Синхронизация и измерение
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        return sum(times) / len(times)

    print("✓ Функции измерения времени созданы")
    print("  - measure_cpu_time: использует time.perf_counter()")
    print("  - measure_gpu_time: использует torch.cuda.Event()")

    # ========================================================================
    # 3.3 СРАВНЕНИЕ ОПЕРАЦИЙ (10 баллов)
    # ========================================================================
    print("\n" + "=" * 70)
    print("3.3 СРАВНЕНИЕ ОПЕРАЦИЙ")
    print("=" * 70)

    # Определяем операции для тестирования
    operations = {
        "Матричное умножение": lambda A, B: torch.matmul(A, B),
        "Поэлементное сложение": lambda A, B: A + B,
        "Поэлементное умножение": lambda A, B: A * B,
        "Транспонирование": lambda A, B: A.transpose(1, 2).contiguous(),
        "Сумма элементов": lambda A, B: A.sum()
    }

    # Для каждого размера тензоров
    for idx, (name, A_cpu, B_cpu) in enumerate(tensors_cpu):
        print(f"\n{'=' * 70}")
        print(f"Размер тензоров: {name}")
        print(f"{'=' * 70}")

        # Заголовок таблицы
        print(f"\n{'Операция':<25} | {'CPU (мс)':<12} | {'GPU (мс)':<12} | {'Ускорение':<12}")
        print("-" * 70)

        # Для каждой операции
        for op_name, op_func in operations.items():
            # Измеряем на CPU
            cpu_time = measure_cpu_time(op_func, A_cpu, B_cpu)

            # Измеряем на GPU
            if torch.cuda.is_available():
                _, A_gpu, B_gpu = tensors_gpu[idx]
                gpu_time = measure_gpu_time(op_func, A_gpu, B_gpu)
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"{op_name:<25} | {cpu_time:<12.2f} | {gpu_time:<12.2f} | {speedup:<12.2f}x")
            else:
                print(f"{op_name:<25} | {cpu_time:<12.2f} | {'N/A':<12} | {'N/A':<12}")

    # ========================================================================
    # 3.4 АНАЛИЗ РЕЗУЛЬТАТОВ (5 баллов)
    # ========================================================================
    print("\n" + "=" * 70)
    print("3.4 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    print("""
📊 КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:

1. НАИБОЛЬШЕЕ УСКОРЕНИЕ:
   ✓ Матричное умножение (10-100x ускорение)
   ✓ Причина: GPU имеет тысячи CUDA-ядер, оптимизированных для параллельных
     матричных операций. Библиотеки вроде cuBLAS максимально используют
     архитектуру GPU для вычислений с плавающей точкой.

2. ОПЕРАЦИИ С МАЛЫМ УСКОРЕНИЕМ:
   ⚠ Поэлементные операции (сложение, умножение): 2-10x ускорение
   ⚠ Причина: эти операции memory-bound (ограничены пропускной способностью
     памяти), а не вычислительной мощностью. GPU не даёт преимущества, когда
     узкое место — скорость чтения/записи данных.

   ⚠ Транспонирование: может быть быстрее на CPU
   ⚠ Причина: в PyTorch transpose часто создаёт view (представление) без
     копирования данных. Это O(1) операция на CPU. На GPU может потребоваться
     реальное копиров данных для обеспечения contiguous memory layout.

3. ВЛИЯНИЕ РАЗМЕРА МАТРИЦ:
   📈 Большие матрицы (1024×1024): максимальное ускорение на GPU
   📈 Маленькие матрицы (256×256): меньшее ускорение
   📈 Причина: для больших матриц amortized overhead (амортизированные затраты
     на инициализацию) меньше, а параллелизм используется эффективнее.

4. ПЕРЕДАЧА ДАННЫХ CPU ↔ GPU:
   ⚡ Передача данных через PCIe bus занимает время (обычно 1-10 мс)
   ⚡ В реальных задачах данные остаются на GPU, чтобы избежать повторных передач
   ⚡ Для маленьких batch'ов overhead передачи может превышать выигрыш от GPU
   ⚡ Best practice: загружать данные на GPU один раз и держать там

💡 РЕКОМЕНДАЦИИ ДЛЯ PRODUCTION:
   • Используйте GPU для вычислительно сложных операций (matmul, convolutions)
   • Избегайте частых передач данных между CPU и GPU
   • Для маленьких тензоров CPU может быть быстрее (меньше overhead)
   • Всегда профилируйте код на реальных данных, а не только на синтетических
""")

    print("=" * 70)
    print("ЗАДАНИЕ ВЫПОЛНЕНО УСПЕШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()