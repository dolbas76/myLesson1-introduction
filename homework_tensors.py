
import torch


x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)


basic_func = x**2 + y**2 + z**2 + 2*x*y*z


basic_func.backward()


print("--- Результаты расчета PyTorch ---")
print(f"Градиент по x (авто): {x.grad.item()}")  # Ожидаем 28.0
print(f"Градиент по y (авто): {y.grad.item()}")  # Ожидаем 22.0
print(f"Градиент по z (авто): {z.grad.item()}")  # Ожидаем 20.0

print("\n--- Проверка аналитически (наши формулы) ---")
grad_x_analytic = 2 * x.item() + 2 * y.item() * z.item()
grad_y_analytic = 2 * y.item() + 2 * x.item() * z.item()
grad_z_analytic = 2 * z.item() + 2 * x.item() * y.item()

print(f"Градиент по x (аналитически): {grad_x_analytic}")
print(f"Градиент по y (аналитически): {grad_y_analytic}")
print(f"Градиент по z (аналитически): {grad_z_analytic}")


x = torch.tensor([2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 4.0, 5.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y_pred = w * x + b

mse = torch.mean((y_pred - y_true) ** 2)

mse.backward()

print("--- Результаты расчета PyTorch ---")
print(f"MSE ошибка: {mse.item():.4f}")
print(f"Градиент по w (авто): {w.grad.item():.4f}")
print(f"Градиент по b (авто): {b.grad.item():.4f}")

# 6. Проверяем аналитически по нашим формулам
print("\n--- Проверка аналитически (наши формулы) ---")
n = len(x)
# Считаем по формулам: (2/n) * Сумма((y_pred - y_true) * x)
grad_w_analytic = (2 / n) * torch.sum((y_pred - y_true) * x)
grad_b_analytic = (2 / n) * torch.sum(y_pred - y_true)

print(f"Градиент по w (аналитически): {grad_w_analytic.item():.4f}")
print(f"Градиент по b (аналитически): {grad_b_analytic.item():.4f}")



x = torch.tensor(2.0, requires_grad=True)

f_x = torch.sin(x**2 + 1)

grad_x_auto = torch.autograd.grad(f_x, x)[0] #диффиренцируем f_x по x


print("--- Результаты расчета PyTorch ---")
print(f"Значение функции f(x): {f_x.item():.4f}")
print(f"Градиент df/dx (через autograd.grad): {grad_x_auto.item():.4f}")

# 4. Проверяем аналитически по нашей формуле "с листочка"
# Формула: 2 * x * cos(x^2 + 1)
grad_x_analytic = 2 * x.item() * torch.cos(x**2 + 1).item()

print("\n--- Проверка аналитически (наша формула) ---")
print(f"Градиент df/dx (аналитически): {grad_x_analytic:.4f}")