import torch
import numpy as np

# t = torch.Tensor(1, 2, 3)
# t = torch.tensor([1, 2, 3])
#
# t = torch.tensor([[1, 2.0, 3], [4, 5, 2]], dtype=torch.float)

# dim=0 - столбцы, dim=1 - строки
# stroki = t.dim()
# size = t.size(), t.shape

# d_np = np.array([[1, 2, 3], [4, 5, 6]])
# t2 = torch.from_numpy(d_np)
# view(), reshape() - изменение формы

t_shape_by_random3x4 = torch.rand(3, 4)
t_shape_by_zero2x3x4 = torch.zeros(2, 3, 4)
t_shape_by_ones5x5 = torch.ones(5, 5)
t_shape_by_0_to15_4x4 = torch.arange(0, 16).view(4, 4)



t_A = torch.randint(1, 15, (3, 4))
t_B = t_A.reshape(4, 3)

t_A_trans = t_A.T
t_matmul = torch.matmul(t_A, t_B)
t_mul = t_A * t_B.T
t_A_sum = torch.sum(t_A)

# item() - получение элем из тенсора
# t[axis0, axis1]
# t[1, :], t[:, 1] - полный срез

t_shape5x5x5 = torch.arange(1, 126).view(5, 5, 5)

t_first_axis0 = t_shape5x5x5[:, 0, :]
t_lastaxis1 = t_shape5x5x5[:, :, -1]
t_shape2x2_center = t_shape5x5x5[2, 2:4, 2:4]
t_chet = t_shape5x5x5[t_shape5x5x5 % 2 == 0]



t_count24 = torch.arange(1, 25)

t_count24_shape2x12 = t_count24.view(2, 12)
t_count24_shape3x8 = t_count24.view(3, 8)
t_count24_shape4x6 = t_count24.view(4, 6)
t_count24_shape2x3x4 = t_count24.view(2, 3, 4)
t_count24_shape2x2x2x3 = t_count24.view(2, 2, 2, 3)

print()