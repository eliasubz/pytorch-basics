import torch

# import triton
# import triton.language as tl
# from triton.runtime import driver

# DEVICE = triton.runtime.driver.active.get_active_torch_device()

# def is_hip():
#     return triton.runtime.driver.active.get_current_target().backend == "hip"


# def is_cdna():
#     return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
#                                                                                    'gfx90a', 'gfx908')

def naive_softmax(x):
    # x (m, n)
    max = torch.max(x, dim=1).values[:, None] # m: max of each row because n is collapsed
    x_d = x - max # (m,n): max is broadcasted

    exp = torch.exp(x_d)

    sum_exps = torch.sum(exp, dim=1) # (m); sum of each row in exp (m,n)

    smaxed = exp / sum_exps[:, None] 

    return smaxed # (m,n) each row should sum to one

x = torch.randn((15,5))

gt = torch.softmax(x, 1)
mt = naive_softmax(x)

print(gt - mt)