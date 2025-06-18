import torch 

cpu_gen = torch.random.default_generator

print(cpu_gen.get_state())