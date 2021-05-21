import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# hyperparameters
use_sim = True
is_denoise = False
denoiser = None

model_name = "sim_s4_k3_n3"
batch_size = 16 if is_denoise else 64 
num_iter = 100000
lr = 1e-4

num_split = 4
use_k = num_split // 2

sigma = 0.03
