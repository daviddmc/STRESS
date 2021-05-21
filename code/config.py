import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# hyperparameters
use_sim = True
is_denoise = False
denoiser = None #"../results/epi_denoiser/epi_denoiser.pt" #"denoiser.pt"

model_name = "sim_s4_k0_n3" #"sim_s4_k2_n5"  #"epi_s4_k0"  # "sim_s6_k0_n3"
batch_size = 16 if is_denoise else 64 
num_iter = 1 # sim,split4,k2,noise0 0.033+ || sim,split4,k2,noise3 0.16+ 
lr = 1e-4

num_split = 4
use_k = 0 #num_split // 2

sigma = 0.03