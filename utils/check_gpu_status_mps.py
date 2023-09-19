import torch 


if torch.backends.mps.is_available(): 
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device) 
    print("MPS device is available. Successfully initiated. ")
    print(x) 
else: 
    print("mps device is not found.")
