import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count())  # Check how many devices are detected
print(torch.cuda.get_device_name(0))  # Print the name of the detected GPU
