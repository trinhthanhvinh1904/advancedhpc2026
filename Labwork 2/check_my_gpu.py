from numba import cuda

def check_gpu():
    device = cuda.get_current_device()
    cores_per_sm = {
            (1, 0): 8, (1, 1): 8, (1, 2): 8, (1, 3): 8,
            (2, 0): 32, (2, 1): 48,
            (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
            (5, 0): 128, (5, 2): 128, (5, 3): 128,
            (6, 0): 64, (6, 1): 128, (6, 2): 128,
            (7, 0): 64, (7, 2): 64, (7, 5): 64,
            (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
            (9, 0): 128, (9, 0): 128 
        }
    
    sm_count = device.MULTIPROCESSOR_COUNT
    compute_capability = device.compute_capability
    cores_per_sm_value = cores_per_sm.get(compute_capability, 0)
    total_cores = sm_count * cores_per_sm_value
    
    print(f"Device Name: {device.name}")
    print(f"Device ID: {device.id}")
    print(f"Device  Multiprocessor Count: {sm_count}")
    print(f"Device Total Cores: {total_cores}")
    print(f"Device Compute Capability: {compute_capability}")
    print(f"Device memory size: {cuda.current_context().get_memory_info()}")

check_gpu()

