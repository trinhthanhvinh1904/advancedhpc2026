from numba import cuda

def check_gpu():
    device = cuda.get_current_device()
    print(f"Device Name: {device.name}")
    print(f"Device ID: {device.id}")
    print(f"Device  Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")
    print(f"Device Compute Capability: {device.compute_capability}")
    print(f"Device memory size: {cuda.current_context().get_memory_info()}")

check_gpu()

