from matplotlib.image import imread
from matplotlib import pyplot as plt
from numba import cuda
import numpy as np
import time

def preprocessing(rgb):
    image = imread(rgb)
    image_for_cuda = np.array(image, dtype=np.uint8)
    height, width, channels = image_for_cuda.shape
    pixels = image_for_cuda.reshape(-1, channels)
    return pixels, image_for_cuda

def rgb2gray_cpu(rgb):
    pixels = preprocessing(rgb)[0]
    for i in range(pixels.shape[0]):
        r, g, b = pixels[i]
        gray = np.uint8((r / 3.0) + (g / 3.0) + (b / 3.0))
        pixels[i] = [gray, gray, gray]
    return pixels

def rgb2gray_cpu_time(rgb):
    start_time = time.time()
    pixels = rgb2gray_cpu(rgb)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"CPU time: {runtime} seconds")
    return pixels, runtime

@cuda.jit
def rgb2gray_gpu(pixels, gray_pixels):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < pixels.shape[0]:
        r = pixels[tidx, 0]
        g = pixels[tidx, 1]
        b = pixels[tidx, 2]
        gray = np.uint8((r / 3.0) + (g / 3.0) + (b / 3.0))
        gray_pixels[tidx, 0] = gray
        gray_pixels[tidx, 1] = gray
        gray_pixels[tidx, 2] = gray

def rgb2gray_gpu_time(rgb):
    pixels = preprocessing(rgb)[0]
    pixels_count = pixels.shape[0]
    block_size = 64
    grid_size = (pixels_count + block_size - 1) // block_size
    devScr = cuda.to_device(pixels)
    devDst = cuda.device_array((pixels_count, 3), dtype=np.uint8)
    start_time = time.time()
    rgb2gray_gpu[grid_size, block_size](devScr, devDst)
    cuda.synchronize()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"GPU time: {runtime} seconds")
    return devDst.copy_to_host(), runtime

if __name__ == "__main__":
    rgb = "images.jpg"    
    cpu_result, cpu_time = rgb2gray_cpu_time(rgb)
    _, original_image = preprocessing(rgb)
    height, width, channels = original_image.shape
    cpu_result_image = cpu_result.reshape(height, width, 3)  
    gpu_result, gpu_time = rgb2gray_gpu_time(rgb)
    gpu_result_image = gpu_result.reshape(height, width, 3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cpu_result_image)
    axes[0].set_title(f"CPU Result\nTime: {cpu_time} seconds")
    axes[0].axis("off")
    axes[1].imshow(gpu_result_image)
    axes[1].set_title(f"GPU Result\nTime: {gpu_time} seconds")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()