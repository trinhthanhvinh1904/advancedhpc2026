from matplotlib.image import imread
from matplotlib import pyplot as plt
from numba import cuda
import numpy as np
import time

def preprocessing(rgb):
    image = imread(rgb)
    image_for_cuda = np.array(image, dtype=np.uint8)
    return image_for_cuda

@cuda.jit
def rgb2gray_gpu(pixels, gray_pixels):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y < pixels.shape[0] and x < pixels.shape[1]:
        r = pixels[y, x, 0]
        g = pixels[y, x, 1]
        b = pixels[y, x, 2]
        gray = np.uint8((r / 3.0) + (g / 3.0) + (b / 3.0))
        gray_pixels[y, x, 0] = gray
        gray_pixels[y, x, 1] = gray
        gray_pixels[y, x, 2] = gray

def rgb2gray_gpu_time(rgb, repeats = 5):
    original_image = preprocessing(rgb)
    height, width, channels = original_image.shape
    block_size = (32, 32)
    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]
    grid_size = (grid_x, grid_y)
    devScr = cuda.to_device(original_image)
    devDst = cuda.device_array((height, width, channels), dtype=np.uint8)

    #warmup
    rgb2gray_gpu[grid_size, block_size](devScr, devDst)
    cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(repeats):
        rgb2gray_gpu[grid_size, block_size](devScr, devDst)
    cuda.synchronize()
    end_time = time.perf_counter()
    runtime = (end_time - start_time) / repeats
    print(f"GPU time (block size {block_size}): {runtime} seconds")
    result = devDst.copy_to_host()
    return result, runtime

if __name__ == "__main__":
    rgb = "images.jpg"    
    gpu_result_image, gpu_time = rgb2gray_gpu_time(rgb)
    plt.imshow(gpu_result_image)
    plt.title(f"GPU Result\nTime: {gpu_time} seconds")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("result.jpg")
    plt.show()
