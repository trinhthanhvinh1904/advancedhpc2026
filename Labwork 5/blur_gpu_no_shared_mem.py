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
def gaussian_blur_gpu(pixels, blur_pixels):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    weights = (
    0, 0, 1, 2, 1, 0, 0,
    0, 3, 13, 22, 13, 3, 0,
    1, 13, 59, 97, 59, 13, 1,
    2, 22, 97, 159, 97, 22, 2,
    1, 13, 59, 97, 59, 13, 1,
    0, 3, 13, 22, 13, 3, 0,
    0, 0, 1, 2, 1, 0, 0)
    if y < pixels.shape[0] and x < pixels.shape[1]:
        r = 0.0
        g = 0.0
        b = 0.0
        for ky in range(-3, 4):
            for kx in range(-3, 4):
                ny = y + ky
                nx = x + kx
                if ny >= 0 and ny < pixels.shape[0] and nx >= 0 and nx < pixels.shape[1]:
                    weight = weights[(ky + 3) * 7 + (kx + 3)]
                    r += pixels[ny, nx, 0] * weight
                    g += pixels[ny, nx, 1] * weight
                    b += pixels[ny, nx, 2] * weight
        blur_pixels[y, x, 0] = np.uint8(r / 1003.0)
        blur_pixels[y, x, 1] = np.uint8(g / 1003.0)
        blur_pixels[y, x, 2] = np.uint8(b / 1003.0)

def gaussian_blur_gpu_time(rgb, repeats=5, block_size=(32,32)):
    original_image = preprocessing(rgb)
    height, width, channels = original_image.shape
    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]
    grid_size = (grid_x, grid_y)
    devScr = cuda.to_device(original_image)
    devDst = cuda.device_array((height, width, channels), dtype=np.uint8)

    # warmup
    gaussian_blur_gpu[grid_size, block_size](devScr, devDst)
    cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(repeats):
        gaussian_blur_gpu[grid_size, block_size](devScr, devDst)
        cuda.synchronize()
    end_time = time.perf_counter()
    runtime = (end_time - start_time) / repeats
    print(f"GPU time (block size {block_size}): {runtime} seconds")
    result = devDst.copy_to_host()
    return result, runtime

if __name__ == "__main__":
    rgb = "images.jpg"
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    for block_size in block_sizes:
        gpu_result_image, gpu_time = gaussian_blur_gpu_time(rgb, block_size=block_size)
        plt.figure()
        plt.imshow(gpu_result_image)
        plt.title(f"GPU Result ({block_size[0]}x{block_size[1]})\n"
                  f"Time: {gpu_time} seconds")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"result_{block_size[0]}x{block_size[1]}_no_shared.jpg")
        plt.close()