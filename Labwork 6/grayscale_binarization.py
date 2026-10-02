import sys
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
def binarization_gpu(pixels, bin_pixels, threshold):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y < pixels.shape[0] and x < pixels.shape[1]:
        r = pixels[y, x, 0]
        g = pixels[y, x, 1]
        b = pixels[y, x, 2]
        gray = np.uint8((r + g + b) / 3.0)
        if gray < threshold:
            val = 0
        else:
            val = 255
        bin_pixels[y, x, 0] = val
        bin_pixels[y, x, 1] = val
        bin_pixels[y, x, 2] = val

def binarization_gpu_time(rgb, threshold = 128, repeats = 5, block_size = (32,32)):
    if threshold > 255 or threshold < 0:
        raise ValueError("Threshold must be between 0 and 255")
    original_image = preprocessing(rgb)
    height, width, channels = original_image.shape
    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]
    grid_size = (grid_x, grid_y)
    devScr = cuda.to_device(original_image)
    devDst = cuda.device_array((height, width, channels), dtype=np.uint8)

    #warmup
    binarization_gpu[grid_size, block_size](devScr, devDst, threshold)
    cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(repeats):
        binarization_gpu[grid_size, block_size](devScr, devDst, threshold)
        cuda.synchronize()
    end_time = time.perf_counter()
    runtime = (end_time - start_time) / repeats
    print(f"GPU time (block size {block_size}): {runtime} seconds")
    result = devDst.copy_to_host()
    return result, runtime

if __name__ == "__main__":
    threshold = 128
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
        except ValueError:
            print(f"'{sys.argv[1]}' is not valid. Using default: 128")
            threshold = 128
    rgb = "images.jpg"
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    results = []
    for block_size in block_sizes:
        gpu_result_image, gpu_time = binarization_gpu_time(rgb, threshold=threshold, block_size=block_size)
        results.append((block_size, gpu_result_image, gpu_time))
    figure = plt.figure(figsize=(12, 10))
    layout = figure.add_gridspec(2, 4)
    axes = [
        figure.add_subplot(layout[0, 0:2]),
        figure.add_subplot(layout[0, 2:4]),
        figure.add_subplot(layout[1, 1:3]),
    ]
    for axis, (block_size, gpu_result_image, gpu_time) in zip(axes, results):
        if gpu_result_image.ndim == 2:
            axis.imshow(gpu_result_image, cmap="gray")
        else:
            axis.imshow(gpu_result_image)
        axis.set_title(
            f"Binarization result ({block_size[0]}x{block_size[1]})\n"
            f"Threshold: {threshold} | Time: {gpu_time} s"
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig("binarization_results.jpg")
    plt.close(figure)
