import sys
from matplotlib.image import imread
from matplotlib import pyplot as plt
from numba import cuda
import numpy as np
import time

def preprocessing(img):
    img = imread(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    image_for_cuda = np.array(img, dtype=np.uint8)
    return image_for_cuda

@cuda.jit
def blend_gpu(img1, img2, out_img, c):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y < img1.shape[0] and x < img1.shape[1]:
        for ch in range(img1.shape[2]):
            val1 = img1[y, x, ch]
            val2 = img2[y, x, ch]
            blended_val = (c * val1) + ((1.0 - c) * val2)
            out_img[y, x, ch] = min(255, max(0, int(blended_val)))

def blend_gpu_time(img1_path, img2_path, c=0.5, repeats=5, block_size=(32,32)):
    if not (0.0 <= c <= 1.0):
        raise ValueError("weight c must be between 0 and 1")
    img1_arr = preprocessing(img1_path)
    img2_arr = preprocessing(img2_path)
    if img1_arr.shape != img2_arr.shape:
        raise ValueError(f"2 images must have the same shape")
    height, width, channels = img1_arr.shape
    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]
    grid_size = (grid_x, grid_y)
    dev_img1 = cuda.to_device(img1_arr)
    dev_img2 = cuda.to_device(img2_arr)
    dev_dst = cuda.device_array((height, width, channels), dtype=np.uint8)

    # Warmup
    blend_gpu[grid_size, block_size](dev_img1, dev_img2, dev_dst, c)
    cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(repeats):
        blend_gpu[grid_size, block_size](dev_img1, dev_img2, dev_dst, c)
        cuda.synchronize()
    end_time = time.perf_counter()
    runtime = (end_time - start_time) / repeats
    print(f"GPU time (block size {block_size}): {runtime:.6f} seconds")
    result = dev_dst.copy_to_host()
    if channels == 1:
        result = result.squeeze(axis=-1)
    return result, runtime

if __name__ == "__main__":
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"
    c = 0.5
    if len(sys.argv) >= 3:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
    if len(sys.argv) >= 4:
        try:
            c = float(sys.argv[3])
        except ValueError:
            print(f"'{sys.argv[3]}' is not valid for weight c, using default (0.5)")
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    results = []
    for block_size in block_sizes:
        gpu_result_image, gpu_time = blend_gpu_time(img1_path, img2_path, c=c, block_size=block_size)
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
            f"Blend result ({block_size[0]}x{block_size[1]})\n"
            f"Weight: {c} | Time: {gpu_time} s"
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig("blend_results.jpg")
    plt.close(figure)