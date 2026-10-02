from matplotlib.image import imread
from matplotlib import pyplot as plt
import numpy as np
import time

def preprocessing(rgb):
    image = imread(rgb)
    image_for_cuda = np.array(image, dtype=np.uint8)
    height, width, channels = image_for_cuda.shape
    pixels = image_for_cuda.reshape(-1, channels)
    return pixels, image_for_cuda

def gaussian_blur_cpu(image):
    image = image.copy()
    height, width, channels = image.shape
    result = np.zeros_like(image)
    kernel = np.array([
        [0, 0, 1, 2, 1, 0, 0],
        [0, 3, 13, 22, 13, 3, 0],
        [1, 13, 59, 97, 59, 13, 1],
        [2, 22, 97, 159, 97, 22, 2],
        [1, 13, 59, 97, 59, 13, 1],
        [0, 3, 13, 22, 13, 3, 0],
        [0, 0, 1, 2, 1, 0, 0]
    ], dtype=np.float32)
    kernel = kernel / 1003.0
    padded_image = np.pad(image, ((3, 3), (3, 3), (0, 0)), mode='edge')
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                region = padded_image[i:i+7, j:j+7, c]
                result[i, j, c] = np.uint8(np.sum(region * kernel)) 
    return result

def gaussian_blur_cpu_time(rgb):
    pixels, original_image = preprocessing(rgb)
    start_time = time.perf_counter()
    final_3d_image = gaussian_blur_cpu(original_image)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"CPU time: {runtime} seconds")
    return final_3d_image, runtime

if __name__ == "__main__":
    rgb = "images.jpg"
    cpu_result_image, cpu_time = gaussian_blur_cpu_time(rgb)
    plt.figure()
    plt.imshow(cpu_result_image)
    plt.title(f"CPU time: {cpu_time:} seconds")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("result_cpu.jpg")
    plt.show()
