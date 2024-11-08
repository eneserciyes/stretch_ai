# %%
import numpy as np
import cv2


# %%
def visualize_rgb_vid():
    vid = np.load("/home/enes/rgb_vid.npy")

    writer = cv2.VideoWriter("/home/enes/rgb_vid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 50, (256, 256))
    for i in range(vid.shape[0]):
        writer.write(vid[i])

    writer.release()

visualize_rgb_vid()

# %%

def visualize_depth_vid():
    vid = np.load("/home/enes/depth_vid.npy")
    print(vid.shape)

    global_min = 0
    global_max = 2

    vid[vid < global_min] = 0
    vid[vid > global_max] = 0

    writer = cv2.VideoWriter("/home/enes/depth_vid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 50, (vid.shape[2], vid.shape[1]))
    for i in range(vid.shape[0]):
        depth_image = (vid[i] - global_min) / (global_max - global_min)
        depth_image = (depth_image * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        writer.write(depth_colormap)

    writer.release()

visualize_depth_vid()