import zmq
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stretch_ros2_bridge.ros.msg_numpy import image_to_numpy

class IphoneCamSaver(Node):
    def __init__(self, rgb_topic, depth_topic):
        super().__init__('iphone_cam_saver')

        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.timestamps = []
        self.frame_ids = []
        self.rgb_images = []
        self.depth_images = []

    def rgb_callback(self, msg):
        self.timestamps.append(msg.header.stamp)
        self.frame_ids.append(msg.header.frame_id)
        self.rgb_images.append(image_to_numpy(msg))

    def depth_callback(self, msg):
        self.depth_images.append(image_to_numpy(msg))

    def save_images(self):
        rgb_vid = np.stack(self.rgb_images, axis=0)
        np.save("/home/enes/rgb_vid.npy", rgb_vid)

        if len(self.depth_images) > 0:
            depth_vid = np.stack(self.depth_images, axis=0)
            np.save("/home/enes/depth_vid.npy", depth_vid)


def main():
    """Init and Spin the Node"""

    rclpy.init()
    iphone_cam_saver = IphoneCamSaver("ee_rgb", "ee_depth")
    try:
        rclpy.spin(iphone_cam_saver)
    except KeyboardInterrupt:
        iphone_cam_saver.get_logger().info("SAVING IMAGES...")
        iphone_cam_saver.save_images()
    finally:
        iphone_cam_saver.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()