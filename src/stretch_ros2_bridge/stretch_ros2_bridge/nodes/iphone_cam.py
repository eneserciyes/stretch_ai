import time
import cv2
import numpy as np
from record3d import Record3DStream
from threading import Event

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from stretch_ros2_bridge.ros.msg_numpy import numpy_to_image


class R3DApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.stream_stopped = True

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        self.stream_stopped = True
        print("Stream stopped")

    def connect_to_device(self, dev_idx):
        print("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        print("{} device(s) found".format(len(devs)))
        for dev in devs:
            print("\tID: {}\n\tUDID: {}\n".format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError(
                "Cannot connect to device #{}, try different index.".format(dev_idx)
            )

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing
        self.stream_stopped = False

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def start_process_image(self):
        self.event.wait(5)
        rgb = self.session.get_rgb_frame()
        depth = self.session.get_depth_frame()
        camera_pose = self.session.get_camera_pose()
        pose = np.array(
            [
                camera_pose.qx,
                camera_pose.qy,
                camera_pose.qz,
                camera_pose.qw,
                camera_pose.tx,
                camera_pose.ty,
                camera_pose.tz,
            ]
        )
        return rgb, depth, pose

# TODO: make this a ROS Node
class R3DCameraPublisher(Node):
    def __init__(self, ee_rgb_topic, ee_depth_topic, use_depth):
        super().__init__("r3d_camera_publisher")

        self.ee_rgb_topic = ee_rgb_topic
        self.ee_depth_topic = ee_depth_topic
        self.use_depth = use_depth

        self._rgb_pub = self.create_publisher(Image, ee_rgb_topic, 10)
        if use_depth:
            self._depth_pub = self.create_publisher(Image, ee_depth_topic, 10)

        self._seq = 0

        # start the Record3D streaming
        self._start_camera()

        # start publishing the RGB and depth images
        timer_freq = 50
        self.timer = self.create_timer(1 / timer_freq, self.timer_callback)

    # start the Record3D streaming
    def _start_camera(self):
        self.app = R3DApp()
        dev_idx = 0
        while self.app.stream_stopped:
            try:
                self.app.connect_to_device(dev_idx=dev_idx)
            except RuntimeError as e:
                self.get_logger().error(str(e))
                self.get_logger().error(
                    f"Retrying to connect to device with id {dev_idx}, make sure the device is connected and id is correct..."                )
                time.sleep(2)

    # get the RGB and depth images from the Record3D
    def get_rgb_depth_images(self):
        image = None
        while image is None:
            image, depth, pose = self.app.start_process_image()
            image = np.moveaxis(image, [0], [1])[..., ::-1, ::-1]
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        if self.use_depth:
            depth = np.ascontiguousarray(np.rot90(depth, -1)).astype(np.float64)
            return image, depth, pose
        else:
            return image, pose

    # get RGB images at 50Hz and publish them to the ZMQ port
    def timer_callback(self):
        if self.app.stream_stopped:
            try:
                self.app.connect_to_device(dev_idx=0)
            except RuntimeError as e:
                print(e)
                print(
                    "Retrying to connect to device with id {idx}, make sure the device is connected and id is correct...".format(
                        idx=0
                    )
                )
                time.sleep(2)
        else:
            if self.use_depth:
                image, depth, pose = self.get_rgb_depth_images()
                rgb_msg = numpy_to_image(image, "rgb8")
                rgb_msg.header.stamp = self.get_clock().now().to_msg()
                rgb_msg.header.frame_id = str(self._seq)
                depth_msg = numpy_to_image(depth, "64FC1")
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = str(self._seq)
                self._rgb_pub.publish(rgb_msg)
                self._depth_pub.publish(depth_msg)
            else:
                image, pose = self.get_rgb_depth_images()
                rgb_msg = numpy_to_image(image, "rgb8")
                rgb_msg.header.stamp = self.get_clock().now().to_msg()
                rgb_msg.header.frame_id = str(self._seq)
                self._rgb_pub.publish(rgb_msg)
            self._seq += 1


def main():
    """Init and Spin the Node"""

    rclpy.init()
    r3d_camera_publisher = R3DCameraPublisher("ee_rgb", "ee_depth", use_depth=False)
    rclpy.spin(r3d_camera_publisher)

    r3d_camera_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()