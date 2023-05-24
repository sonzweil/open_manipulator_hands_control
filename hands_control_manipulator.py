import rclpy
import cv2

from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState, Image
from hand_solution import HandProcessing
from open_manipulator_msgs.msg import KinematicsPose, JointPosition
from open_manipulator_msgs.srv import SetJointPosition, SetKinematicsPose
from mediapipe.python.solutions.hands import HandLandmark

bridge = CvBridge()

present_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

x_delta = 0.002
y_z_delta = 0.0005  # meter
joint_angle_delta = 0.05  # radian
path_time = 1.0  # second

REQUEST_FRAME_UNIT = 30

class HandsControlManipulator(Node):
    def __init__(self):
        super().__init__('hands_publisher')
        qos = QoSProfile(depth=10)

        self.timer_period = 0.01
        self.cap = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(2)
        self.hand_processing = HandProcessing()

        self.timer = self.create_timer(self.timer_period, self.time_callback)
        self.image_publisher = self.create_publisher(Image, 'image', qos)
        self.image_publisher2 = self.create_publisher(Image, 'image2', qos)

        self.goal_task_space = self.create_client(SetKinematicsPose, 'goal_task_space_path')
        self.goal_task_space_request = SetKinematicsPose.Request()
        self.goal_task_space_request.end_effector_name = 'gripper'
        self.goal_task_space_request.path_time = path_time
        self.goal_task_space_request.kinematics_pose = KinematicsPose()

        self.tool_control = self.create_client(SetJointPosition, 'goal_tool_control')
        self.tool_control_request = SetJointPosition.Request()
        self.tool_control_request.path_time = path_time
        self.tool_control_request.joint_position = JointPosition()
        self.tool_control_request.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']

        self.reference_landmark_pos = [HandLandmark.INDEX_FINGER_TIP, 320, 240]
        self.current_landmark_pos = [HandLandmark.INDEX_FINGER_TIP, 320, 240]
        self.reference_distance = 30
        self.current_distance = 30

        self.kinematics_pose_subscription = self.create_subscription(
            KinematicsPose,
            'kinematics_pose',
            self.kinematics_pose_callback,
            qos)
        self.kinematics_pose_subscription
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            qos)
        self.joint_state_subscription

        self.is_open = False
        self.frame_counter = 0

    def kinematics_pose_callback(self, msg):
        present_kinematics_pose[0] = msg.pose.position.x
        present_kinematics_pose[1] = msg.pose.position.y
        present_kinematics_pose[2] = msg.pose.position.z
        present_kinematics_pose[3] = msg.pose.orientation.w
        present_kinematics_pose[4] = msg.pose.orientation.x
        present_kinematics_pose[5] = msg.pose.orientation.y
        present_kinematics_pose[6] = msg.pose.orientation.z

    def joint_state_callback(self, msg):
        present_joint_angle[0] = msg.position[0]
        present_joint_angle[1] = msg.position[1]
        present_joint_angle[2] = msg.position[2]
        present_joint_angle[3] = msg.position[3]
        present_joint_angle[4] = msg.position[4]

    def time_callback(self):
        ret, image = self.cap.read()
        ret2, image2 = self.cap2.read()
        image = cv2.flip(image, 1)
        image_h, image_w, _ = image.shape
        my_hands = self.hand_processing.process_hands(image)

        for hand in my_hands:
            hand.draw_landmarks(image)
            hand.draw_bounding_box(image)

            if self.frame_counter >= REQUEST_FRAME_UNIT:
                if hand.get_handedness() == "Left": # left hand for dx
                    if hand.get_num_of_fingers_up() == 5: # move manipulator x
                        self.current_distance = hand.get_hand_distance()
                        dx = 1 * (self.current_distance - self.reference_distance) * x_delta
                        dy = 0
                        dz = 0
                        self.kinematics_request(dx, dy, dz)
                else: # right hand for gripper, dy, dz
                    if hand.get_num_of_fingers_up() == 5: # gripper open
                        self.tool_control_request.joint_position.position = [present_joint_angle[0], present_joint_angle[1], present_joint_angle[2], present_joint_angle[3], present_joint_angle[4] + 0.003]
                        self.gripper_request()
                    elif hand.get_num_of_fingers_up() == 0: # gripper close
                        self.tool_control_request.joint_position.position = [present_joint_angle[0], present_joint_angle[1], present_joint_angle[2], present_joint_angle[3], present_joint_angle[4] - 0.003]
                        self.gripper_request()
                    elif hand.get_num_of_fingers_up() == 1: # move manipulator
                        self.current_landmark_pos = hand.get_selected_landmark(HandLandmark.INDEX_FINGER_TIP)
                        print(self.current_landmark_pos)
                        dx = 0
                        dy = -1 * (self.current_landmark_pos[1] - self.reference_landmark_pos[1]) * y_z_delta
                        dz = -1 * (self.current_landmark_pos[2] - self.reference_landmark_pos[2]) * y_z_delta
                        self.reference_landmark_pos = self.current_landmark_pos

                        self.kinematics_request(dx, dy, dz)
                    elif hand.get_num_of_fingers_up() == 2: # set referece landmark
                        self.reference_landmark_pos = hand.get_selected_landmark(HandLandmark.INDEX_FINGER_TIP)
                self.frame_counter = 0

        if ret == True:
            image_encoded = bridge.cv2_to_imgmsg(image, encoding='bgr8')
            self.image_publisher.publish(image_encoded)
        if ret2 == True:
            image2_encoded = bridge.cv2_to_imgmsg(image2, encoding='bgr8')
            self.image_publisher2.publish(image2_encoded)

        self.frame_counter += 1
        #cv2.imshow('image', image)
        cv2.waitKey(1)

    def kinematics_request(self, dx, dy, dz):
        self.goal_task_space_request.kinematics_pose.pose.position.x = present_kinematics_pose[0] + dx
        self.goal_task_space_request.kinematics_pose.pose.position.y = present_kinematics_pose[1] + dy
        self.goal_task_space_request.kinematics_pose.pose.position.z = present_kinematics_pose[2] + dz
        self.goal_task_space_request.kinematics_pose.pose.orientation.w = present_kinematics_pose[3]
        self.goal_task_space_request.kinematics_pose.pose.orientation.x = present_kinematics_pose[4]
        self.goal_task_space_request.kinematics_pose.pose.orientation.y = present_kinematics_pose[5]
        self.goal_task_space_request.kinematics_pose.pose.orientation.z = present_kinematics_pose[6]

        try:
            self.goal_task_space.call_async(self.goal_task_space_request)
        except Exception as e:
            self.get_logger().info('Sending Goal Kinematic Pose failed %r' % (e,))

    def gripper_request(self):
        try:
            self.tool_control.call_async(self.tool_control_request)
        except Exception as e:
            self.get_logger().info('gripper open failed %r' % (e,))

    def kinematics_pose_callback(self, msg):
        present_kinematics_pose[0] = msg.pose.position.x
        present_kinematics_pose[1] = msg.pose.position.y
        present_kinematics_pose[2] = msg.pose.position.z
        present_kinematics_pose[3] = msg.pose.orientation.w
        present_kinematics_pose[4] = msg.pose.orientation.x
        present_kinematics_pose[5] = msg.pose.orientation.y
        present_kinematics_pose[6] = msg.pose.orientation.z

    def joint_state_callback(self, msg):
        present_joint_angle[0] = msg.position[0]
        present_joint_angle[1] = msg.position[1]
        present_joint_angle[2] = msg.position[2]
        present_joint_angle[3] = msg.position[3]
        present_joint_angle[4] = msg.position[4]

def print_present_values():
    print('Joint Angle(Rad): [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(
        present_joint_angle[0],
        present_joint_angle[1],
        present_joint_angle[2],
        present_joint_angle[3],
        present_joint_angle[4]))
    print('Kinematics Pose(Pose X, Y, Z | Orientation W, X, Y, Z): {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        present_kinematics_pose[0],
        present_kinematics_pose[1],
        present_kinematics_pose[2],
        present_kinematics_pose[3],
        present_kinematics_pose[4],
        present_kinematics_pose[5],
        present_kinematics_pose[6]))

def main(args=None):
    rclpy.init(args=args)
    node = HandsControlManipulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Publish Stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


