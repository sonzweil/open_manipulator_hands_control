import cv2
import time
import math


from mediapipe.python.solutions.hands import Hands
from mediapipe.python.solutions.hands import HandLandmark

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_center(self):
        cx, cy = self.x + (self.w / 2), self.y + (self.h / 2)
        return (cx, cy)

    def get_box(self):
        return (self.x, self.y, self.w, self.h)

class Hand:
    def __init__(self, handedness="", landmarks=None, bounding_box=None):
        self.tip_indexes = [HandLandmark.THUMB_TIP,
                            HandLandmark.INDEX_FINGER_TIP,
                            HandLandmark.MIDDLE_FINGER_TIP,
                            HandLandmark.PINKY_TIP,
                            HandLandmark.RING_FINGER_TIP]
        self.mid_indexes = [HandLandmark.THUMB_IP,
                            HandLandmark.INDEX_FINGER_PIP,
                            HandLandmark.MIDDLE_FINGER_PIP,
                            HandLandmark.PINKY_PIP,
                            HandLandmark.RING_FINGER_PIP]

        self.handedness = handedness
        self.landmarks = landmarks
        self.bounding_box = bounding_box

        self.finger_up_list = []
        self.unique_landmark = None
        self.update_fingers()


    def update_fingers(self):
        if self.landmarks == None:
            self.finger_up_list.clear()
            return

        self.finger_up_list = []
        for i in range(0, len(self.tip_indexes)):
            _, tip_x, tip_y = self.landmarks[self.tip_indexes[i]]
            _, mid_x, mid_y = self.landmarks[self.mid_indexes[i]]

            # thumb
            if i == 0 and self.handedness == "Left":
                if tip_x > mid_x:
                    self.finger_up_list.append(True)
                else:
                    self.finger_up_list.append(False)
            elif i == 0 and self.handedness == "Right":
                if tip_x < mid_x:
                    self.finger_up_list.append(True)
                else:
                    self.finger_up_list.append(False)
            else:
                if tip_y < mid_y:
                    self.finger_up_list.append(True)
                else:
                    self.finger_up_list.append(False)

        self.num_of_fingers_up = self.finger_up_list.count(True)
        if self.num_of_fingers_up == 1:
            for i in range(0, len(self.tip_indexes)):
                if self.finger_up_list[i] == 1:
                    self.unique_landmark = self.landmarks[self.tip_indexes[i]]

    def update(self, handedness="", landmarks=None, bounding_box=None):
        self.handedness = handedness
        self.landmarks = landmarks
        self.bounding_box = bounding_box
        self.update_fingers()

    def draw_landmarks(self, image):
        if image is not None:
            for index, finger_up in zip(self.tip_indexes, self.finger_up_list):
                _, x, y = self.landmarks[index]
                if finger_up == False:
                    cv2.circle(image, (x, y), 6, (229, 238, 242), cv2.FILLED)
                elif index == HandLandmark.INDEX_FINGER_TIP:
                    cv2.circle(image, (x, y), 7, (174, 188, 241), cv2.FILLED)
                else:
                    cv2.circle(image, (x, y), 6, (221, 226, 195), cv2.FILLED)

        return image

    def draw_bounding_box(self, image):
        x1, y1, w, h = self.bounding_box.get_box()
        x2, y2 = x1 + w, y1 + h
        color = (0, 255, 255)
        thickness = 2

        cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), color, thickness)
        cv2.putText(image, f'{self.handedness}:{self.num_of_fingers_up}',
                    (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN,
                    2, (218, 206, 110), 2)

    def get_handedness(self):
        return self.handedness
    def get_landmarks(self):
        return self.landmarks
    def get_bounding_box(self):
        return self.bounding_box
    def get_finger_up_list(self):
        return self.finger_up_list
    def get_num_of_fingers_up(self):
        return self.num_of_fingers_up
    def get_selected_landmark(self, selected_landmark):
        return self.landmarks[selected_landmark]
    def get_hand_distance(self):
        _, x1, y1 = self.get_selected_landmark(HandLandmark.THUMB_TIP)
        _, x2, y2 = self.get_selected_landmark(HandLandmark.PINKY_TIP)
        length = math.hypot(x2 - x1, y2 - y1)
        actual_width = 8.5
        focal_length = 650
        distance = actual_width*focal_length/length
        return distance


class HandProcessing(Hands):
    def __init__(self, mode=False, max_hands=2, model_complex=1, detection_conf=0.85, track_conf=0.6):
        super().__init__(mode, max_hands, model_complex, detection_conf, track_conf)
        self.prev_num_of_hands = 0
        self.all_hands = []

    def process_hands(self, image):
        image_h, image_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = self.process(image_rgb)
        multi_hand_landmarks = processed_image.multi_hand_landmarks
        multi_handedness = processed_image.multi_handedness

        if not multi_hand_landmarks:
            print("No Hands")
            self.all_hands.clear()
            self.prev_num_of_hands = 0
            return self.all_hands

        cur_num_of_hands = len(multi_hand_landmarks)
        cur_hands = []
        for cur_handedness, single_hand_landmarks in zip(multi_handedness, multi_hand_landmarks):
            landmarks = []
            x_list = []
            y_list = []

            for index, landmark in enumerate(single_hand_landmarks.landmark):
                x, y = int(landmark.x * image_w), int(landmark.y * image_h)
                landmarks.append([index, x, y])
                x_list.append(x)
                y_list.append(y)
            min_x, max_x = min(x_list), max(x_list)
            min_y, max_y = min(y_list), max(y_list)
            box_w, box_h = max_x - min_x, max_y - min_y
            bounding_box = BoundingBox(min_x, min_y, box_w, box_h)
            handedness = cur_handedness.classification[0].label

            cur_hands.append([handedness, landmarks, bounding_box])


        if self.prev_num_of_hands == cur_num_of_hands:
            for index, hand in enumerate(cur_hands):
                handedness, landmarks, bounding_box = hand
                self.all_hands[index].update(handedness, landmarks, bounding_box)
        else:
            self.all_hands.clear()
            for index, hand in enumerate(cur_hands):
                handedness, landmarks, bounding_box = hand
                temp = Hand(handedness, landmarks, bounding_box)
                self.all_hands.append(temp)

        self.prev_num_of_hands = cur_num_of_hands
        return self.all_hands





def main():
    prev_time = 0
    cur_time = 0

    hand_processing = HandProcessing()
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        my_hands = hand_processing.process_hands(image)

        for hand in my_hands:
            hand.draw_landmarks(image)
            hand.draw_bounding_box(image)
            #print(hand.get_selected_landmark(HandLandmark.INDEX_FINGER_TIP))

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (145,148,171), 2)
        cv2.imshow("HandSolution", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

