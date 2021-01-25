import cv2
import mediapipe as mp
import time
from random import randint
import math

from data_augmentation import flip_key_points, scale_key_points, move_key_points, rotate_key_points

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)
face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')

# number of keypoints per hand
kp_num = 21

words = ['name', 'i', 'hello', 'learn', 'yes', 'no', 'brother', 'you']
#words = ['you']
train_to_test_ratio = 9

test_file = 'csv_data/test_data_words.csv'
train_file = 'csv_data/train_data_words.csv'
header = 'word,' + ','.join([str(i) for i in range(1, 4 * kp_num + 3)]) + '\n'
with open(test_file, 'a') as f:
    f.write(header)
with open(train_file, 'a') as f:
    f.write(header)

x = 0
for a in words:
    file_list = list(map(lambda x: 'data_words/' + a + '/' + str(x) + '.jpg', range(14)))
    for idx, file in enumerate(file_list):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # detect face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Print handedness and draw hand landmarks on the image.
        #print('handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        
        annotated_image = image.copy()
        rows = []
        row = a
        point_list = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imwrite("marked_data_words/" + a + str(idx) + '.jpg', cv2.flip(annotated_image, 1))
            for i in hand_landmarks.landmark:
                point_list.append([i.x, i.y])
                row += ',' + str(i.x)+ ',' + str(i.y)
        
        if len(point_list) < kp_num * 2:
            row += ',0,0' * (kp_num * 2 - len(point_list))
            point_list += [[0, 0]] * (kp_num * 2 - len(point_list))
        
        if len(faces) > 0:
            center = [faces[0, 0] + faces[0, 2] // 2, faces[0, 1] + faces[0, 3] // 2]
            point_list += [[center[0]/1280, center[1]/720]]
            row += ',' + str(center[0]/1280) + ',' + str(center[1]/720)
            annotated_image = cv2.circle(annotated_image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
        else:
            point_list += [[image.shape[0]//2, image.shape[1]//2]]
            row += ',' + str(0.5) + ',' + str(0.5)
        cv2.imwrite("marked_data_words/" + a + str(idx) + '.jpg', cv2.flip(annotated_image, 1))
        row += '\n'
        
        rows.append(row)
        point_lists = [point_list]
        
        # flip points
        new_list = flip_key_points(point_list)
        point_lists.append(new_list)
        row = a
        for point in new_list:
            row += ',' + str(point[0]) + ',' + str(point[1])
        row += '\n'
        rows.append(row)
        
        # move points
        for i in range(len(point_lists)):
            for j in range(4):
                new_list = move_key_points(point_lists[i])
                point_lists.append(new_list)
                row = a
                for point in new_list:
                    row += ',' + str(point[0]) + ',' + str(point[1])
                row += '\n'
                rows.append(row)
        
        # rotate key points
        for i in range(len(point_lists)):
            for j in range(4):
                new_list = rotate_key_points(point_lists[i], math.pi/4)
                point_lists.append(new_list)
                row = a
                for point in new_list:
                    row += ',' + str(point[0]) + ',' + str(point[1])
                row += '\n'
                rows.append(row)
        
        # scale points
        for i in range(len(point_lists)):
            for j in range(4):
                new_list = scale_key_points(point_lists[i])
                point_lists.append(new_list)
                row = a
                for point in new_list:
                    row += ',' + str(point[0]) + ',' + str(point[1])
                row += '\n'
                rows.append(row)
        
        # split data to test and train
        test_rows = []
        for i in range(len(rows)//(train_to_test_ratio + 1)):
            j = randint(0, len(rows) - 1)
            test_rows.append(rows.pop(j))
        
        with open(test_file, 'a') as f:
            f.write(''.join(test_rows))
        
        with open(train_file, 'a') as f:
            f.write(''.join(rows))


hands.close()