import cv2
import mediapipe as mp
import time
from random import randint
import math

from data_augmentation import flip_key_points, scale_key_points, move_key_points, rotate_key_points

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

file_list = ['B1019.jpg']
# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)

letters = 'abcdefghijklmnopqrstuvwxy'
train_to_test_ratio = 9

header = 'letter,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42\n'

with open('csv_data/test_data.csv', 'a') as f:
    f.write(header)

with open('csv_data/train_data.csv', 'a') as f:
    f.write(header)

for a in letters:
    file_list = list(map(lambda x: 'data/' + a + '/' + str(x) + '.jpg', range(12)))
    for idx, file in enumerate(file_list):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        #print('handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            rows = []
            row = a
            point_list = []
            for i in hand_landmarks.landmark:
                point_list.append([i.x, i.y])
                row += ',' + str(i.x)+ ',' + str(i.y)
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
            with open('csv_data/test_data.csv', 'a') as f:
                f.write(''.join(test_rows))
            
            with open('csv_data/train_data.csv', 'a') as f:
                f.write(''.join(rows))
            
        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite("marked_data/" + a + str(idx) + '.jpg', cv2.flip(annotated_image, 1))


hands.close()