from data_augmentation import flip_key_points, scale_key_points, move_key_points, rotate_key_points
import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


file_list = ['0.jpg']

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)
for idx, file in enumerate(file_list):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    # print('handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        continue
        
    annotated_image = image.copy()
    flipped_image = image.copy()
    scaled_image = image.copy()
    moved_image = image.copy()
    rotated_image = image.copy()
    
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite('annotated_image.png', cv2.flip(annotated_image, 1))
        point_list = []
        for i in hand_landmarks.landmark:
            point_list.append([i.x, i.y])
        
        # Flipped
        flipped_list = flip_key_points(point_list)
        for i, key_point in enumerate(hand_landmarks.landmark):
            key_point.x, key_point.y = flipped_list[i][0], flipped_list[i][1]
        mp_drawing.draw_landmarks(flipped_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite('test_flipped_image.png', cv2.flip(flipped_image, 1))
        
        # Scaled
        scaled_list = scale_key_points(point_list)
        for i, key_point in enumerate(hand_landmarks.landmark):
            key_point.x, key_point.y = scaled_list[i][0], scaled_list[i][1]
        mp_drawing.draw_landmarks(scaled_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite('test_scaled_image.png', cv2.flip(scaled_image, 1))
        
        # Moved
        moved_list = move_key_points(point_list)
        for i, key_point in enumerate(hand_landmarks.landmark):
            key_point.x, key_point.y = moved_list[i][0], moved_list[i][1]
        mp_drawing.draw_landmarks(moved_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite('test_moved_image.png', cv2.flip(moved_image, 1))
        
        # Rotated
        rotated_list = rotate_key_points(point_list, 3)
        for i, key_point in enumerate(hand_landmarks.landmark):
            key_point.x, key_point.y = rotated_list[i][0], rotated_list[i][1]
        mp_drawing.draw_landmarks(rotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite('test_rotated_image.png', cv2.flip(rotated_image, 1))
        
hands.close()
