import os
import haar
import cv2
import cnn_recognition

DIR_NAME = "dataset"

# f = open('results.txt', 'w')
for (root, dirs, filenames) in os.walk(os.path.abspath(DIR_NAME)):
    for file in filenames:
        roi = haar.get_roi(os.getcwd() + '\\' + DIR_NAME + '\\' + file)
        img_current = cv2.imread(os.getcwd() + '\\' + DIR_NAME + '\\' + file)
        f.write(file + ';')
        for (x, y, w, h) in roi:
            roi_current = img_current[y:y + h, x:x + w]
            # f.write(str(x) + ';' + str(y) + ';' + str(x + w) + ';' + str(y + h) + ';')
            class_no = cnn_recognition.get_class_no(roi_current)
            # f.write(str(class_no) + ';')
        # f.write('\n')