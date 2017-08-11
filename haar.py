import cv2

cascade = cv2.CascadeClassifier('cascade.xml')


def get_roi(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    roi_arr = cascade.detectMultiScale(gray, scaleFactor=1.05)

    # test purpose
    #
    # for (x, y, w, h) in signs:
    #    cv2.rectangle(dataset, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #    roi_color = dataset[y:y + h, x:x + w]
    # cv2.imshow('dataset', dataset)
    # cv2.destroyAllWindows()
    return roi_arr
