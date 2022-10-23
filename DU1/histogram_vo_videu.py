import cv2
import cv2 as cv
import matplotlib.pyplot as plt


def add_histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.savefig("./images/hist.jpg")
    plt.clf()
    hist = cv.imread(cv.samples.findFile("./images/hist.jpg"))

    scale = 30
    width = int(hist.shape[1] * scale / 100)
    height = int(hist.shape[0] * scale / 100)
    dim = (width, height)
    resized_hist = cv.resize(hist, dim, interpolation=cv2.INTER_AREA)

    img[0:resized_hist.shape[0], 0:resized_hist.shape[1]] = resized_hist
    return img


cap = cv2.VideoCapture('./images/video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./images/output.mp4', fourcc, 25.0, (320, 576))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        new_frame = add_histogram(frame)
        cv2.imshow('frame', new_frame)
        out.write(new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
s
cap.release()
out.release()
cv2.destroyAllWindows()
