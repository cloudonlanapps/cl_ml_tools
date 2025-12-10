from facenet_pytorch import MTCNN as MTCNN
import cv2

detector0 = MTCNN(
    image_size=640, margin=0, min_face_size=20, keep_all=True, select_largest=True
)
detector1 = MTCNN(
    image_size=240,
    margin=0,
    min_face_size=20,
    keep_all=True,
    select_largest=True,
    post_process=False,
)


def detect_faces(img, detector=detector0, prob_low=0):
    boxes, probs = detector.detect(img)
    if boxes is None:
        return []
    return [
        {"pos": bounded_box(box, img), "prob": prob}
        for (box, prob) in zip(boxes, probs)
        if prob > prob_low
    ]


def bounded_box(box, img):
    x0, y0, x1, y1 = box
    height, width, _ = img.shape
    x0 = int(min(width, max(0, x0)))
    x1 = int(min(width, max(0, x1)))
    y0 = int(min(height, max(0, y0)))
    y1 = int(min(height, max(0, y1)))
    return [x0, y0, x1, y1]


if __name__ == "__main__":
    image_file = "/disks/data/image_repo/image_3498/AHIHI_COLLAGE1564901853654.png"
    # image_file = '/disks/data/image_repo/image_4/IMG_4588.PNG'
    img = cv2.imread(image_file)
    faces = detect_faces(img, detector0)
    for face in faces:
        box = [int(b) for b in face]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Faces Detected", img)  # cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
