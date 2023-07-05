import cv2
import argparse
import numpy as np
from skimage.metrics import structural_similarity
from face_compare.images import get_face
from face_compare.model import facenet_model, img_to_encoding

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

parser = argparse.ArgumentParser()
parser.add_argument("--img", "-i", default=None, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_args()

vc = cv2.VideoCapture(args.img)
#'C:/Users/HP/Desktop/Dersler/Yapay Zekaya Giriş/face_detection/face.mp4'
frame_width = int(vc.get(3))
frame_height = int(vc.get(4))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height),0)

old_frame = None
counter = 0
old_counter = 0
old_face = None

while True:
    ret, frame = vc.read()

    if ret == False:
        break

    frame = cv2.resize(frame, (900, 500), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade_face.detectMultiScale(grayscale, 1.1, 10)
    facee = get_face(frame)
    model = facenet_model(input_shape=(3, 96, 96))

    if old_face is not None:
        embedding_one = img_to_encoding(old_face, model)
        embedding_two = img_to_encoding(facee, model)

        dist = np.linalg.norm(embedding_one - embedding_two)
        print(f'Distance between two images is {dist}')
        if dist > 0.7:
            print('These images are of two different people!')
        else:
            print('These images are of the same person!')

        for (x_face, y_face, w_face, h_face) in face:
            cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 130, 0), 2)
            ri_grayscale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
            ri_color = frame[y_face:y_face + h_face, x_face:x_face + w_face]

            eye = cascade_eye.detectMultiScale(ri_grayscale, 1.1, 10)
            for (x_eye, y_eye, w_eye, h_eye) in eye:
                cv2.rectangle(ri_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)

            smile = cascade_smile.detectMultiScale(ri_grayscale, 3, 30)
            if old_frame is not None:
                first_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                second_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score, diff = structural_similarity(first_gray, second_gray, full=True)
                score = float(format(score * 100))

                for (x_smile, y_smile, w_smile, h_smile) in smile:
                    cv2.rectangle(ri_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)

                    if (old_counter == counter):
                        counter += 1
                        print("Smile: ", counter)

                if(dist > 0.7):
                    old_counter = counter


    out.write(frame)
    cv2.imshow('Video', frame)
    old_frame = frame
    old_face = facee

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()