import cv2
import face_recognition
import os
from safetensors import safe_open
from safetensors.numpy import save_file

print(cv2.__version__)

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

face_code = None
running = True
while running:
    ret, img = capture.read()
    spase = cv2.waitKey(30) & 0xFF

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        coordinate = y - 10
        cv2.rectangle(img, (x, coordinate), (x + 50, coordinate + 10), (255, 0, 0), 2)
        if spase == 32:
            face_img = img[y:y + h, x:x + w]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)  # Обязательно переводим в rgb
            face_encoded = face_recognition.face_encodings(face_img_rgb)
            if len(face_encoded) > 0:
                face_code = face_encoded[0]
                # Это numpy массив в котором лежит 128 числа типа float64
                print(face_code.shape, type(face_code), face_code.dtype)
                running = False
                break
            if len(face_encoded) < 0:
                print("No face, try again.")

    cv2.imshow("from camera", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    if cv2.getWindowProperty('from camera', cv2.WND_PROP_VISIBLE) < 1:
        break

face_id_path = "face_id.safetensors"
if os.path.isfile(face_id_path):
    face_id = {}
    with safe_open(face_id_path, framework="np", device="cpu") as f:
        for key in f.keys():
            face_id[key] = f.get_tensor(key)
else:
    face_id = {}

# ToDo
print(face_id.keys(), face_id.values())

if face_code is not None:
    print("input")
    name = input()
    print(name)
    face_id[name] = face_code
    save_file(face_id, face_id_path)

capture.release()
cv2.destroyWindow('from camera')
