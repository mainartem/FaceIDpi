import os
import face_recognition
import cv2
from safetensors import safe_open
print(cv2.__version__)

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

face_id_path = "face_id.safetensors"
if os.path.isfile(face_id_path):
    face_id = {}
    with safe_open(face_id_path, framework="np", device="cpu") as f:
        for key in f.keys():
            face_id[key] = f.get_tensor(key)
else:
    print('No file', face_id_path)
    raise Exception

while True:
    ret, img = capture.read()
    # copyImage = img.copy()

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # coordinate = y - 10
        # cv2.rectangle(img, (x, coordinate), (x + 50, coordinate + 10), (255, 0, 0), 2)

        face_img = img[y:y+h, x:x+w]  # Вырезаем лицо
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)  # Обязательно переводим в rgb
        face_encoded = face_recognition.face_encodings(face_img_rgb)  # Вычисляем код
        if len(face_encoded) > 0:
            face_code = face_encoded[0]

            results = face_recognition.compare_faces(list(face_id.values()), face_code)
            names = [name for name, result in zip(face_id.keys(), results) if result]
            print(names)

            # ToDo
            # Теперь надо искать в базе данных похожий код
            # Вроде можно все коды вытащить из бд в список и сразу все прогнать
            #
            # face_id_list = my_database.get_all_face_codes()
            # result = face_recognition.compare_faces(face_id_list, face_encoded)
            #
            # result это список с вероятностями совпадения
            # как решать дальше это сложно на самом деле
            # пока можно просто искать лучший результат и принимать его если он больше 90%

            # cv2.imwrite("face.jpg", face_img)

            # cv2.rectangle(copyImage, (x, y), (x+w, y+h), (0, 255, 255), 5)
            # if len(names) > 0:
            #     name = str(names[0])
            # else:
            #     name = "Unknown"
            # font = cv2.FONT_HERSHEY_COMPLEX
            # fontScale = 1
            # thickness = 1
            # textSize = cv2.getTextSize(name, font, fontScale, thickness)[0]
            # text_width = textSize[0]
            # text_height = textSize[1]

            # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            # draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

            # cv2.rectangle(copyImage, (x, y+h - text_height - 10), (x+w, y+h), (0, 255, 255), 5)
            # cv2.putText(copyImage, name, (x, y+h), font, 1, (255, 255, 255), 1)

    # Fase
    # face_locations = face_recognition.face_locations(copyImage)

    """
    for (top, right, bottom, left), face_encoding in zip(face_locations, list(face_id.values())):

        matches = face_recognition.compare_faces(list(face_id.values()), face_encoding)
        name = "Unknown"

        if matches[0] == True:
            name = face_id.keys()
        # print(matches[0])

        cv2.rectangle(copyImage, (left, top), (right, bottom), (0, 255, 255), 15)
        name = str(name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        textSize = cv2.getTextSize(name, font, fontScale, thickness)[0]
        text_width = textSize[0]
        text_height = textSize[1]

        # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        # draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        cv2.rectangle(copyImage, (left, bottom - text_height - 10), (right, bottom), (0, 255, 255), 15)
        cv2.putText(copyImage, name, (left, bottom) , font, 1, (255,255,255), 2)
    """

    # cv2.imshow("face", copyImage)
    # cv2.imshow("from camera", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    #if cv2.getWindowProperty('from camera', cv2.WND_PROP_VISIBLE) < 1:
    #    break

capture.release()
# cv2.destroyWindow()
