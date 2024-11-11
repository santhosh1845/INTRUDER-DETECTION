import cv2
import face_recognition
import os


known_faces = []
known_face_labels = []
          
# Replace this with your dataset
# Each person's images should be in a separate folder with their label as the folder name
# For simplicity, use labeled directories like 'person_1', 'person_2', etc.
dataset_path = 'D:\Python-Backup\SureshProject\intruder detection\known_faces'

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        face_img = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(face_img)[0]
        known_faces.append(face_encoding)
        known_face_labels.append(label)


video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = video_capture.read()


    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_labels[first_match_index]


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


    cv2.imshow('Intruder Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
