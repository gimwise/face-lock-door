
'''
    학습
'''

from imutils import paths
import face_recognition
import pickle
import cv2
import os

# dataset에서 파일 받아오기
print("[안내] 파일에서 이미지를 읽어오는 중입니다.")
imagePaths = list(paths.list_images("dataset"))

# 초기화
knownEncodings = []
knownNames = []


for (i, imagePath) in enumerate(imagePaths):
	# 이미지 경로에서 사람 이름 추출하기
	print("[안내] 이미지 추출 {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# RGB로 변환
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# 입력 이미지의 각 얼굴에 해당하는 경계 좌표 감지
	boxes = face_recognition.face_locations(rgb, model="hog")

	# 계산
	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# 저장
print("[안내] 저장중 ")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
