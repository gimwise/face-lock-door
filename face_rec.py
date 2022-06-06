
'''
    학습 확인용
'''

from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2

# 사용자 이름 초기화 -> unknown
currentname = "unknown"
# train_model.py를 통해 학습된 얼굴로 결정
encodingsP = "encodings.pickle"
#use this xml file
# 미리 구현된 haar cascade 알고리즘 사용
cascade = "haarcascade_frontalface_default.xml"

# xml 파일을 분류기로 설정
print("[안내] 곧 감지가 시작됩니다.")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# 비디오 스트리밍 시작
print("[안내] 스트리밍 시작")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 비디오 파일 스트림에서 프레임을 반복
while True:
    # 사진 리사이징 -> 처리 속도 향상
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
    # 회색조, RGB로 변환 (opencv는 BGR)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# 회색조 프레임에서 얼굴 감지
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    # x,y,w,h 순으로 경계 좌표를 변환하고 재정렬
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	# 경계에 대한 얼굴 인식하기
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# 얼굴 인식 반복문
	for encoding in encodings:
		# 입력된 얼굴과 학습한 얼굴이 맞는지 확인
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" # 얼굴이 인식되지 않으면 name는 Unknown

		# 일치하는 항목이 있는지 확인
		if True in matches:
			# 일치하는 얼굴의 인덱스를 찾아 맞은 횟수 계산
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# 일치하는 인덱스를 반복해 인식한 얼굴 개수 유지
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# 가장 많이 확인된 얼굴을 결정
			name = max(counts, key=counts.get)
			
            # Dataset에서 식별면 이름을 콘솔에 프린트
			if currentname != name:
				currentname = name
				print("현재 : " + currentname)
		
		# 이름 목록 업데이트
		names.append(name)

	# 인식된 얼굴 정보를 화면아 나타내기 위한 반복문
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# 예측한 얼굴의 이름을 적음
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (255, 0, 0), 2)

	# 화면에 정보 표시
	cv2.imshow("실행중", frame)
	key = cv2.waitKey(1) & 0xFF

	# q 누르면 종료
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()