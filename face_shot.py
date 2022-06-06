
'''
    학습용 사진 촬영
'''

import cv2

name = 'tom' # 촬영할 사람 이름 작성

cam = cv2.VideoCapture(0)

cv2.resizeWindow("PHOTO SHOT", 500, 300)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("[촬영] 읽어오기 실패")
        break
    cv2.imshow("[촬영] 스페이스바를 눌러 사진을 촬용하세요.", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC
        print("[안내] 곧 화면이 종료됩니다.")
        break
    elif k%256 == 32:
        # 스페이스바
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()