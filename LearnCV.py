import cv2
cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    diff = cv2.absdiff(prev_gray, gray)

    ret, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for i in contours:
        if cv2.contourArea(i) > 800:
                    x, y, w, h = cv2.boundingRect(i)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    count += 1
    print(count)

    cv2.imshow("rect", frame)
    cv2.imshow("Motion", thresh)

    prev_gray = gray

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()