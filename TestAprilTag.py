import cv2
import apriltag

frameWidth = 720
frameHeight = 720

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

print(" [INFO] Detection of AprilTags...")

options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

while True:
    success, image = cap.read()
    if not success:
        break

    l_channel, a, b = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(10, 10))
    cl = clahe.apply(l_channel)

    seper = cv2.merge((cl, a, b))

    gray = cv2.cvtColor(seper, cv2.COLOR_BGRA2GRAY)

    results = detector.detect(gray)

    for r in results:
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        cv2.line(image, ptA, ptB, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        # Draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
