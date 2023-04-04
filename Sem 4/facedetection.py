import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml")

img = cv2.imread(r"D:\Ayush\DYPU\Mini Project\Sem 4\DSC_7916.jpg", 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)
resized = cv2.resize(img, (531,413))


cv2.imshow("Image", resized)
cv2.waitKey(0)

cv2.destroyAllWindows()