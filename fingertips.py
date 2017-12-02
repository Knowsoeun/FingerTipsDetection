import cv2
import numpy as np

def ConvexHull():
    img = cv2.imread('two-hands.jpg', 0)
    ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(thresh1, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    for cnt in contours:
        if cv2.contourArea(cnt) < 100: #area less than threshold is considered noise
            continue

        hull = cv2.convexHull(cnt, returnPoints = False)    #find convex hull for each contours

        defects = cv2.convexityDefects(cnt, hull)    #find defects in the hull
        if defects is None:
            continue
        moments = cv2.moments(cnt)
        centre = ((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))) #from moments find centre of the contours
        cv2.circle(img, centre, 3, (0, 0, 0), -1)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            if start[1] > centre[1]:
                continue
            cv2.line(img,centre, start,[255,100,0],2)
            cv2.circle(img,start,5,[100, 255, 204],-1)

    cv2.imshow('img',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ConvexHull()