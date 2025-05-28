import cv2 as cv
import mediapipe as mp
import time

class handDetection:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.mpdraw = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findHands(self, frame, draw=True):
        rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgbImage)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS)
        return frame

    def findposition(self, frame, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, _ = frame.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
                    if draw:
                        cv.circle(frame, (cx, cy), 5, (0, 0, 0), cv.FILLED)
        return lmlist


def main():
    ptime = 0
    vid = cv.VideoCapture(0)
    detector = handDetection()
    
    while True:
        isTrue, frame = vid.read()
        if not isTrue:
            break

        frame = detector.findHands(frame)
        lmlist = detector.findposition(frame , draw= False)
        if(lmlist) :
            cv.circle(frame , (lmlist[4][1],lmlist[4][2]) , 15, (0,0,0) , -1)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv.putText(frame, f'FPS: {int(fps)}', (10, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv.imshow("Hand Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
