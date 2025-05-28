import cv2 as cv 
import mediapipe as mp 
import time

class PoseDetector:
    def __init__(self, mode=False, complexity=1,
                 smooth_landmark=True, segmentation=False,
                 smooth_segmentation=True, min_detect=0.5, min_track=0.5):
        self.mode = mode 
        self.complexity = complexity
        self.smooth_landmark = smooth_landmark
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detect = min_detect
        self.min_track = min_track

        self.mppose = mp.solutions.pose 
        self.pose = self.mppose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmark,
            enable_segmentation=self.segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detect,
            min_tracking_confidence=self.min_track
        )
        self.drawPose = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.drawPose.draw_landmarks(frame, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)
        return frame

    def findPosePosition(self, frame, draw=True):
        lmlist = [] 
        if self.results.pose_landmarks: 
            h, w, _ = frame.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark): 
                cx, cy = int(lm.x * w), int(lm.y * h)
                if draw: 
                    cv.circle(frame, (cx, cy), 5, (0, 0, 0), cv.FILLED)
                lmlist.append([id, cx, cy])
        return lmlist

def main():
    vid = cv.VideoCapture("videos/videoplayback.mp4")
    detector = PoseDetector()
    ptime = 0

    while True:
        isTrue, frame = vid.read()
        if not isTrue:
            break
        
        frame = detector.findPose(frame)
        lmlist = detector.findPosePosition(frame , draw= False)
        if lmlist:
             cv.circle(frame , (lmlist[14][1] ,lmlist[14][2] ) , 10 , (0,0,0) , -1 )

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv.putText(frame, f'FPS: {int(fps)}', (10, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv.imshow("Pose Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
