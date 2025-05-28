import cv2 as cv 
import mediapipe as mp 
import time as t

class faceMesh :
    def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) : 
        self.mode = static_image_mode
        self.maxFaces = max_num_faces
        self.refineLandmarks = refine_landmarks
        self.minDetection = min_detection_confidence
        self.minTracking = min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
            min_detection_confidence=self.minDetection,
            min_tracking_confidence=self.minTracking
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.landmarkCanvas = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=2)
        self.connectionCanvas = self.mpDraw.DrawingSpec(color=(255,0,0), thickness=1)

    def findFaceMesh(self, frame, draw=True):
        rgbImg = cv.cvtColor(frame , cv.COLOR_BGR2RGB )
        self.result = self.faceMesh.process(rgbImg)

        if self.result.multi_face_landmarks :
            if draw :
                for facelms in self.result.multi_face_landmarks :
                    self.mpDraw.draw_landmarks(frame, facelms, self.mpFaceMesh.FACEMESH_TESSELATION , self.landmarkCanvas , self.connectionCanvas)
        return frame
    
    def getlm(self , frame , draw = True ) : 
        lmlist = [] 
        h,w,_ = frame.shape
        if self.result.multi_face_landmarks :
            for facelms in self.result.multi_face_landmarks :
                for id ,lm in enumerate(facelms.landmark) :
                    x,y = int(lm.x * w) , int(lm.y*h)
                    lmlist.append([id , x, y])
                    if draw : 
                        cv.putText(frame , str(id) , (x,y) , cv.FONT_HERSHEY_PLAIN , 0.5 , (255,255,255) , 1)
        return lmlist

def main() :
    vid = cv.VideoCapture(0) 
    ctime = 0 
    ptime = 0 
    facemeshDetector = faceMesh(max_num_faces=2)

    while True : 
        isTrue , frame = vid.read() 
        if not isTrue :
            break 
        
        frame = facemeshDetector.findFaceMesh(frame , draw= False)
        lmlist = facemeshDetector.getlm(frame , draw=True )

        if len(lmlist) > 400 : 
            cv.circle(frame , (lmlist[151][1],lmlist[151][2]) , 10 , (0,0,0) , -1)

        ctime = t.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv.putText(frame, f'FPS: {int(fps)}', (10, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv.imshow("FaceMesh Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__" : 
    main()