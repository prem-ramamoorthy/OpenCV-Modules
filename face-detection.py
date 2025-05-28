import cv2 as cv
import mediapipe as mp 
import time

class FaceDetection:
    def __init__(self , min_detection_confidence=0.5, model_selection=0 ):
         self.minDetection = min_detection_confidence
         self.modelSelection = model_selection

         self.mpFaceDetection = mp.solutions.face_detection
         self.faceDetection = self.mpFaceDetection.FaceDetection(
            model_selection=self.modelSelection,
            min_detection_confidence=self.minDetection
         ) 
         self.mpDraw = mp.solutions.drawing_utils
    
    def facedetection(self , img , draw = True ) :
         rgbImg = cv.cvtColor(img , cv.COLOR_BGR2RGB)
         self.result = self.faceDetection.process(rgbImg)
         if self.result.detections: 
              if draw : 
                for id ,detection in enumerate(self.result.detections)  : 
                    self.mpDraw.draw_detection(img , detection)

         return img
    
    def facePosition(self , frame , draw = True) : 
         h , w, _ = frame.shape
         lmlist = []
         if self.result.detections :
            for id , detection in enumerate(self.result.detections) : 
                bboxC = detection.location_data.relative_bounding_box
                confidence = detection.score
                bbox = int(bboxC.xmin * w ) , int(bboxC.ymin * h ) , int(bboxC.width*w) , int(bboxC.height*h)
                lmlist.append([id , bbox , confidence])
                self.attributes  = bbox
                if draw : 
                    x, y, w_box, h_box = bbox
                    cv.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 0), 1)
                    cv.putText(frame , str(int(confidence[0] * 100)) , (bbox[0] , bbox[1] -20 ) , cv.FONT_HERSHEY_SIMPLEX , 1, (255,255,255) , 1 )
         return lmlist
    
    def fancyDraw(self , img , l =30 , t = 10 ) :
        x, y ,w , h = self.attributes
        x1 , y1 = x+w , y+h 

        cv.line(img , (x,y) , (x+l , y) , (255,0,255) , t )
        cv.line(img , (x,y) , (x , y+l) , (255,0,255) , t )

        cv.line(img , (x1,y1) , (x1 , y1-l) , (255,0,255) , t )
        cv.line(img , (x1,y1) , (x1-l , y1) , (255,0,255) , t )

        cv.line(img , (x1,y) , (x1 , y+l) , (255,0,255) , t )
        cv.line(img , (x1,y) , (x1-l , y) , (255,0,255) , t )

        cv.line(img , (x,y1) , (x , y1-l) , (255,0,255) , t )
        cv.line(img , (x,y1) , (x+l , y1) , (255,0,255) , t ) 

        return img 

def main() :
    vid = cv.VideoCapture(0)
    ctime = 0 
    ptime = 0 
    facedetector = FaceDetection(min_detection_confidence=0.90)

    while True : 
         isTrue , frame = vid.read() 
         if not isTrue : 
              break 
         
         frame = facedetector.facedetection(frame , draw = False)
         lmlist = facedetector.facePosition(frame , )
         frame = facedetector.fancyDraw(frame , l = 20  , t = 10)
         ctime = time.time()
         fps = 1 / (ctime - ptime)
         ptime = ctime

         cv.putText(frame, f'FPS: {int(fps)}', (10, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
         cv.imshow("Pose Detection", frame)
         if cv.waitKey(1) & 0xFF == ord('d'):
             break
        #  cv.waitKey(20)
    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__" :
     main()