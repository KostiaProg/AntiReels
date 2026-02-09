import cv2
from model import get_phones

def camera() -> cv2.VideoCapture:
    # setup
    webcam = cv2.VideoCapture(0) # 0 default
    skeleton = cv2.VideoCapture("skeleton.mp4")

    if not webcam.isOpened():
        print("Couldn't find a camera")
        return None

    counter = 1
    while True:
        ret, frame = webcam.read()
        if ret: # true if frame captured correctly
            counter = camera_logic(frame, skeleton, counter)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Couldn't capture a frame")
            break
    
    webcam.release()
    skeleton.release()
    cv2.destroyAllWindows()

def camera_logic(frame: cv2.typing.MatLikem, skeleton: cv2.VideoCapture, counter: int):
    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam", frame)

    # are there any phones in the frame
    if get_phones(frame):
        # get video speed
        original_fps = skeleton.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            original_fps = 30 # Default to 30
        speed_up_delay = int((1000 / original_fps) / counter) # increase it

        # show skeleton
        ret, skeleton_frame = skeleton.read()

        if not ret: # video has over, increase speed
            counter += 1
        
        # add text
        RED = (0, 0, 255)
        FONT =  cv2.FONT_HERSHEY_COMPLEX
        FONT_SCALE = 1.5
        THICKNESS = 3
        TEXT = "REELS SCROLLING DETECTED"
        ready_size, _ = cv2.getTextSize(TEXT, FONT, FONT_SCALE, THICKNESS)
        cv2.addText(skeleton_frame, TEXT, ((skeleton_frame.shape[1] - ready_size[0]) // 2, 50), FONT, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

        # actual skeleton
        cv2.imshow(skeleton_frame)

        # increasing speed + exit key
        key = cv2.waitKey(speed_up_delay) & 0xFF
        if key == ord('q'):
            return 1
    else:
        counter = 1

    return counter

camera()