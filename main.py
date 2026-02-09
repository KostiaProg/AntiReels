import cv2
import vlc
from model import get_phones

finished = False
def on_video_ended(event, mediaplayer):
    global finished
    finished = True

def camera() -> cv2.VideoCapture:
    # ask user does he want to see webcam
    ans = ''
    while ans != 'y' and ans != 'n':
        ans = input("Do you want to see webcam (y/n): ").lower()
    show_webcam = True if ans == 'y' else False

    # setup webcam
    webcam = cv2.VideoCapture(0) # 0 default
    if not webcam.isOpened():
        print("Couldn't find a camera")
        return None

    # setup skeleton video
    instance = vlc.Instance()
    mediaplayer = instance.media_player_new()
    mediaplayer.set_media(vlc.Media("else/skeleton.mp4"))
    speed = 0.5

    # when video ends close it
    event_manager = mediaplayer.event_manager()
    event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_video_ended, mediaplayer)

    # webcam loop
    while True:
        ret, frame = webcam.read()
        if ret: # true if frame captured correctly
            speed = camera_logic(frame, mediaplayer, speed, show_webcam)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Couldn't capture a frame")
            break
    
    # close all
    webcam.release()
    mediaplayer.release()
    instance.release()
    cv2.destroyAllWindows()

def camera_logic(frame: cv2.typing.MatLike, mediaplayer, speed: int, show_webcam: bool = True):
    # reset video
    global finished
    if finished:
        finished = False
        mediaplayer.stop()

    # are there any phones in the frame
    if get_phones(frame):
        # replay with higher speed
        if not mediaplayer.is_playing():
            speed += 0.5
            mediaplayer.set_rate(speed)
            mediaplayer.play()

        # add text to webcam
        if show_webcam:
            TEXT = "REELS SCROLLING DETECTED"
            RED = (0, 0, 255)
            FONT = cv2.FONT_HERSHEY_COMPLEX
            FONT_SCALE = 1
            THICKNESS = 1

            ready_size, _ = cv2.getTextSize(TEXT, FONT, FONT_SCALE, THICKNESS)
            position = ((frame.shape[1] - ready_size[0]) // 2, 50)
            cv2.putText(frame, TEXT, position, FONT, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
    else:
        # stop video and reset speed
        speed = 0.5
        if mediaplayer.is_playing():
            mediaplayer.stop()

    # show webcam
    if show_webcam:
        frame = cv2.flip(frame, 1)
        cv2.imshow("Webcam", frame)
    return speed

camera()