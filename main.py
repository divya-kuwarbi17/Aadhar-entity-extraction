import cv2
from PIL import Image
# from video_kyc import extract_aadhar
from vkyc_new import extract_aadhar
import re
videoCaptureObject = cv2.VideoCapture(0)
result = True
inp =[{'aadhar':{'status':False,'content':[]},'face':{'status':False}}]
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("Original_image/NewPicture.jpg",frame)
    cv2.imshow("V",frame)
    if cv2.waitKey(25) & 0xFF ==ord("q"):
        break
    res = extract_aadhar(Image.fromarray(frame),inp)
    inp = res
    print("I'm res",res)
    
    if len(res[0]['aadhar']['content']) <=0:
        result = True
    else:
        # aadh = res[0]['aadhar']
        text = "".join(x["text"].strip() for x in res[0]["aadhar"]["content"])
        # print(text)
        text = re.sub(" +","",text)
        if len(text)==12 and bool(re.match('^[0-9]+$',text)):
            # print("length",type(res[0]['aadhar']['content'][0]['text']))
            # print(text)
            result = False
        else:
            result = True
    
videoCaptureObject.release()
cv2.destroyAllWindows()
