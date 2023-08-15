import cv2
from deepface import DeepFace
# import cv2
import numpy as np
from PIL import Image
import io
from Utils import *
from yolov5.detect import run
import easyocr

reader = easyocr.Reader(['en'])


def extract_aadhar(data,inp):
    # print(inp)
    data = np.array(data)
    # face_done = False
    results = []
    aadhar_done= False
    bbox_aadhar = {}
    bbox_face = {}
    results.append({'aadhar':{'status':aadhar_done,'content':[]}})                # results.append((bbox['label'],bound[1]))
    image = Image.fromarray(data)
    #<<<<<<<<<<<<<<<<<<<<<< Real time face detection >>>>>>>>>>>>>>>>>>>>>>>>>
    # if inp[0]['face']['status'] == False:
    # # image = Image.open(data)#input is data which is an object of image
    #     faces = DeepFace.extract_faces(data, detector_backend='opencv',enforce_detection=False)
    #     bbox = faces[0]['facial_area']
    #     bbox_face = bbox
        
    #         # Draw the rectangle on the image
    #     img1 = cv2.rectangle(data, (bbox['x'],bbox['y']), (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), (0, 255, 0), 2)  # BGR color: (0, 255, 0), thickness: 2
    #     # cv2.imwrite("./Face/try_face.jpg",img1)
    #     face_done = True
    # else:
    #     face_done = inp[0]['face']['status']
    # # print("<<<<<<<<<extracting aadhar>>>>>>>>>")
    # # img = Image.open(image_path)
    # cv2.imwrite("./Face/new_aadhar_orig_image.jpg",data)

    #<<<<<<<<<<<<<<<<<<<<<<< Real time face detection >>>>>>>>>>>>>>>>>>>>>>>>
    ENCODING = 'utf-8'
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_content = buf.getvalue()

        # second: base64 encode read datapip 
        # result: bytes (again)
    base64_bytes = base64.b64encode(byte_content)

        # third: decode these bytes to text
        # result: string (in utf-8)
    base64_string = base64_bytes.decode(ENCODING)

    res = run(weights='./weights/aadhar_v1.pt',  # model.pt path(s)
                                source=base64_string,  # file/dir/URL/glob, 0 for webcam
                                imgsz=(416, 416),  # inference size (height, width)
                                conf_thres=0.6,  # confidence threshold,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                project = 'detect',
                                device="cpu",
                                nosave=True)
    coords = []
    # image = cv2.imread(image_path)
    if res:
        for i in res:
            for j in i['coords']:
                x = j.item()
                coords.append(int(x))        
        bbox = {'x':coords[0],'y':coords[1],'w':coords[2],'h':coords[3]}
        bbox_aadhar = bbox  
        # img2 = cv2.rectangle(img1, (bbox['x'],bbox['y']), (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), (0, 255, 0), 2)  # BGR color: (0, 255, 0), thickness: 2
        img2 = cv2.rectangle(data, (bbox['x'],bbox['y']), (bbox['w'],bbox['h']), (0, 255, 0), 2)  # BGR c
        # cv2.imwrite("./Aadhar/try_aadhar.jpg",img2)
        aadhar_done = True
        cropped_image = data[bbox['y']:bbox['h'], bbox['x']:bbox['w']]
        # cv2.imwrite("./Aadhar/try_aadhar.jpg",cropped_image)
        # image_path = "./Aadhar/try_aadhar.jpg"
        image = Image.fromarray(cropped_image)
        # cv2.imwrite("./Face/image_cropped.jpg",cropped_image)

        # image = Image.open(image_path)
        ENCODING = 'utf-8'
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_content = buf.getvalue()

            # second: base64 encode read data
            # result: bytes (again)
        base64_bytes = base64.b64encode(byte_content)

            # third: decode these bytes to text
            # result: string (in utf-8)
        base64_string = base64_bytes.decode(ENCODING)
        res = run(weights='./weights/robo.pt',  # model.pt path(s)
                            source=base64_string,  # file/dir/URL/glob, 0 for webcam
                            imgsz=(416, 416),  # inference size (height, width)
                            conf_thres=0.6,  # confidence threshold,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                            project = 'detect',
                            device="cpu",
                            nosave=True)
        bboxes = []
        result = []
            # if res:
        for i in res:
            coords = []
            label = i['label']
            if label =="Adharno":
                for j in i['coords']:
                    x = j.item()
                    coords.append(x)
                bbox = {'x':int(coords[0]),'y':int(coords[1]),'w':int(coords[2]),'h':int(coords[3]),'label':i['label']}
                img3 = cv2.rectangle(cropped_image, (bbox['x'],bbox['y']), (bbox['w'],bbox['h']), (0, 255, 0), 2)  # BGR c
                # cv2.imwrite("./Face/cropped_text.jpg",img3)
                print("I'm aadharno. bbox",bbox)
                # print("cropped_image",cropped_image)
                img3 = cropped_image[bbox['y']:bbox['h'],bbox['x']:bbox['w']]
                # cv2.imwrite("./Face/aadharno.jpg",img3)
                bounds = reader.readtext(img3)
                print("I'm bounds",bounds)
                
        
            
                for bound in bounds:
                    # flat_lst = [j for i in bound[0] for j in i]
                    prob = bound[2]
                    # # print(flat_lst)
                    # min_ = min(flat_lst)
                    # max_ = max(flat_lst)
                    
                    result.append({"label":"Aadhar No.","text":bound[1],"prob":prob})

                results[0]['aadhar']['content'] = result             
        # results.append({'aadhar':{'status':aadhar_done,'content':result},'face':{'status':face_done}})                # results.append((bbox['label'],bound[1]))
    results[0]['aadhar']['status'] = aadhar_done
    results[0]['aadhar']['bbox'] = bbox_aadhar
    # results[0]['face']['status'] = face_done
    # results[0]['face']['bbox'] = bbox_face
    # print("I'm results",results)
    return results    