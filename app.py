import streamlit as st
import cv2
from PIL import Image
from vkyc_new import extract_aadhar
import re

def main():
    st.title("Aadhar Card Extraction")

    cap = cv2.VideoCapture(0)  # Open the camera

    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return

    st.write("Capturing live video...")

    result = True
    inp = [{'aadhar': {'status': False, 'content': []}, 'face': {'status': False}}]
    a = 0  # Counter for unique button key

    while result:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break

        # Display the live video feed
        st.image(frame, channels="BGR", caption="Live Video Feed")

        if st.button(f"Extract Aadhar {a}"):  # Use unique key with counter 'a'
            res = extract_aadhar(Image.fromarray(frame), inp)
            inp = res
            st.write("Result:", res)

            if len(res[0]['aadhar']['content']) <= 0:
                result = True
            else:
                text = "".join(x["text"].strip() for x in res[0]["aadhar"]["content"])
                text = re.sub(" +", "", text)
                if len(text) == 12 and bool(re.match('^[0-9]+$', text)):
                    result = False
                else:
                    result = True

        a += 1  # Increment the counter

    cap.release()  # Release the camera
    st.write("Video feed closed.")

if __name__ == '__main__':
    main()
