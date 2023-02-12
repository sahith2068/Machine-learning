import cv2
import numpy as np
from PIL import Image
import base64

import streamlit as st
from io import BufferedReader, BytesIO


def sketch(image):


    #read image file
    # img=cv2.imread(image)
    image = Image.open(image)
    image = np.array(image)
    

    #Converting an image into gray_scale image
    img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Inverting the image
    img_invert = cv2.bitwise_not(img_gray)

    #Smoothing the image
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)

    #Obtaining the final sketch
    final_img = cv2.divide(img_gray,255-img_smoothing,scale=250)
    #cv2.imshow('Sketch',final_img)
    st.image(final_img)
    
   
    im_rgb = final_img[:, :] #numpy.ndarray
    ret, img_enco = cv2.imencode(".png", im_rgb)  #numpy.ndarray
    srt_enco = img_enco.tostring()  #bytes
    img_BytesIO = BytesIO(srt_enco) #_io.BytesIO
    img_BufferedReader = BufferedReader(img_BytesIO) #_io.BufferedReader

    st.download_button(
    label="Download",
    data=img_BufferedReader,
    file_name="sketch_img.jpeg",
    mime="image/png"
    )

    


def load_image(image_file):
	img = Image.open(image_file)
	return img


if __name__ == '__main__':
    st.header("Get your pencil sketch ready")
    st.subheader("This is a simple app that helps you to get a pencil sketch ready")

    image_file = st.file_uploader("Upload your image",type=["png","jpg","jpeg"])
    if image_file is not None:
        st.image(load_image(image_file), use_column_width=True)
        sketch(image_file)


		