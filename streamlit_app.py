import os
import streamlit as st
import replicate
from dotenv import dotenv_values
from PIL import Image
config = dotenv_values(".env")
os.environ['REPLICATE_API_TOKEN'] = config['REPLICATE_API_TOKEN']

uploaded_files = st.file_uploader("Choose an image file", accept_multiple_files=True)
image = None
bytes_data = None
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)
image = Image.open("owen.png")

output = replicate.run(
    "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
    input={"image": open("./owen.png", "rb"),
           "prompt": "this is a picture of owen, describe to me what owen is doing and what is going on in this picture"}
)
# The yorickvp/llava-13b model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
res = ""
for item in output:
    # https://replicate.com/yorickvp/llava-13b/versions/6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc/api#output-schema
    res += item
if res and image:
    st.image(image, caption=res)
    #st.write(res)