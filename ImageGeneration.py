# Introduction
# This AI will create a model to determine a NSFW content, 
# which is producted by a Stable Diffusion Model. 
# Untill now there are only bad NSFW detectors. 
# We will try to make one model specific, 
# by creating safe and not safe content from the SD Model, 
# classifying it manualy and creating this ML model to classify the content. 
# It should be applicable to any model.

# Start the SD Server

import pandas as pd
import json
import base64
import time
import requests

# helper to precassify the images
from help_nsfw_detector import predict
model = predict.load_model('./backEnd/nsfw_detector/customDetector/help_nsfw_detector/model/saved_model.h5')


# move files
import os
import shutil

# Load Prompts
SafePosPrompt = pd.read_csv('./backEnd/nsfw_detector/customDetector/safe pos prompts.csv', delimiter=";", skiprows=range(1, 1700))
NotSafePosPrompt = pd.read_csv('./backEnd/nsfw_detector/customDetector/not safe pos prompts.csv', delimiter=";")
NegPrompt = pd.read_csv('./backEnd/nsfw_detector/customDetector/neg prompt.csv', delimiter=";")

# Porn treshhold
porn_treshhold = 0.1

#for row in range(len(SafePosPrompt)):
#    print(SafePosPrompt.at[row, 'prompt'])

# Setup the Server
url = "http://192.168.0.109:7861"
# this will change the model on the backend server
option_payload = {
    #"sd_model_checkpoint": "AnythingV5_v32.safetensors", # switch model
    "sd_model_checkpoint": "animeModel_UpdateV10ImUsing.safetensors", # switch model
}
response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)

# Generate Pictures
# Helper Defs
def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))
def save_encoded_image(b64_image: str, output_path: str):
    filename = time.strftime("%Y%m%d-%H%M%S") + "_picture.png"
    full_path = output_path + filename
    """
    Save the given image to the given output path.
    """
    with open(full_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))
    return full_path
def nsfw_detector(image_path:str):
    predict_df = predict.classify(model, [image_path])
    hentai = predict_df[image_path]["hentai"]
    porn = predict_df[image_path]["porn"]
    if porn > porn_treshhold or hentai > porn_treshhold:
        return True, porn, hentai # explicit content
    else:
        return False, porn, hentai # not explicit content

# Generate Images
txt2img_url = 'http://192.168.0.109:7861/sdapi/v1/txt2img'
df_prompts = SafePosPrompt
print(str(len(df_prompts)) + " images will be generated - NSFW")
for row in range(len(df_prompts)):
    data = {'prompt': df_prompts.at[row, 'prompt'],
            'negative_prompt': "(worst quality, low quality:1.4), monochrome, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed eyes, ((disfigured)), bad art, deformed, ((extra limbs)), ((duplicate)), morbid, multilated, bad body, on hand with less than 5 fingers, crown , stacked torses, stacked hands, totem pole",
            'width': 512,
            'height': 512,
            'steps': 50}
    response = submit_post(txt2img_url, data)
    fullpath = save_encoded_image(response.json()['images'][0], "backEnd/nsfw_detector/customDetector/classes/new_pictures/")
    isNSFW, porn, hentai = nsfw_detector("./" + fullpath)
    hentai = round(hentai * 100)
    porn = round(porn * 100)
    if isNSFW:
        print(str(row) + "- UNAPROPRIATE, Hentai " + str(hentai) +"% Porn: " + str(porn) + "%")
        shutil.move(fullpath, "backEnd/nsfw_detector/customDetector/classes/nsfw")
    else:
        print(str(row) + "- OK, Hentai " + str(hentai) +"% Porn: " + str(porn) + "%")
        shutil.move(fullpath, "backEnd/nsfw_detector/customDetector/classes/sfw")
    print("PROMPT: " + df_prompts.at[row, 'prompt'])
    # print(row , end='->')