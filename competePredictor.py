# helper to precassify the images
from help_nsfw_detector import predict224
from help_nsfw_detector import predict299
import tensorflow as tf
import numpy as np
# image_path2 = "backEnd/nsfw_detector/customDetector/classes/porn/20230722-232307_picture.png" # hentai
# image_path2 = "backEnd/nsfw_detector/customDetector/classes/sfw/20230806-173325_picture.png" # sfw
image_path2 = "backEnd/nsfw_detector/customDetector/classes/nsfw/20230729-153424_picture.png" # nsfw zwei frauen

image_path = "./" + image_path2

img_height = 512
img_width = 512

# # My Model
# myModel=tf.keras.models.load_model('./backEnd/nsfw_detector/customDetector/myModels/myModel_small.h5')

# img = tf.keras.utils.load_img(
#     image_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
# #img_array /= 255

# predictions = myModel.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# Downloaded Model
downloadedModel224 = predict224.load_model('./backEnd/nsfw_detector/customDetector/help_nsfw_detector/model/saved_model.h5')
downloadedModel299 = predict299.load_model('./backEnd/nsfw_detector/customDetector/help_nsfw_detector/model/february_2019_nsfw.299x299.h5')

predict_downloaded_df224 = predict224.classify(downloadedModel224, [image_path2])
predict_downloaded_df_out244 = predict_downloaded_df224[image_path2]
predict_downloaded_df299 = predict299.classify(downloadedModel299, [image_path2])
predict_downloaded_df_out299 = predict_downloaded_df299[image_path2]

# class_names = ['porn', 'sfw']
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
# )
# print("*** My Predict ***")
# print("['nsfw', 'porn', 'sfw']")
# print(predictions)
print("*** Predict 244***")
print(predict_downloaded_df_out244)
print("*** Predict 299***")
print(predict_downloaded_df_out299)

log_info = ""
import os
import time

def write_log(image_path: str, log_info:str):
    filename = "1log_" + time.strftime("%Y%m%d") + ".txt"
    log_file_path = os.path.join(image_path, filename)

    # log_info = f"NSFW checked for: {image_path}\n"
    # log_info += f"Creation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    # log_info += f"NSFW detection results:\n{nsfw_detector}\n"
    # log_info += f"******\n"

    if os.path.exists(log_file_path):
        with open(log_file_path, "a") as log_file:
            log_file.write(log_info + f"\n")
    else:
        with open(log_file_path, "w") as log_file:
            log_file.write("name;path;drawings;hentai;neutral;porn;sexy"+ f"\n")
            log_file.write(log_info+ f"\n")
    log_info = ""


#path = "backEnd/nsfw_detector/customDetector/classes/sfw"
#path = "backEnd/nsfw_detector/customDetector/classes/nsfw"
path = "backEnd/nsfw_detector/customDetector/classes/porn"
i = 0
with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".png") and entry.is_file():
            print("***********" + str(i) + "***********")
            i = i + 1
            #print(entry.name, entry.path)
            img_path = "./" + entry.path
            def_classify_224 = predict224.classify(downloadedModel224, [entry.path])
            log_info = entry.name + ";" + entry.path + ";"+ str(round(def_classify_224[entry.path]['drawings'],4)) + ";"+ str(round(def_classify_224[entry.path]['hentai'],4)) + ";"+ str(round(def_classify_224[entry.path]['neutral'],4)) + ";"+ str(round(def_classify_224[entry.path]['porn'],4)) + ";"+ str(round(def_classify_224[entry.path]['sexy'],4))
            write_log(path+"/224", log_info)
            log_info = ""
            def_classify_299 = predict299.classify(downloadedModel299, [entry.path])
            log_info = entry.name + ";" + entry.path + ";"+ str(round(def_classify_299[entry.path]['drawings'],4)) + ";"+ str(round(def_classify_299[entry.path]['hentai'],4))+ ";"+ str(round(def_classify_299[entry.path]['neutral'],4)) + ";"+ str(round(def_classify_299[entry.path]['porn'],4)) + ";"+ str(round(def_classify_299[entry.path]['sexy'],4))
            write_log(path+"/299", log_info)
            log_info = ""