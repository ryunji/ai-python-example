import PIL.Image
from fastapi import FastAPI, File, UploadFile
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
# 추론기를 먼저 선언
# 추론 객체는 한번만 만들고 재사용 한다. 앱 객체가 만들어지기 전에
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

import io
import PIL
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    
    #이거 읽어줘야한다는데, 이유는? : 아직까지 텍스트라 함.
    byte_file = await file.read()

    # STEP 3: Load the input image. : local 파일이 아닌 메모리상의 이미지를 사용해야 함.
    # image = mp.Image.create_from_file(IMAGE_FILENAMES[0])

    # convert char array to binary array
    # 텍스트를 바이너리 코드로 바꿔 줌.
    # 양자화된 이미지 파일
    image_bin    = io.BytesIO(byte_file)
    
    # create PIL Image from binary array
    # 이미지 디코딩 하여 바이너리 코드로 변경, 비트맵
    pil_img      = PIL.Image.open(image_bin)
    
    # Convert MP Image from PIL IMAGE
    image        = mp.Image(image_format = mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)
    print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    # 현재 코드는 사진이 하나라 classifications[0]
    count = 3
    results = []
    for i in range(count):
        category = classification_result.classifications[0].categories[i]
        results.append({"category":category.category_name, "score": category.score})
        #result = f"{top_category.category_name} - ({top_category.score:.2f})"
                       
    return {"result": results}