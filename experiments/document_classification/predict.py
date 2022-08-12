# Created by Hansi at 8/12/2022
import torch
import time

from algo.classification.classification_model import ClassificationModel
from experiments.document_classification.config import MODEL_TYPE, MODEL_NAME, config, CUDA_DEVICE

text_en = "Beijing to build city's tallest building - People's Daily Online\nBeijing to build city's tallest building\n08:16, September 20, 2011\nBEIJING, Sept. 19 (Xinhua) -- A groundbreaking ceremony for a new skyscraper took place in Beijing's central business district (CBD) on Monday, marking the beginning of construction on what will eventually be the city's tallest skyscraper.The design of the China Zun building was in"

print(f'loading model')
start_time = time.time()
model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config,
                            use_cuda=torch.cuda.is_available(), cuda_device=CUDA_DEVICE)
end_time = time.time()
print(f'Model loaded in {(end_time - start_time)} seconds \n')
print(f'loaded: {MODEL_NAME}')

print(f'sleeping')
time.sleep(10)

print(f'predicting')
start_time = time.time()
predictions, raw_predictions = model.predict([text_en])
end_time = time.time()
print(f'Predicted in {(end_time - start_time)} seconds \n')

print(f'predictions: {predictions}')
print(f'raw predictions: {raw_predictions}')
