# Created by Hansi at 6/2/2022
import torch
from scipy.special import softmax

from algo.classification.classification_model import ClassificationModel

texts = [
    'Police arrested five more student leaders on Monday when implementing the strike call given by MSU students union as a mark of protest against the decision to introduce payment seats in first-year commerce programme.']

# model = ClassificationModel('bigbird', 'EventMiner/bigbird-roberta-large-en-doc', use_cuda=torch.cuda.is_available())
model = ClassificationModel('xlmroberta', 'EventMiner/xlm-roberta-large-en-pt-es-doc', use_cuda=torch.cuda.is_available())
predictions, raw_predictions = model.predict(texts)
print(f'predictions: {predictions}')
# print(f'raw predictions: {raw_predictions}')

# convert raw predictions to probabilities
raw_preds_probabilities = softmax(raw_predictions, axis=1)
print(f'probabilities: {raw_preds_probabilities}')
