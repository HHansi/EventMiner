# Created by Hansi at 6/2/2022
import torch

from algo.classification.classification_model import ClassificationModel
from algo.util.file_util import delete_create_folder
from experiments.document_classification.config import TEMP_DIRECTORY

delete_create_folder(TEMP_DIRECTORY)

model = ClassificationModel('bigbird', 'EventMiner/bigbird-roberta-large-en-doc', use_cuda=torch.cuda.is_available(),
                            use_auth_token=True)
predictions, raw_predictions = model.predict(['Police arrested five more student leaders on Monday when implementing the strike call given by MSU students union as a mark of protest against the decision to introduce payment seats in first-year commerce programme.'])
print(predictions)
print(raw_predictions)