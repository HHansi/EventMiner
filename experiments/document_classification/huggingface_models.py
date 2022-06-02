# Created by Hansi at 6/2/2022
import torch

from algo.classification.classification_model import ClassificationModel

model = ClassificationModel('bigbird', 'EventMiner/bigbird-roberta-large-en-doc', use_cuda=torch.cuda.is_available(),
                            use_auth_token=True)
predictions, raw_predictions = model.predict('police arrested five more student leaders on monday. the student leaders were picked up from science faculty by sayajigunj police when they were implementing the strike call given by msu students union as a mark of protest against the decision to introduce payment seats in first year commerce programme.')
print(predictions)
print(raw_predictions)