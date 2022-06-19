# EventMiner

EventMiner is developed for multilingual news event detection along with CASE 2021 shared task 1: *Multilingual Protest 
News Detection*. This system is powered by state-of-the-art general transformer models and sentence transformer 
models involving supervised and unsupervised learning techniques. Please refer to the paper [DAAI at CASE 2021 Task 1: 
Transformer-based Multilingual Socio-political and Crisis Event Detection](https://aclanthology.org/2021.case-1.16/) for 
more details about this approach and conducted experiments. 

Our best results won first place in English for the document level task while ranking within the top four solutions 
for other languages: Portuguese, Spanish and Hindi.

## Models

We have open-sourced our best-performing models via the [HuggingFace model hub](https://huggingface.co/EventMiner).
All these models were trained on complete training sets of the multilingual version of GLOCON gold standard dataset released 
with [CASE 2021](https://aclanthology.org/2021.case-1.11/) workshop.

For accessing the models, please refer to the [huggingface_models.py](https://github.com/HHansi/EventMiner/blob/master/experiments/document_classification/huggingface_models.py) 
script or the instructions available with HuggingFace model cards.

## Citation
```
@inproceedings{hettiarachchi-etal-2021-daai,
    title = "{DAAI} at {CASE} 2021 Task 1: Transformer-based Multilingual Socio-political and Crisis Event Detection",
    author = "Hettiarachchi, Hansi  and
      Adedoyin-Olowe, Mariam  and
      Bhogal, Jagdev  and
      Gaber, Mohamed Medhat",
    booktitle = "Proceedings of the 4th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.case-1.16",
    doi = "10.18653/v1/2021.case-1.16",
    pages = "120--130",
}
```


