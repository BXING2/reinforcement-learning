## General
This example demonstrates funetuning BERT model for text classification tasks. 

## Dataset
The original dataset has around 40K movie reviews with positive and negative reviews labeled as 1 and 0, respectively. In this example, 1K positive reviews and 1K negative reviews are sampled for finetuning the model. The dataset is splited into three parts, 60% for training, 20% for validation and 20% for testing. 

Dataset Link: https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis

## Model
The model is BERT (Bidirectional Encoder Representations from Transformers) Base model with a sequence classification head (BertForSequenceClassification) from HuggingFace. The model consists of the embedding layer, 12 encoder layers, and classification layer. The weights of the 11th encoder layer (index start from 0) and the classification layer are finetuned for 20 epoches, with all other model parameters frozen.

## Evaluation
<img src="figures/train_return.png" height="300" /> <img src="figures/train_step_count.png" height="300" />

**Figure 1. Average return and number of steps during training.**

<img src="figures/test_return.png" height="300" /> <img src="figures/test_step_count.png" height="300" />

**Figure 2. Average return and number of steps during testing.**


Through finetuning, the model achieves an accuracy of 90.5% on the test dataset. From the confusion matrix in Table 2, the model exhibits similar performance on detecting positive and negative reviews on this balanced dataset

## Reference
1. https://huggingface.co/docs/transformers/en/model_doc/bert
2. Kenton, Jacob Devlin Ming-Wei Chang, and Lee Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." Proceedings of naacL-HLT. Vol. 1. 2019.
