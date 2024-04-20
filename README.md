# LM assigment
## Part 1 - 4 pt
1. Replace RNN with a Long-Short Term Memory (LSTM) network
2. Add two dropout layers: 
   - one after the embedding layer,
   - one before the last linear layer
3. Replace SGD with AdamW 

## Part 2 - 11 pt
Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:
- Weight Tying 
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD 

# NLU assigment
## Part 1 - 4 pt 
Modify the baseline architecture Model IAS by:
1. Adding bidirectionality #mettere il tag a true????
2. Adding dropout layer

*Intent classification*: accuracy <br>
*Slot filling*: F1 score with conll

## Part 2 - 11 pt
Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. 
> one of the challenges of this is to handle the sub-tokenization issue.

*Note*: The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.


The models that you can experiment with are [*BERT-base* or *BERT-large*](https://huggingface.co/google-bert/bert-base-uncased). 

*Intent classification*: accuracy <br>
*Slot filling*: F1 score with conll

# Sentiment Analysis assigment - 3 pt
### Aspect Based Sentiment Analysis 
Implement a model based on  Pre-trained Language model (such as BERT or RoBERTa) for the Aspect Based Sentiment Analysis task regarding the extraction of the aspect terms only. 
    
**Dataset**: The dataset that you have to use is the Laptop partition of SemEval2014 task 4, you can download it from [here](https://github.com/lixin4ever/E2E-TBSA/tree/master/data).

**Evaluation**:  For the evaluation you can refer to this [script](https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py) or the official script provided by [SemEval](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) (Baseline, Evaluation and Evaluation link). Report F1, Precision and Recall.

**References**:

- Hu, M., Peng, Y., Huang, Z., Li, D., & Lv, Y. (2019, July). Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 537-546).

**Hint** 
To do this exercise you can adapt the model and the code that you develop for intent classification and slot filling tasks.