# Natural Language Understanding Assignments

## LM assignment
**Goal**: obtain the value of the perplexity lower than 250, to improve the performance of the language model at each step
### Part 1
1. Replace RNN with a Long-Short Term Memory (LSTM) network
2. Add two dropout layers: 
   - one after the embedding layer,
   - one before the last linear layer
3. Replace SGD with AdamW 

### Part 2
Starting from the `LM_RNN` in which you replaced the RNN with an LSTM model, apply the following regularisation techniques:
1. Weight Tying
2. Variational Dropout
3. Non-monotonically Triggered AvSGD 

## NLU assignment
**Goal**: evaluate the performance of slot filling and intent classification task at each step
### Part 1
Modify the baseline architecture Model IAS by:
1. Adding bidirectionality
2. Adding a dropout layer

*Intent classification*: accuracy <br>
*Slot filling*: F1 score with conll

### Part 2
Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. One of the challenges is handling the _sub-tokenization_ issue.

*Note*: The fine-tuning process involves further training a model on a specific task/s after it has been pre-trained on a different (potentially unrelated) task/s.

*Intent classification*: accuracy <br>
*Slot filling*: F1 score with conll

-------
Experiment with [*BERT-base* or *BERT-large*](https://huggingface.co/google-bert/bert-base-uncased). 
