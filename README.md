# LM assignment
## Part 1
1. Replace RNN with a Long-Short Term Memory (LSTM) network
2. Add two dropout layers: 
   - one after the embedding layer,
   - one before the last linear layer
3. Replace SGD with AdamW 

## Part 2
Starting from the `LM_RNN` in which you replaced the RNN with an LSTM model, apply the following regularisation techniques:
- Weight Tying 
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD 

# NLU assignment
## Part 1
Modify the baseline architecture Model IAS by:
1. Adding bidirectionality #mettere il tag a true????
2. Adding dropout layer

*Intent classification*: accuracy <br>
*Slot filling*: F1 score with conll

## Part 2
Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. 
> one of the challenges is handling the sub-tokenization issue.

*Note*: The fine-tuning process involves further training a model on a specific task/s after it has been pre-trained on a different (potentially unrelated) task/s.


The models that you can experiment with are [*BERT-base* or *BERT-large*](https://huggingface.co/google-bert/bert-base-uncased). 

*Intent classification*: accuracy <br>
*Slot filling*: F1 score with conll
