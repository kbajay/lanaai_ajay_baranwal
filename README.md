# Work for lanai
- Problem definition given in the PDF
- A baseline jupyter notebook was provided

### I used the BBC News Train data to train the model
- The training data has 1490 rows with three columns

## There are three models trained: 
- A SVM based on the training data <br>
- A Simple NN network based on vectorization layer and embedding layer <br>
- A complex NN network with LSTM/Bi-directional <br>

#### SVM model
- The model is trained on the training data and tested on the same
- It utilized TF-IDF vectors
- Linear SVM kernel is used

#### Simple NN
- mimicing a logistic regression
- Used tf Vectorize layer
- Used tf.data.Dataset for feature extraction, such as lowering, removing punctuation, etc.
- Used tf.data.Dataset to split train, val, and test 
- NN learns a word embedding in the training phase

#### Complex NN
- Used LSTM/Bi-directional/dropout layers
- data pipeline is the same as the simple NN

## Conclusion based on the three models
- Due to smaller trainig data size, the comparison isn't fair. 
- SVM, Simple NN, and Complext NN work as good
- SVM and Simple NN trainings are quite fast, but the complext NN training is slow. So is true with the inference
- The confusion matrixes show good results, but they do poorely for some classes due to class imbalance

## Improvements
- Try word embeddings directly, instead of learning in the process
- Try transformers, self-attention next
- Also, we should try LLMs, such as llama2/3, OpenAI with 0-shot or single-shot in-context learning on the prompt to see the results
- Try a larger training and test dataset to see how these model compare 

# References I used
- kaggle
- stackoverflow
- tensorflow.org
- https://www.tensorflow.org/tutorials/keras/text_classification
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization#adapt
- https://github.com/mmalam3/BBC-News-Classification-using-LSTM-and-TensorFlow/blob/main/bbc_news_classification.ipynb 





