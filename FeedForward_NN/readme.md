# Topic Classification with a Feedforward Network

### Gameplan

- Text processing methods for transforming raw text data into input vectors.


- A Feedforward network consisting of:
    - **One-hot** input layer mapping words into an **Embedding weight matrix**
    - **One hidden layer** computing the mean embedding vector of all words in input followed by a **ReLU activation function**
    - **Output layer** with a **softmax** activation


- The Stochastic Gradient Descent (SGD) algorithm with **back-propagation** to learn the weights of the Neural network.
    - Use (and minimise) the **Categorical Cross-entropy loss** function 
    - Perform a **Forward pass** to compute intermediate outputs
    - Perform a **Backward pass** to compute gradients and update all sets of weights 
    - Implement and use **Dropout** after each hidden layer for regularisation


- Re-train the network by using pre-trained embeddings ([GloVe](https://nlp.stanford.edu/projects/glove/)) trained on large corpora. 


### Data

Subset of the [AG News Corpus](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) and you can find it in the `./data_topic` folder in CSV format:

- `data_topic/train.csv`: contains 2,400 news articles, 800 for each class to be used for training.
- `data_topic/dev.csv`: contains 150 news articles, 50 for each class to be used for hyperparameter selection and monitoring the training process.
- `data_topic/test.csv`: contains 900 news articles, 300 for each class to be used for testing.

### Pre-trained Embeddings

You can download pre-trained GloVe embeddings trained on Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip). No need to unzip, the file is large.

### Save Memory

To save RAM, when we finish each experiment we delete the weights of your network using `del W` followed by Python's garbage collector `gc.collect()`