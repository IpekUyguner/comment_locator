# C/C++ Comment Location Suggestions
This plugin uses the model structure from the paper "Where should I comment my code? A dataset and model for predicting locations that need comments" (https://homes.cs.washington.edu/~mernst/pubs/predict-comments-icse2020-abstract.html)



## PART 1: Creating deep learning model for the IntelliJ plugin:
If you want to train the model from scratch, you can do as follow:

### install requirements
```
pip install -r requirements.txt
```
### install training data and extract it 
```
wget https://groups.inf.ed.ac.uk/cup/comment-locator/data.tar.gz
tar -xf data.tar.gz
```
### training the model 
You can train the model with arguments as follow. You can also specify more arguments from below. 
There is also notebook version 'train_orig_ptl.ipynb' which you can run on Google Colab.
```
python train_orig.py --mode train  --data_train_path  path/to/data/split30_word_data/train/data.txt \
                 --data_validation_path path/to/data/split30_word_data/valid/data.txt \
                 --code_embeddings  path/to/data/code-big-vectors-negative300.bin \
```
command-line arguments 
```
--data_train_path, default="data/split30_word_data/train/data.txt",
--data_test_path, default="data/split30_word_data/test/data.txt",
--data_validation_path, default="data/split30_word_data/valid/data.txt",
--check_point_dir, default="model_save/", help="Directory for saving model during training",
--code_embeddings, default="data/code-big-vectors-negative300.bin", help="Pretrained embeddings",
--embed_size, default=300, type=int, help="Word token dimension, do not change it unless you change preembedding for words",
--max_blks, default=15, type=int, help="Max number of blocks in file",
--max_len_blk, default=15, type=int, help="Max size of a block",
--max_len_stmt, default=15, type=int, help="Max size of a line",
--vocab_size, default=5000, type=int, help="Whole vocabulary size",
--embedding_type, default="avg_wembed", help="Only average embeddings are implemented",

# Model related arguments #
--use_gpu, default=0, type=int, help="1 for GPU usage",
--lstm_hidden_size, default=800, type=int, help="")
--lstm_num_layers, default=1, help="")
--batch_size, default=64, type=int, help="Batch size for training"
--learning_rate, default=0.001, type=float)
--max_epochs, default=1, type=int)
--mode, default="train", help="train or predict")
--model_for_prediction, default="none", help="Path for model to predict"



```

### exporting Pytorch model to ONNX model
To be able to integrate the previous Pytorch model to Plugin, it needs to be converted ONNX format. Following command will will extract it as "model.onnx"

```
python onnx_export.py
```


## PART 2: Comment Locations Suggestion IntelliJ Plugin:
This plugin uses the model from PART 1. It uses inspection to find out comment worthy locations and show them as warning.
When hover over the line, it displays the confidence level for the prediction.


![example](https://user-images.githubusercontent.com/40366759/135456209-e5dd4816-31f6-40f8-8bd9-a577e3f29fcf.png)

### Limitations
- As explained in the paper, the recall value is too low for the model (approx. 15%). It means, it cannot find the most comment worthy locations. However,
precision is higher around 80 %, so if it suggest something, it is more reliable. 

- It is assumed that the each code chunks seperated by empty lines are coherent itself. So, if developer writes code by putting random epty spaces, the model could not make sense out of it.


### Future Work:
 This is an beginning of studies on this subject. There are many directions to be worth to follow in terms of model improvement as well as the Plugin implementation. For example:
- The dataset for the training data can be extended to improve results. 
- The idea can be applied to other languages. However there is no existed dataset for different languages. (Some preliminary analysis done to create such JAVA datasets that can be found under analysis_java_files.ipynb. Java repositories from the link with more than 50 stars  (https://jetbrains.team/p/ccrm/repositories/fl-dataset/files/docs/README.md#download) are investigated.)
- Further step could be also combine this project with projects related to WHAT TO COMMENT, not only WHERE TO COMMENT.

