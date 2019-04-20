## Better Modeling of Incomplete Annotation for Named Entity Recognition 

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.
The code provided is used for the paper "[Better Modeling of Incomplete Annotation for Named Entity Recognition](http://www.statnlp.org/research/ie/zhanming19naacl-ner.pdf)" published in 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (*NAACL*).

__NOTE: I'm planning to implement a (more user-friendly) pytorch version of this project. Let me know in the issue if you are interested in that.__

### Requirements
* DyNet 2.0 
* Python 3.5+

Put your dataset under the data folder. You can obtain the `conll2003` and `conll2002` datasets from other sources. We have put our collected industry datasets `ecommerce` and `youku` under the data directory. 

Also, put your embedding file under the data directory to run. You need to specify the path for the embedding file.

### Running the baslines
To obtain the performance of the `Simple` model used in the paper:
```bash
python3 all_baseline.py --embedding_file ${PATH_TO_EMBEDDING} --model_type simple --entity_keep_ratio 0.5 --dataset conll2003 --use_char_rnn 1
```
Simply change the `0.5` to `1.0`. You can obtain the results of the `Complete` model. Remember to set `use_char_rnn` to 0 when using the Chinese dataset. 

To obtain the performance of other baselines, change the `model_type` from `simple` to `partial` (LSTM-M-CRF), `partial_perceptron` (LSTM-PP) or `transductive perceptron` (LSTM-TP). 

### Running our approaches
```bash
python3 our_approach.py --embedding_file ${PATH_TO_EMBEDDING} --model_type our_hard --entity_keep_ratio 0.5 --dataset conll2003 --use_char_rnn 1
```

Change the `model_type` to `our_soft` to run our soft variant. 


### Future Work
Working on a Neural Partial CRF Suite with PyTorch, which should be a neural network version of the [partial-CRF suite](https://github.com/Oneplus/partial-crfsuite).

### Citation
If you use this software for research, please cite our paper as follows:

```
@inproceedings{jie2019better,
  title={Better Modeling of Incomplete Annotations for Named Entity Recognition},
  author={Jie, Zhanming and Xie, Pengjun and Lu, Wei and Ding, Ruixue and Li, Linlin},
  booktitle={Proceedings of NAACL},
  year={2019}
}
```