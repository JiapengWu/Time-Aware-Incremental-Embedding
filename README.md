# Incremental Temporal Knowledge Graph Completion

In this project, we explore different ways of training TKGC models with new edges and entities. The basic setting corresponds to training the model using on the edges at time step `t`. 

### Incremental learnign with basic setting

```
python main.py --module SRGCN -d wiki --n-gpu 0 --score-function complex --negative-rate 500 
```
  `--n-gpu`: index of the gpus for usage, e.g. `--n-gpu 0 1 2` for using GPU indexed 0, 1 and 2.
  
 `dataset` or `-d`: name of the dataset, `wiki` or  `yago`.
 
 `--module`: name of the base encoder. Use `Static` for static embedding methods; `DE` for diachronic embeddding; `SRGCN` for static RGCN model.

  `--score-function`: decoding function. Choose among `TransE`, `distmult` and `complex`.
  
  `--negative-rate`: number of negative samples per training instance. Note that for both object and subject we sample this amount of negative entities.
  
  `--max-nb-epochs`: maximum number of training epoches at each time step.
  
  `--patience`: stop training after waiting for this number of epochs after model achieving the best performance on validation set.
   
   Flag arguments:
   `--cpu`: running experiments using cpu only.
   
   `--overfit`: running overfitting experiments by evaluating on training set.
   
   `--debug`: instead of redirecting the logs to a log file, print them on stdout.
   
   `--fast`: both training and valication in each epoch are only performed on one batch. Fast sanity check for training and validation.  