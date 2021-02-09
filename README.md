# Incremental Temporal Knowledge Graph Completion

In this project, we explore different ways of training TKGC models with new edges and entities. The basic setting corresponds to training the model using on the edges at time step `t`. 

Installation:
`conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch && pip install dgl-cu101 `
 `&& python -m pip install -U matplotlib && pip install test-tube==0.7.5 && pip install pytorch-lightning==0.8.1`

`conda install -c anaconda ipykernel && python -m ipykernel install --user --name=$your_conda_env_name`



### Train a base model
In our paper, we use the first 70% of the snapshots to train a base model, namely 54 and 42 for wikidata and yago respectively. For example, to train a pretrained DE model for wikidata dataset, run this command:
```
python main.py --module DE -d wiki --n-gpu 3 --end-time-step 54 --patience 100
```

We have done this step for you. The pretrained checkpoints can be found under `experiments/base-model/`. The folder names are formatted as `$module-$dataset`. In this setting, the resulting checkpoint is located in `DE-wiki`. 

### Incremental learning
To replicate the results of TIE in the original paper:
```
python main.py --module DE -d wiki --n-gpu 0 --score-function complex --negative-rate 500 --addition \
            --deletion --self-kd --KD-reservoir --CE-reservoir --historical-sampling --sample-neg-entity
```

  `--n-gpu`: index of the gpus for usage, e.g. `--n-gpu 0 1 2` for using GPU indexed 0, 1 and 2.
  
 `--dataset` or `-d`: name of the dataset, `wiki` or  `yago`.
 
 `--module`: name of the base encoder. Use `Static` for static embedding methods; `DE` for diachronic embeddding; `hyte` for hyperplane-based embedding model.

  `--score-function`: decoding function. Choose among `TransE`, `distmult` and `complex`. Default: complex
  
  `--negative-rate`: number of negative samples per training instance. Note that for both object and subject we sample this amount of negative entities.
  
  `--max-nb-epochs`: maximum number of training epoches at each time step.
  
  `--patience`: stop training after waiting for this number of epochs after model achieving the best performance on validation set.
  
   `--base-model-path`: if `load-base-model` is set to true, the base model path is automatically determined by `module` and `--dataset`. Use this argument only if there is an additional base mode path.
   
   
   `--start-time-step`: index of the first snapshot the model is trained on. Default 0.
   
   `--end-time-step`: index of the last snapshot the model is trained on. Default maximum number of time step in the dataset.
   
   `--num-samples-each-time-step`: number of positive historical facts to sample at each time step
   
   `--train-seq-len`: number of time steps preceding each time step `t`, from which historical facts are sampled. 
   
   Flag arguments:
   
   `--cpu`: running experiments using cpu only.
   
   `--overfit`: running overfitting experiments by evaluating on training set.
   
   `--debug`: instead of redirecting the logs to a log file, print them on stdout.
   
   `--fast`: both training and valication in each epoch are only performed on one batch. Fast sanity check for training and validation.  
   
   `--addition`: at each time step, fine-tuning using added facts compared to the last time step. See section "learning with the added facts" in the paper
   
   `--deletion`: at each time step, fine-tuning using the facts that invalid at the current time step but valid in the last time step. See section "learning with deleted facts" in the paper
   
   `--self-kd` : use temporal regularization. See section "Temporal Regularization".
    
   `--KD-reservoir`: use knowledge distillation loss for experience replay
    
   `--CE-reservoir`: use cross entropy loss for experience replay. To replicate the setting in our paper, set both flags.
   
   `--a-gem`: use A-GEM for training. Please use it together with `--CE-reservoir`.
    
   `--historical-sampling`: use experience replay. Set this flag if either `--KD-reservoir` or `--CE-reservoir` is set to True.
    
   `--sample-neg-entity`: sampling negative samples for each the sampled historical positive edges.
   
   `--load-base-model`: load the pretrained model from `./experiments/base-model`,  The `start-time-step` will be set correspondingly.
   
To reproduce the results of FT:

 ```
python main.py --module DE -d wiki --n-gpu 0 --score-function complex --negative-rate 500 --addition 
```

To reproduce the results of TR:
 ```
python main.py --module DE -d wiki --n-gpu 0 --score-function complex --negative-rate 500 --addition --self-kd 
```

To reproduce the results of FB:
```
python main.py --module DE -d wiki --n-gpu 0 --load-base-model --CE-reservoir --historical-sampling \
                --sample-neg-entity --negative-rate-reservoir 500 --num-samples-each-time-step 10000
```


To reproduce the results of FB_Future:
```
python main.py --module DE -d wiki --n-gpu 0 --train-base-model
```

### Logging and Experiment Management
By default, the experiment logs are stored in the folder `logs` and experiments are stored under `experiments`. The file and folder are named after the date and time of the beginning of the experiment.

### Inference and Visualize
The results on validation set are automatically stored in the experiment folder after each epoch. One can run the jupyter notebook to visualize the results.
To obtain the test results, run the following command:
```
python inference_base_model.py --checkpoint-path $experiment_path --test-set --n-gpu 0
```
where the `$experiment_path` variable is the folder where your checkpoint is located, e.g. `./experiments/DE-wiki-complex-patience-100/version_202010062205`. The flag `--test` indicates that the inference is performed on the test set, otherwise the inference will be conducted on the validation set.
