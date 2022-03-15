# Deep Filtering

Small repo solving a Filtering problem using [Neural Rough Differential equations](https://arxiv.org/abs/2009.08295).
The loss function is built assuming that the conditional expectation E(X_t | F_t^Y) (where X_t is the state process and Y_t is the observation process) is the orthogonal projection of X_t on the space of rvs F_t^Y - measurable, and Doob-Dynkin lemma.   

Running the code:
```
usage: train.py [-h] [--base_dir BASE_DIR] [--device DEVICE] [--use_cuda] [--seed SEED]
                [--num_epochs NUM_EPOCHS] [--depth DEPTH] [--T T] [--n_steps N_STEPS]
                [--window_length WINDOW_LENGTH] [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --base_dir BASE_DIR
  --device DEVICE
  --use_cuda
  --seed SEED
  --num_epochs NUM_EPOCHS
  --depth DEPTH
  --T T
  --n_steps N_STEPS     number of steps in time discrretisation
  --window_length WINDOW_LENGTH
                        lag in fine time discretisation to create coarse time discretisation
  --plot
```

For example,

```python
python train.py --num_epochs 10 --depth 3 --use_cuda 
```

Plots and neural rde weights are saved in [./numerical_results](https://github.com/msabvid/DeepFiltering/tree/main/numerical_results).

![Filtering](https://github.com/msabvid/DeepFiltering/blob/main/numerical_results/filtering.png)


## Relevant modules
- Neural Rough Differential eqautions using [torchcde](https://github.com/patrick-kidger/torchcde)
- Path signatures using [Signatory](https://signatory.readthedocs.io/en/latest/pages/usage/introduction.html)

