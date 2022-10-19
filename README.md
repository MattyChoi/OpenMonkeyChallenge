# OpenMonkeyChallenge

This repo contains a simple template for training models with the PyTorch Lightning framework with Hydra for organizing config files. To show a use case, I've tailored this template to train a pose estimation model for monkeys using the [OpenMonkey Challenge](https://openmonkeychallenge.com/) dataset. 

Once in the directory containing the contents of the repository, run
```
pip install -r requirements.txt
```
if you do not have all the necessary packages listed.

## Navigating the Codebase

This template is separated into many folders but allows for clean separation of different sections for training a neural network. The [config](https://github.com/MattyChoi/OpenMonkeyChallenge/tree/main/config) folder contains yaml files to allow fast changes to parameters of your hyperparameters for training, dataset configurations, model parameters, etc. The other folders contain code relevant to the name of the folder. 

## Train 

To train the model, run the following command in the terminal:
```
python tools/trainer.py
```

and to test the model, 
```
python tools/predictor.py
```
