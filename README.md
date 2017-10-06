# SAP-ML
Stellar Atmospheric Parameters - Machine Learning


# Installation
Using `virtualenv` and `pip`

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

# Usage
Use the input from [ARES](https://github.com/sousasag/ARES) to the script:

## Get parameters

```
$ python parametersML.py -l linelist.ares
```

## Train the model
Note that a model is already provided (`FASMA_ML.pkl`)
```
$ python parametersML.py -t -c [linear,ridge,lasso]
```

## Get help
```
$ python parametersML.py -h
```
