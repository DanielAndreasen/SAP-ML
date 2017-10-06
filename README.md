[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/DanielAndreasen)

[![Updates](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/shield.svg)](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/)
[![Python 3](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/python-3-shield.svg)](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/)

# SAP-ML
Stellar Atmospheric Parameters - Machine Learning.


# Installation
Using `virtualenv` and `pip`

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

# Usage
Input linelist should consists of wavelength in the first column and EW in the
last column. The input file is read with [`np.loadtxt`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html) using standard settings.
Note that there can be other columns, however the first and last have to be
wavelength and EW, respectively.

## Get parameters

```
$ python parametersML.py -l linelist.dat
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

# Citation

Since we use a subset of the line list by [Sousa+ 2008](https://ui.adsabs.harvard.edu/#abs/2008A&A...487..373S/abstract),
we kindly ask you to cite this paper if you use this tool in your research.

We are also very interested if you find this tool useful, so do let us know.

# Known issues

* At the moment this does not include derivation of stellar parameters directly
from a spectrum. We use a subset of the line list by [Sousa+ 2008](https://ui.adsabs.harvard.edu/#abs/2008A&A...487..373S/abstract).
