[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg?style=flat-square)](https://saythanks.io/to/DanielAndreasen)

[![Build Status](https://travis-ci.org/DanielAndreasen/SAP-ML.svg?branch=master)](https://travis-ci.org/DanielAndreasen/SAP-ML)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/da52b2a1511a4226a810d82a3fcee346)](https://www.codacy.com/app/daniel.andreasen/SWEETer-Cat?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DanielAndreasen/SWEETer-Cat&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/da52b2a1511a4226a810d82a3fcee346)](https://www.codacy.com/app/daniel.andreasen/SWEETer-Cat?utm_source=github.com&utm_medium=referral&utm_content=DanielAndreasen/SWEETer-Cat&utm_campaign=Badge_Coverage)
[![Updates](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/shield.svg?style=flat-square)](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/)
[![Python 3](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/python-3-shield.svg?style=flat-square)](https://pyup.io/repos/github/DanielAndreasen/SAP-ML/)

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

This is useable in the optical and tested on nearly 600 FGK dwarf stars. The
wavelengths can be seen in the file called `linelist.lst`.

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
