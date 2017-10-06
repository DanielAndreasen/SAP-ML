from __future__ import division
import os
import numpy as np
import pandas as pd
from time import time
from glob import glob
import matplotlib.pyplot as plt
try:
    import cPickle
except ImportError:
    import _pickle as cPickle
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn import linear_model
    sklearn_import = True
except ImportError:
    sklearn_import = False
    print('Install scikit-learn with: pip install sklearn')
import argparse

linelists = glob('linelist/*.moog')
linelists = list(map(lambda x: x[9:], linelists))
wavelengths = pd.read_csv('linelist.lst', delimiter=r'\s+', usecols=('WL',))
wavelengths = list(map(lambda x: round(x[0], 2), wavelengths.values))
wavelengths += ['teff', 'logg', 'feh', 'vt']


def _parser():
    parser = argparse.ArgumentParser(description='Spectroscopic parameters with ML')
    parser.add_argument('-s', '--spectrum',
                        help='Spectrum to analyze')
    parser.add_argument('-l', '--linelist',
                        help='Line list to analyze')
    parser.add_argument('-t', '--train',
                        help='Retrain the classifier',
                        default=False, action='store_true')
    parser.add_argument('-c', '--classifier',
                        help='Which classifier to use',
                        choices=('linear', 'ridge', 'lasso'),
                        default='linear')
    parser.add_argument('--save',
                        help='Save the re-trained model',
                        default=False, action='store_true')
    parser.add_argument('--plot',
                        help='Plot the results from training a new model',
                        default=False, action='store_true')
    return parser.parse_args()


def find_star(star):
    affixes = ('', '_rv', '_rv2')
    for affix in affixes:
        fname = '{}{}.moog'.format(star, affix)
        if fname in linelists:
            return 'linelist/{}'.format(fname)
    else:
        raise IOError('{} not found'.format(star))


def read_star(fname):
    columns = ('wavelength', 'element', 'EP', 'loggf', 'EW')
    df = pd.read_csv(fname, delimiter=r'\s+',
                     names=columns,
                     skiprows=1,
                     usecols=['wavelength', 'EW'])
    return df


def add_parameters(df_all, df, star):
    for parameter in ('teff', 'logg', 'feh', 'vt'):
        df_all.loc[star, parameter] = df.loc[star, parameter]
    return df_all


def merge_linelist(df_all, df, star):
    for wavelength in df['wavelength']:
        df_all.loc[star, wavelength] = df[df['wavelength']==wavelength]['EW'].values[0]
    return df_all


def prepare_linelist(linelist, wavelengths):
    d = np.loadtxt(linelist)
    w, ew = d[:, 0], d[:, -1]
    w = np.array(map(lambda x: round(x, 2), w))
    s = np.zeros(len(wavelengths))
    i = 0
    for wavelength in wavelengths:
        idx = wavelength == w
        if sum(idx):  # found the wavelength
            s[i] = ew[idx][0]
            i += 1
    return s.reshape(1, -1)


def train(clf, save=True, plot=True):
    if not os.path.isfile('combined.csv'):
        df = pd.read_csv('parameters.csv', delimiter=r'\s+',
                         usecols=('linelist', 'teff', 'logg', 'feh', 'vt'))
        df.set_index('linelist', inplace=True)
        # Collect all the EWs
        # Each column is one star
        df_combined = pd.DataFrame(index=df.index, columns=wavelengths)
        fnames = {star: find_star(star) for star in df.index}
        N = len(fnames)
        for i, (star, fname) in enumerate(fnames.iteritems()):
            print('{}/{}'.format(i+1, N))
            df_sub = read_star(fname)
            df_combined = add_parameters(df_combined, df, star)
            df_combined = merge_linelist(df_combined, df_sub, star)
        df_combined.to_csv('combined.csv')
    else:
        df_combined = pd.read_csv('combined.csv')
        df_combined.dropna(axis=1, inplace=True)
        df_combined.set_index('linelist', inplace=True)
    df = df_combined
    xlabel = df.columns.values[:-4]
    ylabel = df.columns.values[-4:]
    X = df.loc[:, xlabel]
    y = df.loc[:, ylabel]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf.fit(X_train, y_train)

    N = len(y_test)
    t = time()
    y_pred = clf.predict(X_test)
    t = time()-t
    speedup = 60*N/t
    print('Calculated parameters for {} stars in {:.2f}ms'.format(N, t*1e3))
    print('Speedup: {} million times'.format(int(speedup/1e6)))

    for i, label in enumerate(ylabel):
        score = mean_absolute_error(y_test[label], y_pred[:, i])
        print('Mean absolute error for {}: {:.2f}'.format(label, score))
        if plot:
            plt.figure()
            plt.plot(y_test[label], y_test[label].values - y_pred[:, i], 'o')
            plt.grid()
            plt.title(label)

    if save:
        with open('FASMA_ML.pkl', 'wb') as f:
            cPickle.dump(clf, f)
    return clf


if __name__ == '__main__':
    args = _parser()
    if args.train and not sklearn_import:
        print('Not possible to train without scikit-learn installed.')
        print('Will use pre-trained model')
        args.train = False

    if args.train:
        if args.classifier == 'linear':
            clf = linear_model.LinearRegression()
        elif args.classifier == 'ridge':
            clf = linear_model.RidgeCV(alphas=[100.0, 0.01, 0.1, 1.0, 10.0])
        else:
            clf = linear_model.LassoLars(alpha=0.001)
        clf = train(clf, save=args.save, plot=args.plot)
    else:
        with open('FASMA_ML.pkl', 'rb') as f:
            clf = cPickle.load(f)

    if not args.spectrum and not args.linelist:
        print('No input (spectrum/line list) provided...')
        raise SystemExit('Bye bye...')

    if args.spectrum:
        raise SystemExit('Please run ARES yourself. This is difficult enough')
    elif args.linelist:
        df = pd.read_csv('combined.csv')
        df.dropna(axis=1, inplace=True)
        wavelengths = np.array(map(lambda x: round(float(x), 2), df.columns[1:-4]))
        x = prepare_linelist(args.linelist, wavelengths=wavelengths)
        p = clf.predict(x)[0]
        print('\nStellar atmospheric parameters:')
        print('Teff:   {:.0f}K'.format(p[0]))
        print('logg:   {:.2f}dex'.format(p[1]))
        print('[Fe/H]: {:.2f}dex'.format(p[2]))
        print('vt:     {:.2f}km/s'.format(p[3]))

    if args.train and args.plot:
        plt.show()
