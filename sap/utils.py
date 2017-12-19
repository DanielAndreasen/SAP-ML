import numpy as np
import pandas as pd
from glob import glob


def create_combined():
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
    return df_combined


def find_star(star):
    linelists = glob('linelist/*.moog')
    linelists = list(map(lambda x: x[9:], linelists))

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
