import pytest
import pandas as pd
from utils import find_star, read_star


def test_find_star_exists():
    star = 'HD16008'
    actual = find_star(star)
    expected = 'linelist/HD16008_rv.moog'
    assert actual == expected


def test_find_star_not():
    star = 'wrong'
    with pytest.raises(IOError):
        find_star(star)


def test_read_star():
    fname = find_star('HD16008')
    df = read_star(fname)

    assert df.shape[0] > 100
    assert df.shape[1] == 2
    assert isinstance(df, pd.DataFrame)
    assert df.columns[0] == 'wavelength'
    assert df.columns[1] == 'EW'
