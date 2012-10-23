import os
from astroML.decorators import pickle_results


def test_pickle_results():
    filename = 'tmp.pkl'

    @pickle_results('tmp.pkl')
    def foo(x):
        foo.called = True
        return x * x

    # cleanup if necessary
    if os.path.exists(filename):
        os.remove(filename)

    # initial calculation: function should be executed
    foo.called = False
    assert foo(4) == 16
    assert foo.called is True

    # recalculation: function should not be executed
    foo.called = False
    assert foo(4) == 16
    assert foo.called is False

    # recalculation with different input: function should be executed
    foo.called = False
    assert foo(5) == 25
    assert foo.called is True

    # cleanup
    assert os.path.exists(filename)
    os.remove(filename)
