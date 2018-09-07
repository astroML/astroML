PYTHON ?= python
NOSETESTS ?= nosetests

SOURCES = Makefile setup.py README.rst astroML

VERSION = 0.2-git

all: build install test

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install

clean:
	$(PYTHON) setup.py clean

inplace:
	$(PYTHON) setup.py build_ext -i

test-code: inplace
	$(NOSETESTS) -s astroML

test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture doc/ doc/modules/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=astroML astroML

test: test-code
