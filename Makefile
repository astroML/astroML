PYTHON ?= python
NOSETESTS ?= nosetests

SOURCES = Makefile setup.py setup_addons.py README.rst astroML astroML_addons

VERSION = 0.1

all: build install test

install: install_basic install_addons

build: build_basic build_addons

basic: build_basic install_basic

addons: build_addons install_addons

build_basic:
	$(PYTHON) setup.py build

install_basic:
	$(PYTHON) setup.py install

build_addons:
	$(PYTHON) setup_addons.py build

install_addons:
	$(PYTHON) setup_addons.py install

clean:
	$(PYTHON) setup.py clean
	$(PYTHON) setup_addons.py clean

inplace:
	$(PYTHON) setup.py build_ext -i
	$(PYTHON) setup_addons.py build_ext -i

test-code: inplace
	$(NOSETESTS) -s astroML

test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture doc/ doc/modules/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=sklearn astroML

test: test-code

doc-source-tarball: source-tarball
	mv astroML.tgz doc/_static/astroML_$(VERSION).tgz

source-tarball: astroML.tgz

astroML.tgz: $(SOURCES)
	rm -f astroML/*~
	rm -f astroML/*.pyc
	rm -f astroML/*/*~
	rm -f astroML/*/*.pyc
	rm -f astroML/*/*/*~
	rm -f astroML/*/*/*.pyc
	rm -f astroML_addons/*.so
	rm -f astroML_addons/*~
	rm -f astroML_addonns/*.pyc

	cd ../; tar -czvf astroML/astroML.tgz $(addprefix astroML/, $(SOURCES))