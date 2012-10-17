all: build install

install: install_basic install_addons

build: build_basic build_addons

basic: build_basic install_basic

addons: build_addons install_addons

build_basic:
	python setup.py build

install_basic:
	python setup.py install

build_addons:
	python setup_addons.py build

install_addons:
	python setup_addons.py install

clean:
	python setup.py clean
	python setup_addons.py clean