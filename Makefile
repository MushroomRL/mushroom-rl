package:
	python3 setup.py sdist

all: package
	python3 -m twine upload dist/*

clean:
	rm -rf dist
	rm -rf build
