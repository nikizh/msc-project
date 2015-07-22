# Setup
_Note: These instructions are for OS X_

## Requirements

* Homebrew (Installation instructions - http://brew.sh)

## Install Python

```console
brew install python
```

### Install virtualenvwrapper _(optional)_

https://virtualenvwrapper.readthedocs.org/en/latest/

## Install RDKit

### Install Boost and Boost Python
```console
brew rm boost
brew rm boost-python
brew install boost -–build-from-source
brew install boost-python
```

### Install RDKit

```console
brew install rdkit
```

### Virtualenv _(optional)_
_If you use virtual environment you have to copy RDKit to the virualenv_

```console
>>> cd /usr/local/Cellar/rdkit/2015.03.1/lib/python2.7/site-packages/
>>> cp -r . /Users/YOUR_USER/.virtualenvs/YOUR_VENV_NAME/lib/python2.7/site-packages/
```
