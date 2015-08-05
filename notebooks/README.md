# Notebooks

## Regressions

Title  | Description
------ | -------
[Regression (Linear)][1] | Attempt to reproduce the model from the paper<sup>[1](#paper)</sup>
[Regression (GradientBoostingRegressor)][2] | [Gradient Boosting for regression](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) on the data from the paper<sup>[1](#paper)</sup>
[Regression (GPy)][3] | Gaussian Process Regression on the data from the paper<sup>[1](#paper)</sup>
[Regression (GPy+GU+HMDB)][4] | Gaussian Process Regression on GU data and chemical properties from HMDB
[Regression (GPy+GU+PubChem)][5] | Gaussian Process Regression on GU data and chemical properties from PubChem
[Regression (GPy+HMDB)][6] | Gaussian Process Regression on chemical properties from HMDB

[3]: ./Regression_(GPy).ipynb
[4]: ./Regression_(GPy+GU+HMDB).ipynb
[5]: ./Regression_(GPy+GU+PubChem).ipynb
[6]: ./Regression_(GPy+HMDB).ipynb
[2]: ./Regression_(GradientBoostingRegressor).ipynb
[1]: ./Regression_(Linear).ipynb

## Data Manipulation

Title  | Description
------ | -------
[Merge Data (Sample+PubChem)][20] | Combines the data from the paper and chem properties from PubChem
[Extract Data (HMDB)][21] | Extracts useful data from HMDB sdf file
[Merge Data (Sample+HMDB)][22] | Combines the data from the paper and chem properties from HMDB
[Merge Data (GU+PubChem)][23] | Combines GU data with chem properties from PubChem
[Merge Data (GU+HMDB)][24] | Combines GU data with chem properties from HMDB
[Extract Data (MassBank to MongoDB)][25] | Parses MassBank data and stores it in MongoDB database

[20]: ./data_manipulation/Merge_Data_(Sample+PubChem).ipynb
[21]: ./data_manipulation/Merge_Data_(Sample+HMDB).ipynb
[22]: ./data_manipulation/Extract_Data_(HMDB).ipynb
[23]: ./data_manipulation/Merge_Data_(GU+PubChem).ipynb
[24]: ./data_manipulation/Merge_Data_(GU+HMDB).ipynb
[25]: ./data_manipulation/Extract_Data_(MassBank_to_MongoDB).ipynb

## Other

Title  | Description
------ | -------
[PCA (HMDB)][40] | Principal component analysis on data from the paper combined with chem properties from HMDB
[GP (Tanimoto)][41] | Gaussian Process with Tanimoto kernel on the data from the paper
[GP (Tanimoto+GU)][42] | Gaussian Process with Tanimoto kernel on GU data

[40]: ./PCA_(HMDB).ipynb
[41]: ./GP_(Tanimoto).ipynb
[42]: ./GP_(Tanimoto+GU).ipynb

# Setup
_Note: These instructions are for OS X_

* **Prerequisites:**
  * Homebrew (Installation instructions - http://brew.sh)

## Install Python

```console
brew install python
```

### Install virtualenvwrapper _(optional)_

https://virtualenvwrapper.readthedocs.org/en/latest/

## Install pip requirements

```console
pip install -r requirements.txt
```

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
brew install rdkit --with-inchi
```

### Virtualenv _(optional)_
_If you use virtual environment you have to copy RDKit to the virualenv_

```console
cd /usr/local/Cellar/rdkit/2015.03.1/lib/python2.7/site-packages/
cp -r . /Users/YOUR_USER/.virtualenvs/YOUR_VENV_NAME/lib/python2.7/site-packages/
```

## Install R and Rchemcpp
* R-project: https://www.r-project.org
* Install Rchemcpp package: http://www.bioinf.jku.at/software/Rchemcpp/

*Required for the experiments with Tanimoto kernel*

## Install MongoDB

```console
brew install mongodb
```

To run MongoDB from the Terminal execute:

```console
mongod
```

*Required for the experiments with data from [MassBank](http://massbank.jp)*

<hr />

[<a name="paper">1</a>] [Virtual quantification of metabolites by capillary electrophoresis-electrospray ionization-mass spectrometry: predicting ionization efficiency without chemical standards.](http://www.ncbi.nlm.nih.gov/pubmed/19275147)
