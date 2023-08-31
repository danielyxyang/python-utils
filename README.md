# python-utils

This repository contains personal Python utility functions and can be directly used as a local Python package without being installed globally or in your virtual environment.

*NOTE: This repository is not designed to be used as the top-level Python package. Otherwise, unexpected behavior might occur due to shadowed stdlib modules.*


## Getting Started

To use these utilities in your project, navigate to the parent folder in which you want to place them.

### Setup as Git Repository
If your project is not a Git repository, you can directly clone `python-utils` into its target folder:
```
git clone https://github.com/danielyxyang/python-utils.git utils_ext
pip install -r utils_ext/requirements.txt
```

### Setup as Git Submodule
If your project is a Git repository, you have to add `python-utils` as a submodule:
```
git submodule add https://github.com/danielyxyang/python-utils.git utils_ext
pip install -r utils_ext/requirements.txt
```
Each time you clone your project's repository with `python-utils` as a submodule, do not forget to initialize and update the submodule:
```
git submodule update --init
```
Similarly, if you pull changes from your project's repository involving changes to the submodule or checkout a branch which points to a different commit of the submodule, do not forget to update the submodule:
```
git submodule update
```

### Setup in Jupyter notebooks
If you want your standalone notebook to use `python-utils` as a Git repository, insert the following lines in the first cell:
```
!git clone https://github.com/danielyxyang/python-utils.git utils_ext
%pip install -q -r utils_ext/requirements.txt
```
If you want your standalone notebook to use your project's repository with `python-utils` as a Git submodule, insert the following lines in the first cell:
```
!git clone YOUR_PROJECT_REPO
%cd YOUR_PROJECT
!git submodule update --init
%pip install -q -r PATH_TO/utils_ext/requirements.txt
```

### Usage
Assuming that `python-utils` is located in the folder `utils_ext` within the top-level Python package, you can import its functions in the following way:
```python
from utils_ext.math import cartesian_product
from utils_ext.plot import Plotter
from utils_ext.tools import Profiler
```

## Make Local Changes

If you need to make local changes to `python-utils` for an individual project, use a separate branch with the name `usedby-PROJECT_NAME`. If the changes or some of them are worth to be included in the `main` branch, open a pull request and include a list of changes which should be taken over.
