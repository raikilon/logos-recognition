# Logorec

This directory contains all the files of the main application of this project. This application allows to train classifier, predict the probability that a website is a webshop or to know which services a website offers.

## Contents

Here are listed all the files and subdirectories and their use.

- [Requirements](requirements.txt) - Contains all the needed packages to run the application
- [Docs](docs) - The documentation of the application. All the information to use, update, etc. the different parts of the application are contained here. Please refer to its specific README file
- [Application](logorec) - Directory that contains all the source code and files to run. Please refer to its specific README file
- [Tests](tests) - Directory that contains all the tests suite to be sure that all the components of the application run properly.
  - Execute `python -m unittest discover` inside the folder to execute the tests.
- [Setup](setup.py) - Setup file to install this application as a package

## Installation

These application requires [python](https://www.python.org/) 3.6.2+ to run.

To install all the dependencies run the following command

```
pip install -r requirements.txt
```

Then, to run the application please refer to the README in the [Application](logorec) directory.

## Credits

The project was realized by **Noli Manzoni** (noli.manzoni@students.bfh.ch) for the module [Bachelor Thesis](https://www.ti.bfh.ch/fileadmin/modules/BTI7321-de.xml) at the  [Bern University of Applied Sciences](https://www.bfh.ch).