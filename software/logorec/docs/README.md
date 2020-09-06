# Logorec

This directory contains the application's documentation which describes the use of all classes and file contained in this project.

## Usage

First of all  the project dependencies must be installed and then the package imports in the file [app](../logorec/app.py) in `__init__` (see comments) must be changed. Then in this folder execute `sphinx-apidoc -f -o source/ ../logorec/`.

Then, once the process is finished run the make file `make html`.  Now all the needed file are contained in [build](build).

## Credits

The project was realized by **Noli Manzoni** (noli.manzoni@students.bfh.ch) for the module [Bachelor Thesis](https://www.ti.bfh.ch/fileadmin/modules/BTI7321-de.xml) at the  [Bern University of Applied Sciences](https://www.bfh.ch).