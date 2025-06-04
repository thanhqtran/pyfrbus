# pyfrbus

A copy of the [FRB/US Python](https://www.federalreserve.gov/econres/us-models-python.htm) with some installation notes.

Tested on MacOS 14.7.6 (23H626).

## Installing dependencies

On my PC, the latest `suite-sparse` version `7.10.3` is compatible with `scikit.umfpack`

```
brew install suite-sparse
```
You will also need to install `scikit.umfpack`:

```
pip install scikit-umfpack
```
This may result in errors.  
In that case, navigate to [scikit-umfpack](https://github.com/scikit-umfpack/scikit-umfpack), clone the repo, navigate to its root folder, and install from source:

```
pip install .
```
(A working version as of 2025/06/04 is available [here](https://github.com/thanhqtran/pyfrbus/tree/main/extra_packages/scikit-umfpack-master))

Next, navigate to the `pyfrbus` root folder and run:

```
pip install -e .
```

to install all other dependencies. 
Note that `sympy==1.3` is required. 
If you have a newer version installed, you are likely to encounter errors when the code runs.

## Notes

- Use `test.ipynb` to check out all working examples stored in the `demos` folder.
- A list of equations and the model can be found [here](https://github.com/thanhqtran/pyfrbus/blob/main/documentation/equations.html).
