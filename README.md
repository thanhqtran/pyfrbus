# pyfrbus
A copy of FRB/US Python code with some notes

## Installing dependencies

On my PC, the latest `suite-sparse` version `7.10.3` is compatible with `scikit.umfpack`

```
brew install suite-sparse
```
You will need to install `scikit.umfpack`
```
pip install scikit-umfpack
```
This is likely to result in errors. 
In that case, navigate to [scikit-umfpack](https://github.com/scikit-umfpack/scikit-umfpack). Clone the git. Navigate to its root folder and install from source
```
pip install .
```
(A working version as of 2025/06/04 is achieved [here](https://github.com/thanhqtran/pyfrbus/tree/main/extra_packages/scikit-umfpack-master))

Next, navigate to the `pyfrbus` root folder and run
```
pip install -e .
```
to install all other dependencies. Note that `sympy==1.3` is needed. If you have newer versions installed, you will likely to encounter errors when the code runs.
