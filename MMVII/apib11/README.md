MicMac v2 Python3 API
=====================


Dependencies
------------

    pip3 install pybind11 pybind11_mkdoc


Compilation
-----------

First, compile MMv1 and MMv2.

Then, in 'apib11' directory:

    make

The module can be used from this directory without installation (this local version has priority to installed modules).

Installation
------------

    make install

Distribution
------------

The file dist/MMVII-*.whl, created at compilation, can be distributed to machines with same OS, arch and python version.

It can be installed with:

    pip3 install MMVII-*.whl

Upgrade pip if needed:

    python3 -m pip install --upgrade pip

Usage
-----

```python
    import MMVII
```

See 'examples' directory for use cases.
