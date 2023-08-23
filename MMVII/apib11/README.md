MicMac v2 Python3 API
=====================


Dependencies
------------

As admin:

    apt install python3-pip doxygen

As user:

    pip3 install pybind11 wheel


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

The file dist/MMVII-*.whl, created at compilation, can be distributed to machines with same OS, architecture and python version.

This file contains all the necessary files to run the module:
MMVII does not have to be installed on the machine to use the python module.

It can be installed with:

    pip3 install MMVII-*.whl

Upgrade pip if needed:

    python3 -m pip install --upgrade pip



Usage
-----

```python
    import MMVII
```

The built-in Python help system can be used to have information about the API.

See 'examples' directory for use cases.

