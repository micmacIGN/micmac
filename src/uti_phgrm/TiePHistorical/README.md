# *Historical pipeline* - tie-points extraction between diachronic images

The algorithm uses a python implementation of the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) sparse point detector and matcher. For more details please see:
* the full paper - add ref
* the tutorial - add ref

<p align="center">
  <img src="hitopipe_pipeline.png" width="500">
</p>

## Set-up 

1. Modify the ```PYTHON_PATH``` variable inside ```install.sh```

2. Create virtualenv, clone SuperGluePretrainedNetwork and install depedencies:

    ```./install.sh```

Virtualenv files are stored in python_env/, remove the directory to remove the virtualenv.

## Contents

The pipeline is accessible via 

```mm3d TiePHistoP -help```
