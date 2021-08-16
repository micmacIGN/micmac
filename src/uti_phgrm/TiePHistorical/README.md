# *Historical pipeline* - tie-points extraction in diachronic images

The algorithm uses a python implementation of the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) sparse point detector and matcher. For more details please see:
* the full paper - add ref
* the tutorial - add ref

<p align="center">
  <img src="TiePHisto_pipeline.png" width="500">
</p>

## Set-up 

1. Create virtualenv, clone SuperGluePretrainedNetwork and install depedencies (```PYTHON_PATH``` is the path to ```bin/python```):

    ```bash ./install.sh PYTHON_PATH```

Virtualenv files are stored in python_env/, remove the directory to remove the virtualenv.

## Contents

The pipeline is accessible via 

```mm3d TiePHistoP -help```
