# *Dense matching* with pluggable image correlators

MicMac allows you to replace its native SGM algorithm rwith learning-based approaches. For the moment, the following image matching algortihms are available
* MicMac SGM
* [PSMNet](https://github.com/JiaRenChang/PSMNet)


## Set-up 

1. Modify the variables inside ```install.sh```:
* ```PYTHON_PATH``` 
* ```TrainedModelFile```

2. Create virtualenv, clone a modified version of PSMNet and install depedencies:

    ```./install.sh```

Virtualenv files are stored in python_env/, remove the directory to remove the virtualenv.

## Contents

The pipeline is accessible via

```MMVII DenseMatchEpipGen MMVII -help``` for MicMac SGM
```MMVII DenseMatchEpipGen PSMNet -help``` for PSMNet
