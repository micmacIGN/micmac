# *Dense matching* with pluggable image correlators

MicMac allows you to replace its native SGM algorithm rwith learning-based approaches. For the moment, the following image matching algortihms are available
* MMV1 (MicMac SGM)
* [PSMNet](https://github.com/JiaRenChang/PSMNet)


## Set-up

Compile MMVII by following the [README](https://github.com/micmacIGN/micmac/blob/master/MMVII/Readme.md#to-compile-) instructions.

<details>
  <summary>MMVII [Click to expand]</summary>

  There is no specific setting necessary to run the SGM native to MicMac.

</details>

<details>
  <summary>PSMNet [Click to expand]</summary>

1. Modify the variables inside ```install.sh``` and ```run.sh```:
* ```MODELPATH```
* ```DISPSCALE```

2. Create virtualenv, clone a modified version of PSMNet and install depedencies:

```sh
./install.sh
```

Virtualenv files are stored in python_env/, remove the directory to remove the virtualenv.

</details>

<details>
  <summary>Use with Docker [Click to expand]</summary>

When using the pluggable image correlators in docker environment, you no longer need to create the python virtual environment. In this case, use the files ```install_docker.sh``` and ```run_docker.sh```.

</details>

## Contents

The pipeline is accessible via:

* ```MMVII DenseMatchEpipGen MMV1 -Help``` for MicMac SGM
* ```MMVII DenseMatchEpipGen PSMNet -Help``` for PSMNet

> Note that the images must be accompanied by masks, i.e., for a pair of images **ImageL.tif** and **ImageR.tif** there should be two respective masks: **ImageL_Masq.tif** and **ImageR_Masq.tif**.


### Handling big images

<details>
  <summary>[Click to expand]</summary>
To match very big images, e.g., high-resolution satellite images, MicMac will partition the input image into several patches. The default patch size is set to [2000,1500], you can change it with the SzTile parameter, e.g.:

```sh
MMVII DenseMatchEpipGen PSMNet ImageL.tif ImageR.tif SzTile=[1024,1024]
```

By default MicMac will run the matching in parallel on all available processes. To limit the number of simulataneous processes (and avoid running into out-of-memory problems), use the parameter NbProc, e.g.:

```sh
MMVII DenseMatchEpipGen PSMNet ImageL.tif ImageR.tif SzTile=[1024,1024] NbProc=1
```

</details>
