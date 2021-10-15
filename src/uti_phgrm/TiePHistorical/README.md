# *Historical pipeline* - tie-points extraction in diachronic images

The algorithm uses a python implementation of the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) (Full paper PDF: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763)) sparse point detector and matcher (see the license below). 

For more details about our algorithm, please see our publication, video, blog and vtutorials:
* Zhang, L., Rupnik, E., Pierrot-Deseilligny, M. (2021). Feature matching for multi-epoch historical aerial images. ISPRS Journal of Photogrammetry and Remote Sensing.
* [Introduction video for "Feature matching for multi-epoch historical aerial images"](https://link to be added)
* [Blog for "Feature matching for multi-epoch historical aerial images"](https://github.com/LulinZhang/Feature-matching-for-multi-epoch-historical-images)
* [Tutorial using historical aerial images](https://colab.research.google.com/drive/1poEXIeKbPcJT_2hyQOBhzcj1EEhO8OgD)
* [Tutorial using aerial and satellite images](https://colab.research.google.com/drive/14okQ8bBhEZmy6EGRIQvazTqrN39oc_K5)

<p align="center">
  <img src="TiePHistoP_pipeline.png" width="900">
</p>

## Set-up 

1. Create virtualenv, clone SuperGluePretrainedNetwork and install depedencies (```PYTHON_PATH``` is the path to ```bin/python```):

    ```bash ./install.sh PYTHON_PATH```

Virtualenv files are stored in python_env/, remove the directory to remove the virtualenv.

The SuperGluePretrainedNetwork is necessary for (1) rough co-registration and (2) precise matching when you set "Feature=SuperGlue". If you fail to install SuperGluePretrainedNetwork, you can still perform precise matching by setting "Feature=SIFT" on the condition that your dataset is roughly co-registered.

## Contents

The pipeline is accessible via 

```mm3d TiePHistoP -help```

The command "TiePHistoP" will launch the whole pipeline by automatically calling several subcommands. You can set the "Exe=0" to print all the subcommands instead of executing it.

## License

This code uses third-party code that is not permitted for commercial use. Please refer to [SuperGlue license](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE) for more information.
