The RedTieP tool has been developed by Oscar Martinez-Rubi within the project
Improving Open-Source Photogrammetric Workflows for Processing Big Datasets
The project is funded by the Netherlands eScience Center

This tool reduces the tie-points (homologous points). It uses an algorithm that requires relative orientation
between image pairs. This is done with the NO_AllOri2Im tool. The reduced set of tie-points
can be used in Tapas, which will then require much less memory and will also run faster.
The cost can be a small increase in the orientation errors
(so if you have GCPs we recommend using them to check that errors are still low)

Usage
=====
1) Compute tie-points

mm3d Tapioca All ".*JPG" 1500

2) Compute relative orientation between pairs and put tie-points in Martini format (float, symetric)

mm3d TestLib NO_AllOri2Im ".*JPG" Quick=1

3) Run the reduction tool

mm3d RedTieP ".*JPG"

4) Output reduced tie-points in Homol-Red folder, rename Homol to Homol-Original and Homol-Red to Homol.
   In this way later steps will use the reduced set of tie-points
mv Homol Homol-Original
mv Homol-Red Homol

Advance usage
=============
There are a set of options to configure RedTieP:
 - NumPointsX,NumPointsY configure the target number of tie-points per image pair. For example 12x12 in a 15 Megapixel usually decreases tie-points around 95%. Scale accordingly with your image size
 - SortByNum,Desc configure the order in which images are processed. Default (recommended) is by name in ascending order which usually matches the acquisition order. The user can select to process ordering by name in descending order of by number of tie-points in the images (in both ascending and descending order)
 - ThresholdAccMult configures the weight of the accuracy of a multi-tie-point against its multiplicity (multi-tie-point is the topological merging of tie-points from image pairs that are linked to the same point/location in the reality). Default (recommended) is 0.5.

Noodles
=======
RedTieP uses by default 1 core. To run RedTieP with multiple cores we need to use an external tool called Noodles (for installation instructions, see https://github.com/NLeSC/noodles).
Running RedTieP with the option mExpSubCom=1 outputs a list of tasks in a JSON file.
Running the script in scripts/noodles_exe_parallel.py with the generated JSON file runs the tasks (of RedTieP) in parallel.
Note that any image sorting specified by the user in the RedTieP options will be ignored by Noodles.

Code documentation
==================
All the classes used are declared in TiepRed.h. The cpp files in this folder contain the implementation of the classes.

Even though the files have in-line comments the following paragraphs may help in grasping the general code architecture.

The cAppliTiepRed is the main class of the RedTieP tool.
It consists of a parent process that defines a set of subcommands/tasks (one subcommand/task for each image).
Then the parent process spawns 1 child process that executes the subcommands
Ideally it would spawn several processes. However, since the subcommands have crossdependancies of image pairs, this is not possible within MicMac.
For that purpose we need to use a external tool (Noodles) that can execute the tasks in parallel taking care of crossdependancies.

The parent process code is in the file cAppliTiepRed.cpp. The child (children if Noodles is used) code is in cAppliTiepRed_Algo.cpp (this contains "what" is done in each subcommand/task)

Each subcommand executed by the child/children reduces the tie-points between an image (master image) and its related images (the ones that share tie-points with the master).
In each task/subcommand, we create for each image (in the task) an instance of cImageTiepRed (see cImageTiepRed.cpp). This contain basic info on the image (name, nb tie-points, etc.)
For each image pair in the task, we create an instance of cLnk2ImTiepRed (see cLnk2ImTiepRed.cpp). This contain references to the tie-points between the images, the images and the related camera instances (with info about orientation of the camera from where the pics where taken)

The tie-points of each tasks are merged into multi-tie-points. A multi-tie-point is the topological merging of tie-points from image pairs that are linked to the same point/location in the reality.
Each multi-tie-point is stored in a cPMulTiepRed instance (see cPMulTiepRed.cpp).

For each image in the task, we divide its image space in a grid and for each grid cell we record which multi-tie-points are present. The grid of an image (which multi-tie-points are in which cells) is stored in cImageGrid (see cImageGrid.cpp)

For each cell of the master image grid, we delete the least important tie-points if by doing it we do not affect the good distribution of tie-points in the image pairs.
