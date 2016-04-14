#Usage
======
1) Compute tie points

mm3d Tapioca All _MG_008.*CR2 1500

2) Put in Martini format (float, symetric and relative orientation)

mm3d TestLib NO_AllOri2Im "_.*.CR2" Quick=1

3) Run the reduction tool

mm3d RedTieP  "_.*.CR2"

4) Output created in Homol-Red folder, rename Homol to Homol-Original and Homol-Red to Homol (in this way later steps will use the reduced set of tie-points)
mv Homol Homol-Original
mv Homol-Red Homol


Generate XML API
======================

once you've added your class to the include/XML_GEN/ParamChantierPhotogram.xml file you need to execute this command (from micmac dir)

make -f Makefile-XML2CPP  all

it will generate all the necessary c++ definitions & declarations;



