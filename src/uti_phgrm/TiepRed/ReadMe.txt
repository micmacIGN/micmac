Manipulation of tie points after first step of  Martini processing. Use the class cVirtInterf_NewO_NameManager, defined in "include/XML_GEN/xml_gen2_mmByp.h".



==================================================
Classe for topological merging in  "general/exemple_phgr_formel.h"


    cVarSizeMergeTieP                     => One Multiple point
    cStructMergeTieP<cVarSizeMergeTieP>   => All multiple point

==================================================

Exemple of use with my data set

  *Preliminary command
1) compute tie point

mm3d Tapioca All _MG_008.*CR2 1500

2) Put in Martini format (float, symetric and relative orientation)

mm3d TestLib  NO_AllOri2Im "_.*.CR2" OriCalib=Calib Quick=1


mm3d TestOscar  "_.*.CR2"  OriCalib=Calib 

 Get Nb Images 4
_MG_0080.CR2 _MG_0081.CR2 1408 1408 Rec=0.879833
_MG_0080.CR2 _MG_0082.CR2 144 144 Rec=0.607377
_MG_0080.CR2 _MG_0083.CR2 146 146 Rec=0.492564
_MG_0081.CR2 _MG_0082.CR2 190 190 Rec=0.736466
_MG_0081.CR2 _MG_0083.CR2 113 113 Rec=0.698989
_MG_0082.CR2 _MG_0083.CR2 1112 1112 Rec=0.800119


Generate XML API
======================

once you've added your class to the xml class you need to execute this command (from micmac dir)

make -f Makefile-XML2CPP  all

it will generate all the necessary c++ definitions & declarations;
