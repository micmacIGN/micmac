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

==================================================

Function added for oscar after 22/2 meeting :

        ############   For the export : ############

cVirtInterf_NewO_NameManager
{
   ...
              virtual CamStenope * CalibrationCamera(const std::string  & aName) const = 0;
   ...
};

So you can get the calibration used in Martini to transform pixel 2 radian

And then 

class ElCamera 
{
    ..
          Pt2dr Radian2Pixel(const Pt2dr & aP) const;
    ...
};

To transform a "normalized" point in pixel (a CamStenope inehrit from ElCamera)

        ############   For computing accuracy of TieP with relative orientation : ############



cVirtInterf_NewO_NameManager
{
   ...
       virtual std::pair<CamStenope*,CamStenope*> CamOriRel(const std::string &,const std::string &) const =0;
   ...
};

Return a pair oriented camera computed by the first command of Martini

class ElCamera 
{
    ..
  Pt3dr  PseudoInterPixPrec(Pt2dr aPF2A,const ElCamera & CamB,Pt2dr aPF2B,double & aD) const;
   ..
};

Given 2 oriented camera, and 2 point in each image, compute intersection (no usefull here) and accuracy in pixel.
























