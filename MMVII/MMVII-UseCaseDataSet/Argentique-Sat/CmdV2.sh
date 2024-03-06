set -e
# Import V1 points : 
MMVII TiePConvert  ".*tif" MMV1 V1 V1Ext=_Merged-SuperGlue Post=dat

# Import V1 Orientation 
MMVII V1OriConv Ori-Campari_Refined-SuperGlue/  V1Conik #  SYSTEM 


MMVII ImportInitExtSens Crop-IMG_PHR1A_P_201406121049025.tif [Crop-IMG_PHR1A_P_201406121049025.tif,Ori-Campari_Refined-SuperGlue/UnCorExtern-Orientation-eTIGB_MMDimap2-Crop-UnCorExtern-Orientation-eTIGB_MMDimap2-IMG_PHR1A_P_201406121049025.XML.xml.xml]  V1Sat
 

MMVII ImportInitExtSens Crop-IMG_PHR1A_P_201406121049386.tif [Crop-IMG_PHR1A_P_201406121049386.tif,Ori-Campari_Refined-SuperGlue/UnCorExtern-Orientation-eTIGB_MMDimap2-Crop-UnCorExtern-Orientation-eTIGB_MMDimap2-IMG_PHR1A_P_201406121049386.XML.xml.xml] V1Sat



#  create a RTL sys
MMVII SysCoCreateRTL  Crop.*  RTL Z0=0  InOri=V1Sat


MMVII OriChSysCo "Crop.*tif" RTL V1Sat SatRTL
MMVII OriParametrizeSensor "Crop.*" SatRTL RTL 0 

#   MMVII OriChSysCo "OIS.*tif" RTL V1Conik  RTL
MMVII OriChSysCo "OIS.*tif" RTL V1Conik  RTL SysIn=Lambert93


#MMVII TestSensor  OIS-Reech_IGNF_PVA_1-0__1971-06-21__C2844-0141_1971_FR2117_1.* RTL NbProc=1
MMVII TestSensor  .*tif       RTL NbProc=1 OutPointsMeasure=RTLRef SzGen=[7,3]
MMVII TestSensor  OIS-.*.tif  RTL NbProc=1 OutPointsMeasure=Conik-RTLRef SzGen=[7,3]
MMVII TestSensor  Crop.*tif   RTL NbProc=1 OutPointsMeasure=Sat-RTLRef SzGen=[7,3]


MMVII OriBundleAdj   ".*tif" RTL  AdjRTL GCPDir=Conik-RTLRef GCPW=[0,1] TPDir=V1 TiePWeight=[0.1,0.5]


MMVII OriParametrizeSensor "Crop.*" SatRTL RTL 0  RandomPerturb=0.001
MMVII OriBundleAdj   ".*tif" RTL  AdjRTL GCPDir=Conik-RTLRef GCPW=[0,1] TPDir=V1 TiePWeight=[0.1,0.5]




