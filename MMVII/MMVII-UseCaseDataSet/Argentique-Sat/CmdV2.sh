set -e

#MMVII EditSet Satellite.xml += Crop.*tif
#MMVII EditSet Aerial.xml += OIS-Reech_.*tif
#MMVII EditSet AllIm.xml += .*tif

# Import V1 points : 
#MMVII TiePConvert  ".*tif" MMV1 V1 V1Ext=_Merged-SuperGlue Post=dat
MMVII TiePConvert  AllIm.xml MMV1 V1 V1Ext=-SG-All Post=dat

# Import V1 Orientation 
MMVII V1OriConv Ori-Campari_Refined-SuperGlue/  V1Conik #  SYSTEM 


#MMVII ImportInitExtSens Crop-IMG_PHR1A_P_201406121049025.tif [Crop-IMG_PHR1A_P_201406121049025.tif,Ori-Campari_Refined-SuperGlue/UnCorExtern-Orientation-eTIGB_MMDimap2-Crop-UnCorExtern-Orientation-eTIGB_MMDimap2-IMG_PHR1A_P_201406121049025.XML.xml.xml]  V1Sat
MMVII ImportInitExtSens Satellite.xml '[([A-z]{4})-(.*).tif,Ori-Campari_Refined-SuperGlue/UnCorExtern-Orientation-eTIGB_MMDimap2-$1-UnCorExtern-Orientation-eTIGB_MMDimap2-$2.XML.xml.xml]'  V1Sat

#MMVII ImportInitExtSens Crop-IMG_PHR1A_P_201406121049386.tif [Crop-IMG_PHR1A_P_201406121049386.tif,Ori-Campari_Refined-SuperGlue/UnCorExtern-Orientation-eTIGB_MMDimap2-Crop-UnCorExtern-Orientation-eTIGB_MMDimap2-IMG_PHR1A_P_201406121049386.XML.xml.xml] V1Sat



#  create a RTL sys
#MMVII SysCoCreateRTL  Crop.*  RTL Z0=0  InOri=V1Sat
MMVII SysCoCreateRTL Satellite.xml  RTL Z0=0  InOri=V1Sat

#MMVII OriChSysCo "Crop.*tif" RTL V1Sat SatRTL
#MMVII OriParametrizeSensor "Crop.*" SatRTL RTL 0 
MMVII OriChSysCo Satellite.xml RTL V1Sat SatRTL
MMVII OriParametrizeSensor Satellite.xml SatRTL RTL 0 

#   MMVII OriChSysCo "OIS.*tif" RTL V1Conik  RTL
#MMVII OriChSysCo "OIS.*tif" RTL V1Conik  RTL SysIn=L93
MMVII OriChSysCo Aerial.xml RTL V1Conik  RTL SysIn=L93

 
MMVII TestSensor  AllIm.xml       RTL NbProc=1 OutObjCoordWorld=RTLRef OutObjMesInstr=RTLRef SzGen=[7,3]
MMVII TestSensor  Aerial.xml  RTL NbProc=1 OutObjCoordWorld=Conik-RTLRef OutObjMesInstr=Conik-RTLRef SzGen=[7,3]
MMVII TestSensor  Satellite.xml  RTL NbProc=1 OutObjCoordWorld=Sat-RTLRef OutObjMesInstr=Sat-RTLRef SzGen=[7,3]


MMVII OriBundleAdj  AllIm.xml RTL  AdjRTL GCP2D=[[Conik-RTLRef,1]] GCP3D=[[Conik-RTLRef,0]] TPDir=V1 TiePWeight=[0.1,0.5]


#MMVII OriParametrizeSensor "Crop.*" SatRTL RTL 0  RandomPerturb=0.001
#MMVII OriBundleAdj   ".*tif" RTL  AdjRTL GCPDir=Conik-RTLRef GCPW=[0,1] TPDir=V1 TiePWeight=[0.1,0.5]
MMVII OriParametrizeSensor Satellite.xml SatRTL RTL 0  RandomPerturb=0.001
MMVII OriBundleAdj  AllIm.xml RTL  AdjRTL GCP2D=[[RTLRef,1]] GCP3D=[[RTLRef,0]] TPDir=V1 TiePWeight=[0.1,0.5]




