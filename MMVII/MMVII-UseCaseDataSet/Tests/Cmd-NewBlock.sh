# MMVII OriBundleAdj Im-Calib_OK_Resec.xml  Pannel_BA1_311  Pannel_BA2_311 "GCP3D=[[Pannel,1]]" "GCP2D=[[Pannel,0.1,0.2,1,2]]" 

set -e

MMVII BlockInstrEdit BL0 "PatsIm4Cam=[Im-Calib.xml,.*_(.*).tif]"  #   InMeasureClino=MesClinFilA_043
MMVII BlockInstrEdit BL0 InMeasureClino=MesClinFilA_043
MMVII BlockInstrEdit BL0  CRO=[[A1,A2,1e-6,i-kj]]
MMVII BlockInstrEdit BL0  CRO=[[B1,B2,1e-6,i-kj]]



MMVII BlockInstrInitCam Im-Calib_OK_Resec.xml BL0 Pannel_BA2_311 BL1    


MMVII OriBundleAdj Im-Calib_OK_Resec.xml  Pannel_BA2_311  Pannel_BA3_311 "GCP3D=[[Pannel,1]]" "GCP2D=[[Pannel,0.1,0.2,1,2]]" InInstrBlock=BL1 OutInstrBlock=BL2   BOI=[[,0.01,0.01,1],[]] NbIter=10


#MMVII BlockInstrInitClino Im-Calib_OK_Resec.xml BL2 Pannel_BA3_311 


#MMVII BlockInstrEdit BL2  InMeasureClino=MesClinFilA_043 OutInstrBlock=BL3



