# MMVII OriBundleAdj Im-Calib_OK_Resec.xml  Pannel_BA1_311  Pannel_BA2_311 "GCP3D=[[Pannel,1]]" "GCP2D=[[Pannel,0.1,0.2,1,2]]" 

set -e

NbIterBA_0=2
NbIterBA_1=4

MMVII BlockInstrEdit BL0 FromScratch=1 NPI=[-1]
MMVII BlockInstrEdit BL0 "PatsIm4Cam=[Im-Calib.xml,.*_(.*).tif]"  #   InMeasureClino=MesClinFilA_043
MMVII BlockInstrEdit BL0 InMeasureClino=MesClinFilA_043
MMVII BlockInstrEdit BL0  CstrOrthog=[[A1,A2,1e-6]]
MMVII BlockInstrEdit BL0  CstrOrthog=[[B1,B2,1e-6]]


# ----------------  0 Block inside pannel ----------------------------

#   --- 0.1 Initiale
MMVII BlockInstrInitCam Im-Calib_OK_Resec.xml BL0 Pannel_BA2_311 BL1    

#   --- 0.2  Adjust
MMVII OriBundleAdj Im-Calib_OK_Resec.xml  Pannel_BA2_311  Pannel_BA3_311 "GCP3D=[[Pannel,1]]" "GCP2D=[[Pannel,0.1,0.2,1,2]]" InInstrBlock=BL1 OutInstrBlock=BL2   BOI=[[,0.01,0.01,1],[]] NbIter=${NbIterBA_0}

# ----------------  1 Block on polyg ----------------------------

MMVII OriBundleAdj Im-Clino_OK_Resec.xml  SpaceR_Clino_311 BA_UC_Clino_311  GCP3D=[[Polyg,1]] GCP2D=[[Polyg-Completed,0.1,0.5,-1,2]] PPFzCal=.* NbIter=${NbIterBA_0}


MMVII OriBundleAdj Im-Clino_OK_Resec.xml  BA_UC_Clino_311  BA2_Clino_RigPannel  GCP3D=[[Polyg,1]] GCP2D=[[Polyg-Completed,0.1,0.5,5,2]] PPFzCal=.* InInstrBlock=BL2 OutInstrBlock=BL3    BOI=[[,0.001,0.001,1],[],[0.001,0.001]] NbIter=${NbIterBA_1}

MMVII OriBundleAdj Im-Clino_OK_Resec.xml  BA_UC_Clino_311  BA2_Clino_RigPolyg  GCP3D=[[Polyg,1]] GCP2D=[[Polyg-Completed,0.1,0.5,5,2]] PPFzCal=.* InInstrBlock=BL2 OutInstrBlock=BL4    BOI=[[,0.001,0.001,1],[],] NbIter=${NbIterBA_1}


MMVII BlockInstrInitClino Im-Clino_OK_Resec.xml BL3 BA2_Clino_RigPannel  MesClinPolyg_043  BL5  #  44

MMVII BlockInstrInitClino Im-Clino_OK_Resec.xml BL4 BA2_Clino_RigPolyg  MesClinPolyg_043  BL5 # 20



MMVII OriBundleAdj Im-Clino_OK_Resec.xml   BA2_Clino_RigPolyg BA2_CmpClino  GCP3D=[[Polyg,1]] GCP2D=[[Polyg-Completed,0.1,0.5,5,2]] PPFzCal=.* InInstrBlock=BL5 OutInstrBlock=BL6  NbIter=${NbIterBA_1}  BOI=[[,1,1],[],] InMeasureClino=MesClinPolyg_043  ClinpBOI=[[,1,1],[0],[0,0,0,0]]   



MMVII OriBundleAdj Im-Clino_OK_Resec.xml   BA2_CmpClino BA2_CmpClino2  GCP3D=[[Polyg,1]] GCP2D=[[Polyg-Completed,0.1,0.5,5,2]] PPFzCal=.* InInstrBlock=BL6 OutInstrBlock=BL7  NbIter=${NbIterBA_1}  BOI=[[,1,1],[1],] InMeasureClino=MesClinPolyg_043  ClinpBOI=[[,1,1],[1],[0,0,0,0]]   

#  MMVII BlockInstrInitClino Im-Clino_OK_Resec.xml BL3 BA_UC_Clino_311  MesClinPolyg_043  BL5 NPI=[3]
VoiNono()
{
[0] -> 16.9651
[1] -> 21.9
[2] -> 26.
[3] -> 28.
[0,1,2,3] -> 21.12
}




#BRDirIn=BlockAdjust_${ImM} BRW=[5e-5,2e-5] OutRigBlock=BlocClino_${ImM}



#MMVII BlockInstrEdit BL2  InMeasureClino=MesClinFilA_043 OutInstrBlock=BL3



