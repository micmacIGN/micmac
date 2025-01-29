set -e
#=======================================================================================
# CREATE RTL SYSTEM / Orient Init in RTL /  Ref Data in RTL
#=======================================================================================

MMVII SysCoCreateRTL  AllIm.xml  RTL InOri=SPOT_Init Z0=0
MMVII  OriParametrizeSensor AllIm.xml SPOT_Init SPOT_INIT_RTL  0
MMVII TestSensor AllIm.xml SPOT_INIT_RTL    TestCDI=true  OutObjCoordWorld=RefInit_RTL OutObjMesInstr=RefInit_RTL   NbProc=1


#=======================================================================================
# IMPORT REFERENCE SENSOR test them and generate virtual GCP
#=======================================================================================
#

MMVII  OriParametrizeSensor AllIm.xml SPOT_Init SPOT_Perturb_D2_RTL  2  RandomPerturb=0.003

MMVII TestSensor AllIm.xml SPOT_Perturb_D2_RTL    TestCDI=true      NbProc=1

MMVII OriBundleAdj AllIm.xml SPOT_Perturb_D2_RTL   Adj_SPOT_RTL_Deg2 GCP2D=[[RefInit_RTL,1]] GCP3D=[[RefInit_RTL,0]]  NbIter=5


#=======================================================================================
# GENERATE PERTUBATED SENSOR AND OPTIMIZE Them
#=======================================================================================


# Or : MMVII ImportExtSens AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init  InitialSysCoord=WGS84Degrees

#  MMVII  OriParametrizeSensor AllIm.xml SPOT_Init SPOT_D2_RTL  2 TargetSysCo=RTL


#  Create a sensor W/O perturb
