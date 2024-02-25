set -e
# Import the data to check correctness of RPC implemantation
MMVII ImportM32 verif_1A.txt SjiXYZ XingWGS84 NumL0=13 NumLast=30 NameIm=SPOT_1A.tif
MMVII ImportM32 verif_1B.txt SjiXYZ XingWGS84 NumL0=13 NumLast=30 NameIm=SPOT_1B.tif

#=======================================================================================

MMVII EditSet AllIm.xml =  SPOT_1A.tif   ExtPatFile=false
MMVII EditSet AllIm.xml +=  SPOT_1B.tif   ExtPatFile=false
MMVII EditSet ImA.xml =  SPOT_1A.tif   ExtPatFile=false

#=======================================================================================
# IMPORT REFERENCE SENSOR test them and generate virtual GCP
#=======================================================================================
#
#  SPOT_Init  =>  just the inial sensor in MicMac format with a local copy
#
MMVII ImportInitExtSens AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init
#
#  Generate with 2 image and only "A"
#
MMVII TestSensor SPOT_1A.tif SPOT_Init   ExtPatFile=0  TestCDI=true   OutPointsMeasure=RefInit_A  InPointsMeasure=XingWGS84
MMVII TestSensor AllIm.xml   SPOT_Init   TestCDI=true   OutPointsMeasure=RefInit  InPointsMeasure=XingWGS84 NbProc=1

#=======================================================================================
# GENERATE PERTUBATED SENSOR TEST  OPTIMIZE TEST THEM AGAIN
#=======================================================================================

# ---------------------WITH DEGREE 0 ----------------------------------------------------
MMVII  OriParametrizeSensor AllIm.xml SPOT_Init SPOT_Perturb_D0  0  RandomPerturb=0.001 
MMVII TestSensor AllIm.xml   SPOT_Perturb_D0   TestCDI=true    InPointsMeasure=XingWGS84 NbProc=1
MMVII OriBundleAdj AllIm.xml SPOT_Perturb_D0   Adj_SPOT_Deg0 GCPDir=RefInit  GCPW=[0,1]   NbIter=3
MMVII TestSensor AllIm.xml   Adj_SPOT_Deg0   TestCDI=true    InPointsMeasure=XingWGS84 NbProc=1

# ---------------------WITH DEGREE 2 ----------------------------------------------------
MMVII OriParametrizeSensor AllIm.xml SPOT_Init SPOT_Perturb_D2  2  RandomPerturb=0.003
MMVII TestSensor AllIm.xml   SPOT_Perturb_D2   TestCDI=true    InPointsMeasure=XingWGS84 NbProc=1
MMVII OriBundleAdj AllIm.xml SPOT_Perturb_D2   Adj_SPOT_Deg2 GCPDir=RefInit  GCPW=[0,1]   NbIter=3
MMVII TestSensor AllIm.xml   Adj_SPOT_Deg2   TestCDI=true    InPointsMeasure=XingWGS84 NbProc=1

#----------------------  WITH DEGREE 0 + TieP -----------------------------------------------

#------------------- [1]  Perfect data just to check TieP  -------------------------------------------
MMVII  OriParametrizeSensor AllIm.xml SPOT_Init SPOT_D0  0  
MMVII OriBundleAdj AllIm.xml SPOT_D0   Test   GCPDir=RefInit  GCPW=[0,1]  TPDir=V1  TiePWeight=[1] NbIter=1

MMVII OriBundleAdj AllIm.xml SPOT_Perturb_D0   Test   GCPDir=RefInit  GCPW=[0,1]  TPDir=V1  TiePWeight=[1] NbIter=5




# Or : MMVII ImportExtSens AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init  InitialSysCoord=WGS84Degrees

#  MMVII SysCoCreateRTL  AllIm.xml  WGS84Degrees RTL InOri=SPOT_Init
#  MMVII  OriParametrizeSensor AllIm.xml SPOT_Init SPOT_D2_RTL  2 TargetSysCo=RTL


#  Create a sensor W/O perturb
