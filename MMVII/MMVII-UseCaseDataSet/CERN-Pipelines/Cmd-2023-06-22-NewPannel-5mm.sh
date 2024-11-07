###################################################################################
###################################################################################
########                                                                     ######
########                   PREPARATION OF THE PROJECT                        ######
########                                                                     ######
###################################################################################
###################################################################################

set -e

      #====================   Create sets of images ===============================

MMVII EditSet AllImCalib.xml += .*JPG FFI0=[0136,0195]  PatFFI0=['.*_(.*).JPG','$1']
MMVII EditSet AllImClino.xml += .*JPG FFI0=[0196,0208]  PatFFI0=['.*_(.*).JPG','$1']
MMVII EditSet AllImFil.xml += .*JPG FFI0=[0230,0251]  PatFFI0=['.*_(.*).JPG','$1']

MMVII EditSet AllIm.xml  +=  AllImCalib.xml
MMVII EditSet AllIm.xml  +=  AllImClino.xml

      #====================   Create MetaData & Initial Calib=====================

MMVII EditCalcMTDI Std ModelCam ImTest=043_0136.JPG  Modif=[.*.JPG,"NIKON D5600",0] Save=1
MMVII EditCalcMTDI Std Focalmm ImTest=043_0136.JPG  Modif=[.*.JPG,24,0] Save=1
MMVII EditCalcMTDI Std AdditionalName ImTest=043_0136.JPG  Modif=["(.*)_.*","\$1",0] Save=1

MMVII OriCreateCalib AllImCalib.xml CalibInit_311

###################################################################################
###################################################################################
########                                                                     ######
########                   PANNEL PROCESSING                                 ######
########                                                                     ######
###################################################################################
###################################################################################


   #---------------------------------------------------------------------------
   #---------------------- Pannel measure extraction --------------------------
   #---------------------------------------------------------------------------

         #====================   Create the specification of encoding ========

MMVII  CodedTargetGenerateEncoding CERN 14  NbDig=4
MMVII  CodedTargetGenerate  CERN_Nbb14_Freq14_Hamm1_Run1000_1000_SpecEncoding.xml

         #=======   IMPORT GCP & EXTRACT TARGET & CHECK (vs Homography)  =====

MMVII ImportGCP  Data-Input/Pannel5mm.obc  NXYZ Pannel NbDigName=4
MMVII CodedTargetCircExtract AllImCalib.xml  CERN_Nbb14_Freq14_Hamm1_Run1000_1000_FullSpecif.xml DiamMin=8 OutPointsMeasure=Pannel
MMVII OriPoseEstimCheckGCPDist AllImCalib.xml Pannel CERNFilter=Data-Input/GeomCERNPannel.xml  DirFiltered=FilteredNoCal


   #---------------------------------------------------------------------------
   #---------------------- Orientation w/o block rigid --------------------------
   #---------------------------------------------------------------------------


         #===========   First Orientation : Resec + Bundle + Check with Calib =======

MMVII OriPoseEstimSpaceResection  SetFiltered_GCP_OK_Resec.xml FilteredNoCal CalibInit_311 Resec_311_A ThrRej=5.0 DirFiltered=Filt_Res_311_A
MMVII OriBundleAdj SetFiltered_GCP_OK_Resec.xml Resec_311_A BA_311_A GCPDir=Filt_Res_311_A GCPW=[1,1,3.0]

MMVII OriPoseEstimCheckGCPDist AllImCalib.xml Pannel CERNFilter=Data-Input/GeomCERNPannel.xml  DirFiltered=FilteredCalib Calib=BA_311_A

         #==================  Filter again + refin position of targets + 2nd BA ======================

MMVII OriPoseEstimCheckGCPDist AllImCalib.xml Pannel CERNFilter=Data-Input/GeomCERNPannel.xml  DirFiltered=FilteredCalib Calib=BA_311_A
MMVII CodedTargetRefineCirc SetFiltered_GCP_OK_Resec.xml FilteredCalib BA_311_A OutPointsMeasure=Refined
MMVII OriBundleAdj SetFiltered_GCP_OK_Resec.xml BA_311_A BA_311_B GCPDir=Refined GCPW=[1,1,0.5]

   #---------------------------------------------------------------------------
   #----------------- Rigig Block : Init Calibration  + Adjjust----------------
   #---------------------------------------------------------------------------

MMVII BlockCamInit SetFiltered_GCP_OK_Resec.xml  BA_311_B   "(.*)_(.*).JPG" [1,2]  '[(.*)@(.*)\,$1_$2.JPG,@]' Rig_311_B
MMVII OriBundleAdj SetFiltered_GCP_OK_Resec.xml  BA_311_B BA_311_C GCPDir=Refined GCPW=[1,1,0.5] BRDirIn=Rig_311_B BRW=[1e-2,1e-5]  OutRigBlock=Rig_311_C


###################################################################################
###################################################################################
########                                                                     ######
########                   CLINO PROCESSING                                  ######
########                                                                     ######
###################################################################################
###################################################################################

         # TODO =>   Put the rigbloc as hard constraint,  by the way for now its does not appear to very stable

   #---------------------------------------------------------------------------
   #-----------------  Import GCP & Image Measures  ---------------------------
   #---------------------------------------------------------------------------

         #--- Note MulCoord to have everything in mm, as before
 
MMVII ImportGCP Data-Input/2023-06-19_16h15_Polar_Module.coo "NXYZ" Clino NumL0=23 NumLast=43 PatName=['P\.(.*)','$1']  NbDigName=4 MulCoord=1000.0
MMVII  CodedTargetCircExtract AllImClino.xml  CERN_Nbb14_Freq14_Hamm1_Run1000_1000_FullSpecif.xml DiamMin=8 OutPointsMeasure=Clino


   #---------------------------------------------------------------------------
   #-------- Compute Orientaion : Resec+BA, Calib is imported from Pannel  ----
   #---------------------------------------------------------------------------
         
MMVII  OriPoseEstimSpaceResection  AllImClino.xml  Clino  BA_311_C  Resec_Clino ThrRej=5.0 DirFiltered=Clino_Filtered
MMVII ReportGCP AllImClino.xml Clino_Filtered Resec_Clino 
MMVII  OriBundleAdj  AllImClino.xml  Resec_Clino  BA_Clino_A  GCPDir=Clino_Filtered  GCPW=[1,1,0.5]  PPFzCal=.* BRDirIn=Rig_311_C BRW=[1e-2,1e-5] BRW_Rat=[1e-1,1e-4]


   #---------------------------------------------------------------------------
   #-------- Extract un-coded target and compensate again ---------------------
   #---------------------------------------------------------------------------

MMVII CodedTargetCompleteUncoded AllImClino.xml BA_Clino_A 1.0 InPointsMeasure=Clino OutPointsMeasure=ClinoCompl
MMVII  OriBundleAdj  AllImClino.xml  BA_Clino_A BA_Clino_B   GCPDir=ClinoCompl  GCPW=[1,1,0.5]  PPFzCal=.* BRDirIn=Rig_311_C BRW=[1e-2,1e-5] BRW_Rat=[1e-1,1e-4]
MMVII ReportGCP AllImClino.xml ClinoCompl  BA_Clino_B 


   #---------------------------------------------------------------------------
   #-------- Compute initial value of clinometers -----------------------------
   #---------------------------------------------------------------------------

MMVII ClinoInit Data-Input/ClinoMeasures.txt [949_,.JPG] [1,0] BA_Clino_B  Rel12="i-kj"
MMVII ClinoInit Data-Input/ClinoMeasures.txt [949_,.JPG] [0,1] BA_Clino_B  Rel12="ik-j"

###################################################################################
###################################################################################
########                                                                     ######
########                   LINE EXTRACTION                                   ######
########                                                                     ######
###################################################################################
###################################################################################

MMVII ExtractLine AllImFil.xml true InOri=BA_311_C ShowSteps=true






