set -e
# ====================================================================
#  Done in project with full data to select 20 images
# ====================================================================
#
# To create list of files in initial folder :
#    MMVII EditSet ImAerien.xml =  21FD244720x0001.*tif
#
# To filter the set of tie points :
#
#   MMVII ImportTiePMul all_liaisons.mes NIXY Vexcell NumL0=1 'PatIm=[.*,$&.tif]' NumByConseq=1 ImFilter=ImAerien.xml
#    
#


# indicate the meta data image that are not in xif file (here no file, no xif ;-)
#  Data comes from files s07_UC_Eagle_M3_120.xml, s08_UC_Eagle_M3_120.xml
#  Dont know the association Camera/Image, but as they are identic 
MMVII  EditCalcMTDI  Std PPPix ImTest=21FD244720x00015_00614.tif Modif=[21FD2447.*,[13230,8502],0] Save=1
MMVII  EditCalcMTDI  Std FocalPix ImTest=21FD244720x00015_00614.tif Modif=[21FD2447.*,30975,0] Save=1
MMVII  EditCalcMTDI  Std ModelCam ImTest=21FD244720x00015_00614.tif Modif=[21FD2447.*,"UltraCam Eagle Mark 3",0] Save=1
    #  NOTE : this focal in mm will not be used in photogrammety, as focal in pixel has priority, but it is required in identofier creation
MMVII  EditCalcMTDI  Std Focalmm ImTest=21FD244720x00015_00614.tif Modif=[21FD2447.*,123.9,0] Save=1
#  Dont use the "cylindric systematism"MMVII  
#  MMVII  EditCalcMTDI  Std AdditionalName  ImTest=Vol5_21FD8020x00001_01397.tif  Modif=["(Vol[0-9]+).*","\$1",0] Save=1


# Import Calibration & Orientations
MMVII OriCreateCalib ImAerien.xml  CalibInit Degree=[3,1,1]
MMVII  ImportOri  ExternalData/trajectograhie.opk   NXYZWPK  CalibInit  InitL93Up   NumL0=5 ChgN=[".*","\$&.tif"]   "AngU=degree" "KIsUp=true" FilterImIn=ImAerien.xml SysCo=Lambert93

# Import Tie Poins
MMVII ImportTiePMul ExternalData/Filtered_all_liaisons.mes NIXY Vexcell NumL0=0 'PatIm=[.*,$&.tif]' NumByConseq=1
# Quick test on bundle adj to see if orient are likely to be correctly imported
MMVII   OriBundleAdj ImAerien.xml InitL93Up Test TPDir=Vexcell TiePWeight=[1,1]  PatFzCenters=.*


# Import GCP
MMVII ImportGCP  ExternalData/Filtered_Terrain.APP 'NIXYZ'  AerRTL  ChSys=[Lambert93,RTL]
MMVII ImportMesImGCP  ExternalData/Filtered_Terrain.MES NIXY AerRTL  'PatIm=[.*,$&.tif]' 
MMVII OriChSysCo ImAerien.xml RTL  InitL93Up RTLD0


MMVII   OriBundleAdj ImAerien.xml RTLD0 Test TPDir=Vexcell TiePWeight=[1,1]  PatFzCenters=.* PPFzCal='[Kbp].*' GCPDir=AerRTL GCPW=[0,0.1]



#  Put everting in local tangent system
MMVII OriChSysCo ImAerien.xml RTL  InitL93Up RTLD0
MMVII OriChSysCo ImAerien.xml RTL  InitL93Up RTLD1
MMVII OriChSysCo ImAerien.xml RTL  InitL93Up RTLD2





