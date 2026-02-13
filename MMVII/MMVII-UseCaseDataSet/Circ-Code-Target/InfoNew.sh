set -e
set -x
#  GENERATE TARGET SPECIFICATION
MMVII  CodedTargetGenerateEncoding CERN 14
MMVII  CodedTargetGenerate  CERN_Nbb14_Freq14_Hamm1_Run1000_1000_SpecEncoding.xml

# GENERATE META DATA 
#   As camera is knonw, once we set focal length to 24mm, MMVII will be abel to have an approximate calibration
MMVII EditCalcMTDI Std ModelCam ImTest=043_0005_Scaled.tif  Modif=[.*_Scaled.tif,"NIKON D5600",0] Save=1
MMVII EditCalcMTDI Std Focalmm ImTest=043_0005_Scaled.tif  Modif=[.*_Scaled.tif,24,0] Save=1
MMVII EditCalcMTDI Std AdditionalName ImTest=043_0005_Scaled.tif  Modif=["(.*)_.*_.*","\$1",0] Save=1

#  CREATE an initial calibration with default param
MMVII  OriCreateCalib ".*tif" CalibInit

#  Import GCP , we fix a coordinate system "Pannel", purely local, mainly for documentation
#
MMVII  ImportGCP  Data-Aux/Positions-3D-14bit_lookup.txt NXYZBla Test NbDigName=3 ChSys=[LocalPannel]
MMVII  CodedTargetCircExtract ".*_Scaled.tif" CERN_Nbb14_Freq14_Hamm1_Run1000_1000_SpecEncoding_FullSpecif.xml DiamMin=8  OutObjMesInstr=Test

#  pose estimation init : resection + bundle
MMVII OriPoseEstimSpaceResection .*tif Test Test CalibInit Resec ThrRej=10  DirFiltered=Filt
MMVII  OriBundleAdj .*tif Resec BA GCP2D=[[Filt,1]] GCP3D=[[Test,1]]

#  research uncoded target + new bundle
MMVII CodedTargetCompleteUncoded .*_Scaled.tif BA 1.0 Test Test Completed ThRay=[1.05,4.7,5.3]
MMVII  OriBundleAdj .*tif  BA BA2 GCP2D=[[Completed,1,1,5]] GCP3D=[[Test,1]]

#   Generate a report on GCP quality
MMVII  ReportGCP .*tif Test Completed BA2

#  compute an initial value of the block
MMVII BlockCamInit .*tif BA2 "(.*)_(.*)_Scaled.tif" [1,2]  '[(.*)@(.*),$1_$2_Scaled.tif,@]' Rig

# make a compensation with rigid block 
MMVII  OriBundleAdj .*tif  BA2 BA3 GCP2D=[[Completed,1,1,5]] GCP3D=[[Test,1]] BRDirIn=Rig BRW=[1e-2,1e-5] 

# Idem but show uncertainty 
#  MMVII  OriBundleAdj .*tif  BA2 BA3 GCP2D=[[Completed,1,1,5]] GCP3D=[[Test,1]] BRDirIn=Rig BRW=[1e-2,1e-5] UC_UK="[.*,.*,(F|PP.*),1]"





