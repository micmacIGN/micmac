set -e 

MMVII EditSet AllIm.xml +=  ImAerien.xml
MMVII EditSet AllIm.xml +=  ImPNEO.xml 

MMVII TiePConvert AllIm.xml MMV1 Mixt  UpSByIm=ScaleImage.xml


#  To TEST
#  MMVII OriBundleAdj  AllIm.xml RTLD0  Test  TPDir=Mixt TiePWeight=[1,1]

MMVII OriBundleAdj  AllIm.xml RTLD0  Test   GCP2D=[[ORGI,1],[AerRTL,0.1]] GCP3D=[[RTLSat,1],[AerRTL,0.1]] TPDir=Mixt TiePWeight=[1,1]  AddTieP=[[ORGI,1,1],[Vexcell,1,1]] RefOri=[RTLD0,0.5] PPFzCal='[pbK].*'

MMVII OriBundleAdj  AllIm.xml RTLD0  Test   GCP2D=[[ORGI,1],[AerRTL,0.1]] GCP3D=[[RTLSat,1],[AerRTL,0.1]] TPDir=Mixt TiePWeight=[1,1]  AddTieP=[[ORGI,1,1],[Vexcell,1,1]] RefOri=[RTLD0,0.5] PPFzCal='.*'

