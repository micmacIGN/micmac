set -e 

MMVII EditSet AllIm.xml +=  ImAerien.xml
MMVII EditSet AllIm.xml +=  ImPNEO.xml 

MMVII TiePConvert AllIm.xml MMV1 Mixt  UpSByIm=ScaleImage.xml


#  To TEST
#  MMVII OriBundleAdj  AllIm.xml RTLD0  Test  TPDir=Mixt TiePWeight=[1,1]

MMVII OriBundleAdj  AllIm.xml RTLD0  Test   AddGCPW=[[RTLSat,1,1],[AerRTL,0.1,0.1]]   AddTieP=[[Mixt,1,1],[ORGI,1,1],[Vexcell,1,1]] RefOri=[RTLD0,0.5] PPFzCal='[pbK].*'

MMVII OriBundleAdj  AllIm.xml RTLD0  Test   AddGCPW=[[RTLSat,1,1],[AerRTL,0.1,0.1]]   AddTieP=[[Mixt,1,1],[ORGI,1,1],[Vexcell,1,1]] RefOri=[RTLD0,0.5] PPFzCal='.*'

