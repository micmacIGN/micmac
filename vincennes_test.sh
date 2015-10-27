#!/bin/sh

BIN_DIR=$1
CHANT_DIR=$2

"$BIN_DIR/Tapioca" All  "$CHANT_DIR/Calib-IMGP[0-9]{4}.JPG" 1000
"$BIN_DIR/Tapioca" Line "$CHANT_DIR/Face1-IMGP[0-9]{4}.JPG" 1000 5
"$BIN_DIR/Tapioca" Line "$CHANT_DIR/Face2-IMGP[0-9]{4}.JPG" 1000 5
"$BIN_DIR/Tapioca" All  "$CHANT_DIR/((Lnk12-IMGP[0-9]{4})|(Face1-IMGP529[0-9])|(Face2-IMGP531[0-9])).JPG" 1000

"$BIN_DIR/Tapas" RadialStd "$CHANT_DIR/Calib-IMGP[0-9]{4}.JPG" Out=Calib
"$BIN_DIR/Tapas" RadialStd "$CHANT_DIR/(Face1|Face2|Lnk12)-IMGP[0-9]{4}.JPG" Out=All InCal=Calib

"$BIN_DIR/AperiCloud" "$CHANT_DIR/(Face|Lnk).*JPG" All Out=AllCam.ply

"$BIN_DIR/GCPBascule"  "$CHANT_DIR/(Face1|Face2|Lnk12)-IMGP[0-9]{4}.JPG" All Ground Mesure-TestApInit-3D.xml Mesure-TestApInit.xml

"$BIN_DIR/RepLocBascule"  "$CHANT_DIR/(Face1)-IMGP[0-9]{4}.JPG" Ground  MesureBascFace1.xml Ortho-Cyl1.xml PostPlan=_MasqPlanFace1 OrthoCyl=true
"$BIN_DIR/Tarama"  "$CHANT_DIR/(Face1)-IMGP[0-9]{4}.JPG" Ground  Repere=Ortho-Cyl1.xml  Out=TA-OC-F1 Zoom=4
"$BIN_DIR/Malt" Ortho  "$CHANT_DIR/(Face1)-IMGP[0-9]{4}.JPG"  Ground  Repere=Ortho-Cyl1.xml  SzW=1 ZoomF=1  DirMEC=Malt-OC-F1 DirTA=TA-OC-F1
"$BIN_DIR/Tawny" "$CHANT_DIR/Ortho-UnAnam-Malt-OC-F1/ "
"$BIN_DIR/Nuage2Ply" "$CHANT_DIR/Malt-OC-F1/NuageImProf_Malt-Ortho-UnAnam_Etape_1.xml" Attr="$CHANT_DIR/Ortho-UnAnam-Malt-OC-F1/Ortho-Eg-Test-Redr.tif" Scale=3

"$BIN_DIR/RepLocBascule"  "$CHANT_DIR/(Face2)-IMGP[0-9]{4}.JPG" Ground  MesureBascFace2.xml Ortho-Cyl2.xml PostPlan=_MasqPlanFace2 OrthoCyl=true
"$BIN_DIR/Tarama"  "$CHANT_DIR/(Face2)-IMGP[0-9]{4}.JPG" Ground  Repere=Ortho-Cyl2.xml  Out=TA-OC-F2 Zoom=4
"$BIN_DIR/Malt" Ortho  "$CHANT_DIR/(Face2)-IMGP[0-9]{4}.JPG"  Ground  Repere=Ortho-Cyl2.xml  SzW=1 ZoomF=1  DirMEC=Malt-OC-F2 DirTA=TA-OC-F2 NbVI=2
"$BIN_DIR/Tawny" "$CHANT_DIR/Ortho-UnAnam-Malt-OC-F2/ "
"$BIN_DIR/Tawny" "$CHANT_DIR/Ortho-UnAnam-Malt-OC-F2/ " DEq=0
"$BIN_DIR/Nuage2Ply" "$CHANT_DIR/Malt-OC-F2/NuageImProf_Malt-Ortho-UnAnam_Etape_1.xml" Attr="$CHANT_DIR/Ortho-UnAnam-Malt-OC-F2/Ortho-Eg-Test-Redr.tif" Scale=3

"$BIN_DIR/SBGlobBascule" "$CHANT_DIR/(Face1|Face2|Lnk12)-IMGP[0-9]{4}.JPG" All MesureBascFace1.xml  Glob  PostPlan=_MasqPlanFace1  DistFS=2.0 Rep=ij
