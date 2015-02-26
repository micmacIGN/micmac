#!/bin/bash

echo "Avant de lancer ce script, vérifier que micmac a été téléchargé et que Qt et tiff2rgba sont installés"

REP_INTERF=`echo $0 | gawk -v ref=${PWD} '/^\//;!/^\// {print ref "/" $1}' | gawk -F "/" '{$NF=""; print$0}' | replace " " "/"`
shift
echo ${REP_INTERF}

REP_MICMAC=`find ${REP_INTERF} -type d -name "micmac" -print;`
if [ ! ${REP_MICMAC} ];
then
	REP_MICMAC=`find ${REP_INTERF}.. -type d -name "micmac" -print;`
	shift
fi
if [ ! ${REP_MICMAC} ];
then
	REP_MICMAC=`find ${REP_INTERF}../.. -type d -name "micmac" -print;`
	shift
fi
echo ${REP_MICMAC}

#modification des sources de micmac
cp ${REP_INTERF}"modifmicmac/MakeAll" ${REP_MICMAC}"/MakeAll"
cp ${REP_INTERF}"modifmicmac/MakeISA" ${REP_MICMAC}"/MakeISA"
cp -r ${REP_INTERF}"modifmicmac/test_ISA0/include_isa" ${REP_MICMAC}"/applis/ELA/"
cp ${REP_INTERF}"modifmicmac/test_ISA0/TestIsabelle.cpp" ${REP_MICMAC}"/applis/ELA/TestIsabelle.cpp"
cp ${REP_INTERF}"modifmicmac/test_ISA0/ParamChantierPhotogram.xml" ${REP_MICMAC}"/include/XML_GEN/ParamChantierPhotogram.xml"
mv ${REP_MICMAC}"/include/XML_MicMac" ${REP_MICMAC}"/include/oldXML_MicMac"
cp -r ${REP_INTERF}"modifmicmac/XML_MicMac" ${REP_MICMAC}"/include/"
cp ${REP_INTERF}"modifmicmac/Bascule.cpp" ${REP_MICMAC}"/applis/uti_phgrm/Bascule.cpp"
cp ${REP_INTERF}"modifmicmac/Tarama.cpp" ${REP_MICMAC}"/applis/uti_phgrm/Tarama.cpp"

#compilation de micmac
cd  ${REP_MICMAC}
make all -f MakeAll -j4

#modification du .pro
replace "../../..\/micmac/" ${REP_MICMAC}"/" <${REP_INTERF}interface.pro >${REP_INTERF}tempo
mv ${REP_INTERF}tempo ${REP_INTERF}interface.pro

#compilation de l'interface
cd  ${REP_INTERF}
qmake -makefile
make all

#lanceur
echo "cd " ${REP_INTERF} "retourligne./interfaceMicmac" | gawk -F "retourligne" '{print $1; print $2}' >~/Bureau/interfaceMicmac
chmod +x ~/Bureau/interfaceMicmac

#modification du .pro de filtrageNuage
replace "../../..\/micmac/" ${REP_MICMAC}"/" <${REP_INTERF}lib/filtrageNuagesrc/filtrageNuage.pro >${REP_INTERF}tempo
mv ${REP_INTERF}tempo ${REP_INTERF}lib/filtrageNuagesrc/filtrageNuage.pro

#compilation de filtrageNuage et déplacement
cd  ${REP_INTERF}lib/filtrageNuagesrc
qmake -makefile
make all -j4
mv ${REP_INTERF}lib/filtrageNuagesrc/filtrageNuage ${REP_INTERF}lib/filtrageNuage

~/Bureau/interfaceMicmac
