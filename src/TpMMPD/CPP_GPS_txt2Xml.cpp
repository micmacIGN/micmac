/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "StdAfx.h"

class  cReadPosGps : public cReadObject
{
    public :
        cReadPosGps(char aComCar,const std::string & aFormat) :
               cReadObject(aComCar,aFormat,"S"),
               mPt(-1,-1,-1),
               mInc3(-1,-1,-1),
               mInc (-1)
        {
              AddString("N",&mName,true);
              AddPt3dr("XYZ",&mPt,true);
              AddDouble("Ix",&mInc3.x,false);
              AddDouble("Iy",&mInc3.y,false);
              AddDouble("Iz",&mInc3.z,false);
              AddDouble("I",&mInc,false);
        }

        std::string mName;
        Pt3dr       mPt;
        Pt3dr       mInc3;
        double      mInc;
};

//required : system of time : utc,gpst,jst,arb
//supported format : tow,hms
//internal use : tow + time (s) since origine
//check if a date is valid

int GPS_Txt2Xml_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,3);
    std::string aFilePtsIn,aFilePtsOut;
    eTypeFichierApp aType=eAppEgels;
    std::string aStrType;
	bool Help;
	if (argc >=2)
    {
        aStrType = argv[1];
        StdReadEnum(Help,aType,argv[1],eNbTypeApp,true);

    }
    std::string aStrChSys;
    Pt3dr aOffset(0,0,0);

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aStrType,"Format specification")
                      << EAMC(aFilePtsIn,"GPS input  File", eSAM_IsExistFile),
           LArgMain() << EAM(aFilePtsOut,"Out",true,"Xml Out File",eSAM_IsOutputFile)
                      << EAM(aStrChSys,"ChSys",true,"Change coordinate file")
                      << EAM(aOffset,"OffSet",true,"Subtract an offset to all points")
    );

    if (!MMVisualMode)
    {
        std::string aFormat;
        char        aCom;
        
        if (aType==eAppGeoCub)
        {
             aFormat = "N X Y Z";
             aCom    = '%';
        }
        
        //if format is not correct
        else
        {
            bool Ok = cReadObject::ReadFormat(aCom,aFormat,aStrType,false);
            ELISE_ASSERT(Ok,"Arg0 is not a valid format specif");
        }
		
		//change system coordinates
        cChSysCo * aCSC = 0;
        if (aStrChSys!="")
           aCSC = cChSysCo::Alloc(aStrChSys,"");
		
		//name output .xml file
        if (aFilePtsOut=="")
        {
            aFilePtsOut =StdPrefixGen(aFilePtsIn) + ".xml";
        }

        char * aLine;
        int aCpt=0;

        std::vector<Pt3dr> aVPts;
        std::vector<Pt3dr> aVInc;
        std::vector<std::string> aVName;
        
        std::cout << "Comment=[" << aCom<<"]\n";
        std::cout << "Format=[" << aFormat<<"]\n";
        
        cReadPosGps aReadPosGps(aCom,aFormat);
        ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);
        while ((aLine = aFIn.std_fgets()))
        {
			if (aReadPosGps.Decode(aLine))
            {
                aVPts.push_back(aReadPosGps.mPt);
                double  aInc = aReadPosGps.GetDef(aReadPosGps.mInc,1);
                aVInc.push_back(aReadPosGps.GetDef(aReadPosGps.mInc3,aInc));
                aVName.push_back(aReadPosGps.mName);
            }
            aCpt ++;
        }
        aFIn.close();


        if (aCSC!=0)
        {
            aVPts = aCSC->Src2Cibl(aVPts);
        }
        
        cDicoGpsFlottant  aDico;
	
		for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
		{
			cOneGpsDGF aOAD;
            aOAD.Pt() = aVPts[aKP] - aOffset;
			aOAD.Incertitude() = aVInc[aKP];
			aOAD.TagPt() = 1;
			aOAD.TimePt() = 0;

			aDico.OneGpsDGF().push_back(aOAD);
		}

		MakeFileXML(aDico,aFilePtsOut);

        return 0;
    }
    else
        return EXIT_SUCCESS;
}


int CalcTF_main(int argc,char ** argv)
{
	std::string aInputFile;
	bool aFilter=false;
	std::string aOut="";
	
	ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aInputFile,"GPS .xml input  File", eSAM_IsExistFile),
           LArgMain() << EAM(aFilter,"Filter",false,"Generate New GPS file and keep only fixed positions ; Def=false",eSAM_IsBool)
                      << EAM(aOut,"Out",false,"Name output file with only fixed positions ; Def=InputFile_Fixed.xml",eSAM_IsOutputFile)
    );
    
    //name output .xml file
    if (aOut=="")
    {
		aOut =StdPrefixGen(aInputFile) + "_Fixed" + ".xml";
    }
    
    //read the .xml input file
	cDicoGpsFlottant aFile =  StdGetFromPCP(aInputFile,DicoGpsFlottant);
	// std::list <cOneGpsDGF> & aVP = aFile.OneGpsDGF();
	
	int aCompQ1 = 0;
	cDicoGpsFlottant  aDico;
	
	// for(std::list<cOneGpsDGF>::iterator iT=aVP.begin(); iT!=aVP.end(); iT++)
	for(auto  iT=aFile.OneGpsDGF().begin(); iT!=aFile.OneGpsDGF().end(); iT++)
	{
		if(iT->TagPt() == 1)
		{
			cOneGpsDGF aOAD;
			
			aOAD.Pt() = iT->Pt();
			aOAD.Incertitude() = iT->Incertitude();
			aOAD.NamePt() = iT->NamePt();
			aOAD.TagPt() = iT->TagPt();
			aOAD.TimePt() = iT->TimePt();

			aDico.OneGpsDGF().push_back(aOAD);
			
			aCompQ1++;
		}
	}
	
	double aTF = double(aCompQ1)/aFile.OneGpsDGF().size();
	
	std::cout << " Taux Fixation = " <<  aTF*100 << " %" << std::endl;
	
	//generate file with only Q1
	if(aFilter)
	{
		MakeFileXML(aDico,aOut);
	}
    
	return EXIT_SUCCESS;
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
