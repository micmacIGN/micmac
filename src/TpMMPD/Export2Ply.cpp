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

class  cReadPts : public cReadObject
{
    public :
        cReadPts(char aComCar,const std::string & aFormat) :
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

int Export2Ply_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,3);

    std::string NameFile,Out;
    bool Help;
    eTypeFichierApp aType=eAppEgels;

    std::string aStrType;
    if (argc >=2)
    {
        aStrType = argv[1];
        StdReadEnum(Help,aType,argv[1],eNbTypeApp,true);

    }

    std::string aStrChSys;
    double aMul = 1.0;
    bool   aMulIncAlso = true;
    Pt3di aCoul(0,0,255);
    Pt3di aCoulLP(0,0,0);
    double aRay=0;
    int aNbPts=5;
    int aScale=1;
    int aBin=1;
    int aDiffColor=0;
    Pt3dr aOffset(0,0,0);
	bool aGpsFile=false;
	bool aSFP=false;
	bool aShow=false;
	
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aStrType,"Format specification", eSAM_None, ListOfVal(eNbTypeApp))
                    << EAMC(NameFile,"Name File of Points Coordinates", eSAM_IsExistFile),
        LArgMain()  << EAM(aRay,"Ray",false,"Plot a sphere per point")
                    << EAM(aNbPts,"NbPts",true,"Number of Pts / direc (Def=5, give 1000 points) only with Ray > 0")
                    << EAM(aScale,"Scale",true,"Scaling factor")
                    << EAM(aCoul,"FixColor",true,"Fix the color of points")
                    << EAM(aCoulLP,"LastPtColor",true,"Change color only for last point")
                    << EAM(aDiffColor,"ChangeColor",false,"Change the color each number of points : not with FixColor")
                    << EAM(Out,"Out",true, "Default value is NameFile.ply")
                    << EAM(aBin,"Bin",true,"Generate Binary or Ascii (Def=1, Binary)")
                    << EAM(aOffset,"OffSet",true,"Subtract an offset to all points")
                    << EAM(aGpsFile,"GpsXML",true,"GPS xml input file; Def=false")
                    << EAM(aSFP,"ShiftBFP",true,"Shift by substructing frist point to all ; Def=false")
                    << EAM(aShow,"Show",true,"Show points ; Def=false")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    char * aLine;
    int aCpt=0;
    std::vector<Pt3di> aVCol;
    std::vector<Pt3dr> aPoints;
    std::vector<Pt3dr> aVInc;
    std::vector<std::string> aVName;
    std::vector<int> aVQ;

    if (!MMVisualMode)
    {
        std::string aFormat;
        char        aCom;
        if (aType==eAppEgels)
        {
             aFormat = "N S X Y Z";
             aCom    = '#';
        }
        else if (aType==eAppGeoCub)
        {
             aFormat = "N X Y Z";
             aCom    = '%';
        }
        else if (aType==eAppInFile)
        {
           bool Ok = cReadObject::ReadFormat(aCom,aFormat,NameFile,true);
           ELISE_ASSERT(Ok,"File do not begin by format specification");
        }
        else if (aType==eAppXML)
        {
             aFormat = "00000";
             aCom    = '0';
           // bool Ok = cReadObject::ReadFormat(aCom,aFormat,aFilePtsIn,true);
           // ELISE_ASSERT(Ok,"File do not begin by format specification");
        }
        else
        {
            bool Ok = cReadObject::ReadFormat(aCom,aFormat,aStrType,false);
            ELISE_ASSERT(Ok,"Arg0 is not a valid format specif");
        }

        if (Out=="")
        {
           Out =StdPrefixGen(NameFile) + ".ply";
        }


        if (aType==eAppXML)
        {
				if (Out==NameFile)
					Out = "GCPOut_"+NameFile;
                if(aGpsFile)
                {
					cDicoGpsFlottant aDico =  StdGetFromPCP(NameFile,DicoGpsFlottant);
					for(auto IT=aDico.OneGpsDGF().begin();IT!=aDico.OneGpsDGF().end();IT++)
					{
						aPoints.push_back(IT->Pt());
                        if(IT->TagPt() == 1)
                        {
                            aVCol.push_back(aCoul);
                        }
                        else
                        {
                            Pt3di aCoulMP(255,0,0);
                            aVCol.push_back(aCoulMP);
                        }
						aVInc.push_back(IT->Incertitude());
                        aVName.push_back(IT->NamePt());
                        aVQ.push_back(IT->TagPt());
					}
				}
				else
				{
					cDicoAppuisFlottant aD = StdGetObjFromFile<cDicoAppuisFlottant>
										 (
											  NameFile,
											  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
											  "DicoAppuisFlottant",
											  "DicoAppuisFlottant"
										 );

					for
					(
						std::list<cOneAppuisDAF>::iterator itA=aD.OneAppuisDAF().begin();
						itA!=aD.OneAppuisDAF().end();
						itA++
					)
					{
					  aPoints.push_back(itA->Pt());
					  aVCol.push_back(aCoul);
					  aVInc.push_back(itA->Incertitude());
					  aVName.push_back(itA->NamePt());
					}
				}
        }
        else
        {
            std::cout << "Comment=[" << aCom<<"]\n";
            std::cout << "Format=[" << aFormat<<"]\n";
            cReadPts aReadApp(aCom,aFormat);
            ELISE_fp aFIn(NameFile.c_str(),ELISE_fp::READ);
            while ((aLine = aFIn.std_fgets()))
            {
                 if (aReadApp.Decode(aLine))
                 {
                    aPoints.push_back(aReadApp.mPt);
                    aVCol.push_back(aCoul);
                    double  aInc = aReadApp.GetDef(aReadApp.mInc,1);
                    aVInc.push_back(aReadApp.GetDef(aReadApp.mInc3,aInc));
                    aVName.push_back(aReadApp.mName);
                }
                aCpt ++;
             }
            aFIn.close();
        }

        if (aMul!=1.0)
        {
            for (int aK=0 ; aK<int(aPoints.size()) ; aK++)
            {
                 aPoints[aK] = aPoints[aK] * aMul;
                 if (aMulIncAlso)
                     aVInc[aK] = aVInc[aK] * aMul;
            }
        }

    }

    ELISE_ASSERT((aRay >= 0) ,"The value of Ray should be positive");

    ELISE_ASSERT((aNbPts >=0) , "The value of NbPts should be positive");

    if(aNbPts != 5 && aRay==0)
    {
        ELISE_ASSERT(aRay > 0, "Can't specify number of points without Ray of sphere > 0");
    }

    if(aDiffColor !=0 && (aCoul.x != 255 || aCoul.y !=0 || aCoul.z !=0))
    {
        ELISE_ASSERT(aDiffColor ==0, "Can't fix and change color at the same time");
    }

    ELISE_ASSERT(aDiffColor < (int) aPoints.size(), "Can't be superior to number of points in input");

    if(aOffset.x != 0 || aOffset.y !=0 || aOffset.z != 0)
    {
		for(unsigned int aP=0; aP<aPoints.size(); aP++)
		{
            aPoints.at(aP).x = aPoints.at(aP).x - aOffset.x;
            aPoints.at(aP).y = aPoints.at(aP).y - aOffset.y;
            aPoints.at(aP).z = aPoints.at(aP).z - aOffset.z;
		}
	}
	
	if(aSFP)
	{
		Pt3dr aFP = aPoints.at(0);
		for(unsigned int aP=0; aP<aPoints.size(); aP++)
		{
			aPoints.at(aP).x = aPoints.at(aP).x - aFP.x;
			aPoints.at(aP).y = aPoints.at(aP).y - aFP.y;
			aPoints.at(aP).z = aPoints.at(aP).z - aFP.z;
		}
	}
    //if we do not want to keep all points
    if((int)aScale != 1)
    {
        int aIndice=0;
        int aSizeInit = (int)aPoints.size();
        std::vector<Pt3dr> cPoints;
        std::vector<Pt3di> cVCol;
        while(aIndice*aScale < aSizeInit)
        {
            cPoints.push_back(aPoints.at(aIndice*aScale));
            cVCol.push_back(aVCol.at(aIndice*aScale));
            aIndice++;
        }
        aPoints.clear();
        aVCol.clear();
        aPoints = cPoints;
        aVCol = cVCol;
    }

    //fix color of last point (if fix color is set, give the same color to last point)
    if(aCoul.x != 255 || aCoul.y !=0 || aCoul.z != 0)
    {
        if(aCoulLP.x == 255 && aCoulLP.y == 0 && aCoulLP.z == 0)
        {
            aVCol.at(aVCol.size()-1) = aCoul;
        }
    }
    else
    {
        aVCol.at(aVCol.size()-1) = aCoulLP;
    }

    std::vector<Pt3dr> aVpt;
    std::vector<Pt3di> aVptCol;
    std::list<std::string> aVCom;
    std::vector<const cElNuage3DMaille *> aVNuage;

	if(aShow)
	{
		std::cout << "fixed:\n" << std::fixed;
		std::cout << aPoints.size() << std::endl;
		for(u_int i=0; i<aPoints.size(); i++)
			std::cout << aPoints.at(i) << std::endl;
	}

    //if we want to change color each "aDiffColor" time
    int aMinValue = 0;
    int aMaxValue = 255;
    if(aDiffColor != 0)
    {
        for(unsigned int aSameColor=0 ; aSameColor < aPoints.size() ; aSameColor+=aDiffColor)
        {
            int aRandR = aMinValue + (rand() % (int)(aMaxValue - aMinValue + 1));
            int aRandG = aMinValue + (rand() % (int)(aMaxValue - aMinValue + 1));
            int aRandB = aMinValue + (rand() % (int)(aMaxValue - aMinValue + 1));
            Pt3di aRandColor(aRandR,aRandG,aRandB);

            int aCompt=0;
            for (int aPtsSC=0; aPtsSC < aDiffColor ; aPtsSC++)
            {
                if( aPoints.size()-aSameColor < (unsigned int) aDiffColor)
                {
                    for (unsigned int aPtsL=0 ; aPtsL < aPoints.size()-aSameColor ; aPtsL++)
                    {
                        aVCol.at(aSameColor+aCompt) = aRandColor;
                    }
                    break;
                }
                else
                {
                    aVCol.at(aSameColor+aCompt) = aRandColor;
                }
                aCompt++;
            }
        }
    }

    // if we want a sphere per point
    if(aRay > 0)
    {
        for(unsigned int aNbrPts=0; aNbrPts<aPoints.size(); aNbrPts++)
        {
            for (int anX=-aNbPts; anX<=aNbPts ; anX++)
            {
                for (int anY=-aNbPts; anY<=aNbPts ; anY++)
                {
                    for (int aZ=-aNbPts; aZ<=aNbPts ; aZ++)
                    {
                        Pt3dr aP(anX,anY,aZ);
                        aP = aP * (aRay/aNbPts);
                        if (euclid(aP) <= aRay)
                        {
                            aVpt.push_back(aPoints.at(aNbrPts)+aP);
                            aVptCol.push_back(aVCol.at(aNbrPts));
                        }
                    }
                }
            }
        }

        cElNuage3DMaille::PlyPutFile
        (
          Out,
          aVCom,
          aVNuage,
          &aVpt,
          &aVptCol,
          true
        );
    }



    else
    {
        cElNuage3DMaille::PlyPutFile
    (
          Out,
          aVCom,
          aVNuage,
          &aPoints,
          &aVCol,
          true
    );
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
