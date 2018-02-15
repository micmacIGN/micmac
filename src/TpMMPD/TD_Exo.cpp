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
#include "TpPPMD.h"

// mm3d ElDcraw -t 0  -d -T -4   IMG_1364-CL1.CR2 

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/


/*
   Par exemple :

       mm3d TestLib TD_Test Orientation-IMG_0016.CR2.xml AppuisTest-IMG_0016.CR2.xml
*/

int TD_Exo0(int argc,char ** argv)
{
    std::string aNameIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name of image"),
        LArgMain()  
    );

    cTD_Im aIm = cTD_Im::FromString(aNameIm);

    std::cout << "Bonjour , NameIm=" << aNameIm  
              << " Sz=" <<  aIm.Sz().x 
              << " : " <<  aIm.Sz().y << "\n";

    cTD_Im aImRes(aIm.Sz().x,aIm.Sz().y);

    double aVMax = (1<<16) -1;
    for (int aX=0 ; aX<aIm.Sz().x ; aX++)
    {
        for (int aY=0 ; aY<aIm.Sz().y ; aY++)
        {
            double aVal = aIm.GetVal(aX,aY);
            aImRes.SetVal(aX,aY,aVMax-aVal);
        }
    }
    aImRes.Save("Neg.tif");


    return EXIT_SUCCESS;
}


class  cAppli_MatchEpi
{
	public :
	   cAppli_MatchEpi(int argc,char ** argv);
	   
	   std::string mName1;
	   std::string mName2;
	   cTD_Im      mIm1;
	   cTD_Im      mIm2;
	   int         mSzW;
	   int         mIntPx;
	   Pt2di       mSz1;
	   
	   bool  WinInIm(cTD_Im ,Pt2di aPt);
	   double Dissim(Pt2di aP1,Pt2di aP2);
	   
	   cTD_Im  CalculPx();
};

cTD_Im  cAppli_MatchEpi::CalculPx()
{
	cTD_Im aImPx(mSz1.x,mSz1.y);
	for (int aX=0 ; aX<mSz1.x ; aX++)
	{
		for (int aY=0 ; aY<mSz1.y ; aY++)
		{
			double aBestScore = 1e9;
			int    aBestPx    = 0;
			for (int aPx = -mIntPx; aPx <= mIntPx ; aPx++)
			{
				double aDis = Dissim(Pt2di(aX,aY),Pt2di(aX+aPx,aY));
				if (aDis < aBestScore)
				{
					aBestScore = aDis;
					aBestPx = aPx;
				}
			}
			aImPx.SetVal(aX,aY,aBestPx);
		}
	}
	return aImPx;
}

bool  cAppli_MatchEpi::WinInIm(cTD_Im aIm,Pt2di aPt)
{
	return 	   (aPt.x-mSzW>=0) 
			&& (aPt.y-mSzW>=0)
	        && (aPt.x+mSzW<aIm.Sz().x) 
	        && (aPt.y+mSzW<aIm.Sz().y);
}

double cAppli_MatchEpi::Dissim(Pt2di aP1,Pt2di aP2)
{
	if (!  (WinInIm(mIm1,aP1) && WinInIm(mIm2,aP2)))
	   return 1e10;
	   
	double aSomDif=0.0;
	for (int aDx=-mSzW ; aDx<=mSzW ; aDx++)
	{
		for (int aDy=-mSzW ; aDy<=mSzW ; aDy++)
		{
			Pt2di aDec(aDx,aDy);
			double aV1 = mIm1.GetVal(aP1+aDec);
			double aV2 = mIm2.GetVal(aP2+aDec);
			aSomDif += std::abs(aV1-aV2);
		}
	}
	return aSomDif;
}


cAppli_MatchEpi::cAppli_MatchEpi(int argc,char ** argv) :
   mIm1 (1,1),
   mIm2 (1,1),
   mSzW (3),
   mIntPx (50)
{

	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mName1,"Name image 1")
                    << EAMC(mName2,"Name image 2"),
        LArgMain()  
    );
    
    mIm1 = cTD_Im::FromString(mName1);
    mIm2 = cTD_Im::FromString(mName2);
    mSz1 = mIm1.Sz();
    std::cout << "Bonjour je suis une cAppli_MatchEpi pour " 
               << mName1 << mIm1.Sz() << " et " << mName2 << mIm2.Sz() << "\n";
     
    cTD_Im  aImPx= CalculPx();
    aImPx.Save("Px.tif");
}

int TD_Exo1(int argc,char ** argv)
{
	cAppli_MatchEpi anAppli(argc,argv);
    return EXIT_SUCCESS;
}

int TD_Exo2(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo3(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo4(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo5(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo6(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo7(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo8(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo9(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
