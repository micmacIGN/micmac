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

#include "TpPPMD.h"

const float Beaucoup=1e20;

bool  VignetteInImage(int aSzW,const cTD_Im & aIm1,const Pt2di & aP1)
{
	Pt2di aSzIm = aIm1.Sz();
	
	return      (aP1.x >= aSzW)
	          && (aP1.x < aSzIm.x - aSzW)
	          && (aP1.y >= aSzW)
			  && (aP1.y < aSzIm.y - aSzW);
/*
	aIm1.Ok(aP1.x-aSzW,aP1.y)
	        &&  aIm1.Ok(aP1.x+aSzW,aP1.y)
	        &&  aIm1.Ok(aP1.x,aP1.y-aSzW)
	        &&  aIm1.Ok(aP1.x,aP1.y+);
*/

}

float SimilByCorrel
      (
         int aSzW,
         const cTD_Im & aIm1,const Pt2di & aP1,
         const cTD_Im & aIm2,const Pt2di & aP2
      )
{
    if (! 	VignetteInImage(aSzW,aIm1,aP1)) return Beaucoup;
    if (! 	VignetteInImage(aSzW,aIm2,aP2)) return Beaucoup;

	RMat_Inertie aMat;
    
    Pt2di aDP;
    for (aDP.x= -aSzW ; aDP.x<= aSzW ; aDP.x++)
    {
        for (aDP.y= -aSzW ; aDP.y<= aSzW ; aDP.y++)
        {
			float aV1 = aIm1.GetVal(aP1+aDP);
			float aV2 = aIm2.GetVal(aP2+aDP);
			aMat.add_pt_en_place(aV1,aV2,1.0);
		}
	}

	return 1-aMat.correlation();
}


float SimilByDif
      (
         int aSzW,
         const cTD_Im & aIm1,const Pt2di & aP1,
         const cTD_Im & aIm2,const Pt2di & aP2
      )
{
    if (! 	VignetteInImage(aSzW,aIm1,aP1)) return Beaucoup;
    if (! 	VignetteInImage(aSzW,aIm2,aP2)) return Beaucoup;

    float aSom = 0;
    
    Pt2di aDP;
    for (aDP.x= -aSzW ; aDP.x<= aSzW ; aDP.x++)
        for (aDP.y= -aSzW ; aDP.y<= aSzW ; aDP.y++)
			aSom += fabs(aIm1.GetVal(aP1+aDP)-aIm2.GetVal(aP2+aDP));

	return aSom;
}

// +Pt2di(3,4)

int  TD_Match1_main(int argc,char ** argv)
{

    std::string aNameI1,aNameI2;
    int aDeltaPax=100;
    int aSzW = 5;
    
     ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Im1")
					<< EAMC(aNameI2,"Name Im2"),
        LArgMain()  << EAM(aDeltaPax,"DPax",true,"Delta paralax")
                    << EAM(aSzW,"SzW",true,"Size of Window, Def=5")
    );
    
    // on charger nos deux images
    // image 1
    cTD_Im aI1 = cTD_Im::FromString(aNameI1);
    // image 2
    cTD_Im aI2 = cTD_Im::FromString(aNameI2);
    
    //dimension de nos images, sera utile pour nos boucles
    Pt2di aSz = aI1.Sz();
    
    // on crée un image pour stocker le résultat de la corrélation 
    cTD_Im aICorelMin = cTD_Im(aSz.x, aSz.y);
     // on crée la carte de profondeur
    cTD_Im aIProf = cTD_Im(aSz.x, aSz.y);
    
    // boucle sur tout les pixels de l'image 1
    Pt2di aP;
    for (aP.x=0; aP.x < aSz.x ; aP.x++)
    {
		std::cout << "Reste " << aSz.x-aP.x << "\n";
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
		    float aDiffMin = Beaucoup;
		    int aPaxOpt=0;
		    Pt2di aPPax(0,0);
		    for ( aPPax.x = -aDeltaPax ; aPPax.x<=aDeltaPax ; aPPax.x++)
		    {
				float aDiff =  SimilByDif(aSzW,aI1,aP,aI2,aP+aPPax);
				if  (aDiff < aDiffMin)
				{
					aDiffMin = aDiff;
					aPaxOpt = aPPax.x;
				}
			}
			aIProf.SetVal(aP.x,aP.y,aPaxOpt);
		}
	}
		
    aIProf.Save("CartePax.tif");
    
	return EXIT_SUCCESS;
}

int  TD_Match2_main(int argc,char ** argv)
{

    std::string aNameI1,aNameI2;
    int aDeltaPax=100;
    int aSzW = 5;
    
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Im1")
					<< EAMC(aNameI2,"Name Im2"),
        LArgMain()  << EAM(aDeltaPax,"DPax",true,"Delta paralax")
                    << EAM(aSzW,"SzW",true,"Size of Window, Def=5")
    );
    
       // image 1
    cTD_Im aI1 = cTD_Im::FromString(aNameI1);
    // image 2
    cTD_Im aI2 = cTD_Im::FromString(aNameI2);
    
    //dimension de nos images, sera utile pour nos boucles
    Pt2di aSz = aI1.Sz();
    
    cTD_Im aIBestScore(aSz.x,aSz.y);
    cTD_Im aIBestPax(aSz.x,aSz.y);
    
    Pt2di aP;
    for (aP.x=0; aP.x < aSz.x ; aP.x++)
    {
		for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
		{
		   aIBestScore.SetVal(aP.x,aP.y,Beaucoup);
		   aIBestPax.SetVal(aP.x,aP.y,sin((float)aP.x)*30*sin((float)aP.y));
		}
	}
    
    Pt2di aPPax;
    for ( aPPax.x = -aDeltaPax ; aPPax.x<=aDeltaPax ; aPPax.x++)
	{
		std::cout << "Pax= " << aPPax.x << "\n";
		
	// Calculer images des valeurs absolue des difference trans
		cTD_Im aImDif(aSz.x,aSz.y);

		for (aP.x=0; aP.x < aSz.x ; aP.x++)
		{
			for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
			{
				Pt2di aPTr = aP + aPPax;
				float aDif = 256;
				if (aI2.Ok(aPTr.x,aPTr.y))
				{
					 aDif = aI1.GetVal(aP) - aI2.GetVal(aPTr);
				}
				aImDif.SetVal(aP.x,aP.y,std::fabs(aDif));
			}
		}
		

		 //  Calculer l'image moyenne
		 
		 cTD_Im aImDifMoy = aImDif.ImageMoy(aSzW,1);

		 // Mettre a jour aIBestScore et aIBestPax
		 
		 
		for (aP.x=0; aP.x < aSz.x ; aP.x++)
		{
			for (aP.y=0 ; aP.y < aSz.y ; aP.y++)
			{
				float aDif =aImDifMoy.GetVal(aP);
				if (aDif<aIBestScore.GetVal(aP))
				{
					 aIBestScore.SetVal(aP.x,aP.y,aDif);
					 aIBestPax.SetVal(aP.x,aP.y,aPPax.x);
				}
			}
		}
	}
	aIBestPax.Save("CartePax2.tif");
	
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