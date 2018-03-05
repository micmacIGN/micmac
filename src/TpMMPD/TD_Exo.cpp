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
	   std::string mNameSave;
	   cTD_Im      mIm1;
	   cTD_Im      mIm2;
	   cTD_Im      mImMoy1;
	   cTD_Im      mImMoy2;
	   cTD_Im      mImSigma1;
	   cTD_Im      mImSigma2;
	   cTD_Im      mImScore;
	   int         mSzW;
	   int         mIntPx;
	   Pt2di       mSz1;
	   
	   cTD_Im   ImSigma(cTD_Im aIm,cTD_Im aImMoy);

	   // Indique si une fenetre centree en aPt est incluse
	   // dans l'image maitresse 
	   bool  WinInIm(cTD_Im ,Pt2di aPt);
	   
	   // Calcule la dissimilarite entre deux fenetre, centree en P1
	   // dans Im1 et centree en P2 dans Im2
	   double DissimDifSimple(Pt2di aP1,Pt2di aP2);
	   
	   double DissimNormParMoy(Pt2di aP1,Pt2di aP2);

	   
	   // Calcule une image qui pour chaque pixel memorise la paralaxe
	   // donnant la dissimilarite la plus faible
	   cTD_Im  CalculPx();
	   
	   // Fonction de calcul "rapide" 
	   cTD_Im  CalculRapidePx();
};

cTD_Im   cAppli_MatchEpi::ImSigma(cTD_Im aIm,cTD_Im aImMoy)
{
	Pt2di aSz = aIm.Sz();
	cTD_Im aImRes(aSz.x,aSz.y);
	
	// Dans image res on met le carre de l'image d'entree
	for (int aX=0 ; aX<aSz.x ; aX++)
	{
		for (int aY=0 ; aY<aSz.y ; aY++)
		{
			aImRes.SetVal(aX,aY,ElSquare(aIm.GetVal(aX,aY)));
		}
	}
	// On fait la moyenne des carres,
	aImRes = aImRes.ImageMoy(mSzW,1);
	// On utilise moyenne des carres et moyenne pour y mettre l'ecart type
	for (int aX=0 ; aX<aSz.x ; aX++)
	{
		for (int aY=0 ; aY<aSz.y ; aY++)
		{
			double aMoy = aImMoy.GetVal(aX,aY);
			double aMoyQuad = aImRes.GetVal(aX,aY);
			// Variance = Moyenne des carres moins carre de la moyenne
			double aVar = aMoyQuad - ElSquare(aMoy); 
			double aSigma = 0.0;
			if (aVar >=0)
			{ // Ecart Type = racine carre de la variance
				aSigma = sqrt(aVar);
			}
			else
			{
				std::cout << "? ? ? ! ! ! Var<0 : " << aVar << "\n";
			}
			aImRes.SetVal(aX,aY,aSigma);
		}
	}
	return aImRes;
}

cTD_Im  cAppli_MatchEpi::CalculRapidePx()
{
	// Image de resultat, superposable a l'image 1 ("maitresse")
	cTD_Im aImPx(mSz1.x,mSz1.y);
	// Image de meilleure similarite 
	cTD_Im aImBestSim(mSz1.x,mSz1.y);
	
	// Initialisation des similarite
	for (int aX=0 ; aX<mSz1.x ; aX++)
	{
		for (int aY=0 ; aY<mSz1.y ; aY++)
		{
			// Meilleure initialisee a "+ infini"
			aImBestSim.SetVal(aX,aY,1e9);
			aImPx.SetVal(aX,aY,0);
		}
	}
	
	// Parcours de toute les parallaxes possibles
	for (int aPx = -mIntPx; aPx <= mIntPx ; aPx++)
	{
		std::cout << "PX " << aPx << "\n";
		cTD_Im aImDis(mSz1.x,mSz1.y);
		// Calcul d'une image de difference decalee de aPx
		for (int aX=0 ; aX<mSz1.x ; aX++)
		{
			for (int aY=0 ; aY<mSz1.y ; aY++)
			{
				// Attention au debordement a cause de la paralaxe
				if ( mIm2.Ok(aX+aPx,aY))
				{
				    double aDif = mIm1.GetVal(aX,aY) - mIm2.GetVal(aX+aPx,aY);
				    aImDis.SetVal(aX,aY,std::abs(aDif));
				}
				else
				    aImDis.SetVal(aX,aY,1e9);
			}
		}
		// On moyenne cette image pour avoir la dissimilarite sommee sur
		// une fenetre
		aImDis = aImDis.ImageMoy(mSzW,1);
		// On met a jour la ou cette dissimilarite est meilleure que celle
		// obtenue jusque la
		for (int aX=0 ; aX<mSz1.x ; aX++)
		{
			for (int aY=0 ; aY<mSz1.y ; aY++)
			{
				double aDis = aImDis.GetVal(aX,aY);
				// si meilleur, mise a jour
				if (aDis < aImBestSim.GetVal(aX,aY))
				{
					aImBestSim.SetVal(aX,aY,aDis);
					aImPx.SetVal(aX,aY,aPx);
				}
			}
		}
	}
	return aImPx;
}



cTD_Im  cAppli_MatchEpi::CalculPx()
{
	// Image de resultat, superposable a l'image 1 ("maitresse")
	cTD_Im aImPx(mSz1.x,mSz1.y);
	// Parcourt l'image
	for (int aX=0 ; aX<mSz1.x ; aX++)
	{
		if ((aX%100) ==0) std::cout << "Reste " << (mSz1.x-aX) << "\n";
		for (int aY=0 ; aY<mSz1.y ; aY++)
		{
			// Pour un pixel donne de l'image 1, on recherche la
			// paralaxe donnant le meilleurs homologue
			double aBestScore = 1e9; // "+ l'infini"= valeur neutre pour un min
			int    aBestPx    = 0;
			for (int aPx = -mIntPx; aPx <= mIntPx ; aPx++)
			{
				double aDis = DissimNormParMoy(Pt2di(aX,aY),Pt2di(aX+aPx,aY));
				// Met a jour si meilleur homologue trouve
				if (aDis < aBestScore)
				{
					aBestScore = aDis;
					aBestPx = aPx;
				}
			}
			// memorise le resultat
			aImPx.SetVal(aX,aY,aBestPx);
			mImScore.SetVal(aX,aY,aBestScore);
		}
	}
	return aImPx;
}

bool  cAppli_MatchEpi::WinInIm(cTD_Im aIm,Pt2di aPt)
{
	//  Test si les 4 coins sont inclus 
	return 	   (aPt.x-mSzW>=0) 
			&& (aPt.y-mSzW>=0)
	        && (aPt.x+mSzW<aIm.Sz().x) 
	        && (aPt.y+mSzW<aIm.Sz().y);
}

double cAppli_MatchEpi::DissimNormParMoy(Pt2di aP1,Pt2di aP2)
{
	if (!  (WinInIm(mIm1,aP1) && WinInIm(mIm2,aP2)))
	   return 1e10;

	double aMoy1 = mImMoy1.GetVal(aP1);
	double aMoy2 = mImMoy2.GetVal(aP2);
	
	double aSigma1 = mImSigma1.GetVal(aP1);
	double aSigma2 = mImSigma2.GetVal(aP2);
    	  
	if ( (aSigma1==0)  || (aSigma2==0))
	   return 1e10;
	   
	// Accumulateur de la somme des difference 
	double aSomDif=0.0;
	double aSomScal = 0.0;
	// Parcourt la fenetre
	for (int aDx=-mSzW ; aDx<=mSzW ; aDx++)
	{
		for (int aDy=-mSzW ; aDy<=mSzW ; aDy++)
		{
			Pt2di aDec(aDx,aDy);
			double aV1 = (mIm1.GetVal(aP1+aDec) -aMoy1)/ aSigma1 ;
			double aV2 = (mIm2.GetVal(aP2+aDec) -aMoy2)/ aSigma2 ;
			// Met a jour la somme des differences
			aSomDif += ElSquare(aV1-aV2);
			aSomScal += aV1 * aV2;
		}
	}
	aSomDif /= ElSquare(1+2*mSzW);
	aSomScal /= ElSquare(1+2*mSzW);
	
	
	static double aMaxDif = -1e9;
	static double aMinDif =  1e9;
	if (aSomDif > aMaxDif)
	{
		aMaxDif = aSomDif;
		//std::cout << "INTERVAL DIFF " << aMinDif << " " << aMaxDif << "\n";
	}
	if (aSomDif< aMinDif)
	{
		aMinDif = aSomDif;
	//	std::cout << "INTERVAL DIFF " << aMinDif << " " << aMaxDif << "\n";
		
		// std::cout << aSomDif << " " << aSomScal << "\n";
	}
	return 1-aSomScal;
}
double cAppli_MatchEpi::DissimDifSimple(Pt2di aP1,Pt2di aP2)
{
	if (!  (WinInIm(mIm1,aP1) && WinInIm(mIm2,aP2)))
	   return 1e10;
	  
	// Accumulateur de la somme des difference 
	double aSomDif=0.0;
	// Parcourt la fenetre
	for (int aDx=-mSzW ; aDx<=mSzW ; aDx++)
	{
		for (int aDy=-mSzW ; aDy<=mSzW ; aDy++)
		{
			Pt2di aDec(aDx,aDy);
			double aV1 = mIm1.GetVal(aP1+aDec);
			double aV2 = mIm2.GetVal(aP2+aDec);
			// Met a jour la somme des differences
			aSomDif += std::abs(aV1-aV2);
		}
	}
	return aSomDif;
}


cAppli_MatchEpi::cAppli_MatchEpi(int argc,char ** argv) :
   mNameSave ("Px.tif"),
   mIm1 (1,1),
   mIm2 (1,1),
   mImMoy1 (1,1),
   mImMoy2 (1,1),
   mImSigma1 (1,1),
   mImSigma2 (1,1),
   mImScore  (1,1),   
   mSzW (3),
   mIntPx (50)
{

	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mName1,"Name image 1")
                    << EAMC(mName2,"Name image 2"),
        LArgMain()  << EAM (mSzW,"SzW",true,"Taille de la fenetre de correlation")
                    << EAM (mIntPx,"IntPx",true,"Intervalle de paralaxe explore")
                    << EAM (mNameSave,"Out",true,"Image de sortie (Def=Px.tif)")
                    
    );
    
    mIm1 = cTD_Im::FromString(mName1);
    mSz1 = mIm1.Sz();
    mIm2 = cTD_Im::FromString(mName2);
    
    mImMoy1 = mIm1.ImageMoy(mSzW,1);
    mImMoy2 = mIm2.ImageMoy(mSzW,1);
    
    mImSigma1 = ImSigma(mIm1,mImMoy1);
    mImSigma2 = ImSigma(mIm2,mImMoy2);
    mImScore = cTD_Im(mSz1.x,mSz1.y);
    
     mImMoy1.Save("Moy1.tif");
     mImSigma1.Save("Sigma1.tif");
    
    // Test de la fonction de moyenne
    /*
    for (int aK=1 ; aK<10 ; aK++)
    {
		int aSzW = aK*3;
		ElTimer aChrono;
		cTD_Im aImMoy = mIm1.ImageMoy(aSzW,1);
		std::cout << " Temps= " << aChrono.uval() << " for " << aSzW << "\n";
		aImMoy.Save("Moyenne-" + ToString(aSzW) + ".tif");
	}
	*/
    
   
    std::cout << "Bonjour je suis une cAppli_MatchEpi pour " 
               << mName1 << mIm1.Sz() << " et " << mName2 << mIm2.Sz() << "\n";
     
   //  CalculRapidePx();
     
    ElTimer aChrono;
    cTD_Im  aImPx= CalculPx();    
    double aT = aChrono.uval();
    double aRatioT = aT /(mSz1.x*mSz1.y * double(mIntPx) * pow(1+2*mSzW,2));
 
    std::cout << "Ratio Time " << aRatioT * 1e9 << " Time" << aT <<  "\n";
    aImPx.Save(mNameSave);
    mImScore.Save("Score.tif");
}

int TD_Exo1(int argc,char ** argv)
{
	cAppli_MatchEpi anAppli(argc,argv);
    return EXIT_SUCCESS;
}

int TD_Exo2(int argc,char ** argv)
{
	std::string aNamePx12;
	std::string aNamePx21;
	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNamePx12,"Name Paralaxe  1 vers 2")
                    << EAMC(aNamePx21,"Name Paralaxe  2 vers 1"),
        LArgMain()  
                    
    );
    
    cTD_Im  aPx12 = cTD_Im::FromString(aNamePx12);
    cTD_Im  aPx21 = cTD_Im::FromString(aNamePx21);
    
    // Lire la taille de Px12
    Pt2di aSz12 = aPx12.Sz();
    // Creer une image resultat de meme taille que Px12
    cTD_Im aImQual(aSz12.x,aSz12.y);
    // Parcourir l'image resultat, pour chaque pixel P:
    for (int aX1=0 ; aX1<aSz12.x ; aX1++)
    {
		for (int aY=0 ; aY<aSz12.y ; aY++)
		{
			int Qual = 255;
		 //   calcul  homologue X2 de X1 par Px12
			int aX2 = aX1 + round_ni(aPx12.GetVal(aX1,aY));
			if (aPx21.Ok(aX2,aY))
			{
		//        calcul X1R  homologue de X2 par Px12
				int aX1R = aX2 + aPx21.GetVal(aX2,aY);
		  //        la "qualite" c'est la distance entre X1 et X1R
				Qual = std::abs(aX1-aX1R);
			}
			aImQual.SetVal(aX1,aY,Qual);
		}
	}
	aImQual.Save("QualAR.tif");
    //  sauvegarder Px12
    
    
    return EXIT_SUCCESS;
}

int TD_Exo3(int argc,char ** argv)
{
	std::string aNamePx;
	int aSzW;
	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNamePx,"Name Paralaxe ")
                    << EAMC(aSzW,"Taille de la fenetre"),
        LArgMain()                      
    );
    
    // Lecture de l'image d'entree
    cTD_Im  aPxIn = cTD_Im::FromString(aNamePx);
    Pt2di aSz = aPxIn.Sz();
    // Creation de l'image de sortie
    cTD_Im  aPxOut(aSz.x,aSz.y);
    
    // Parcourt les points qui contiennent la fenetre
    for (int aX=aSzW ; aX<aSz.x-aSzW ; aX++)
    {
		for (int aY=aSzW ; aY<aSz.y-aSzW ; aY++)
		{
			// creer le vecteur
			std::vector<double> aVec;
			// empiler les valeur PxIn de la fenetre centree sur X,Y
			for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
			{
				for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
				{
					Pt2di aP(aX+aDx,aY+aDy);
					double aPx = aPxIn.GetVal(aP);
					aVec.push_back(aPx);
				}
			}
			// calculer la mediane
			std::sort(aVec.begin(),aVec.end());
			int aNbV = aVec.size();
			double aMed = aVec[aNbV/2];
			// sauvegarder dans PxOut
			aPxOut.SetVal(aX,aY,aMed);
		}
	}
    
     aPxOut.Save("Mediane.tif");
    
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
