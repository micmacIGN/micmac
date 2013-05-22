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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "../../include/StdAfx.h"
#include "hassan/reechantillonnage.h"
#include <algorithm>

void Vodka_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     V-ignette                 *\n";
    std::cout <<  " *     O-f                       *\n";
    std::cout <<  " *     D-igital                  *\n";
    std::cout <<  " *     K-amera                   *\n";
    std::cout <<  " *     A-nalysis                 *\n";
    std::cout <<  " *********************************\n\n";
}

vector<vector<double> > ReadPtsHom(string aDir,string aPatIm,string Extension)
{

	vector<double> D1,X1,Y1,D2,X2,Y2,G1,G2;
	Pt2di aSz;
    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

//On parcours toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		cout<<"Getting homologous points from: "<<(*aSetIm)[aK1]<<endl;
		//
		    
		//Reading the image and creating the objects to be manipulated
			Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK1],1,false);
			aSz = aTF1.sz();
			Im2D_U_INT1  aIm1(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF1.all_pts(),
				   aTF1.in(),
				   aIm1.out()
				);

			U_INT1 ** aData1 = aIm1.data();


        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
			Tiff_Im aTF2= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK2],1,false);
			Im2D_U_INT1  aIm2(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF2.all_pts(),
				   aTF2.in(),
				   aIm2.out()
				);

			U_INT1 ** aData2 = aIm2.data();

			string prefixe="";
            if (aK1!=aK2)
            {
               std::string aNamePack =  aDir +  aICNM->Assoc1To2
                                        (
                                           "NKS-Assoc-CplIm2Hom@"+prefixe + "@" + Extension,
                                           (*aSetIm)[aK1],
                                           (*aSetIm)[aK2],
                                           true
                                        );

				   bool Exist = ELISE_fp::exist_file(aNamePack);
                   if (Exist)
                   {
                      ElPackHomologue aPack = ElPackHomologue::FromFile(aNamePack);

                           for 
                           (
                               ElPackHomologue::const_iterator itP=aPack.begin();
                               itP!=aPack.end();
                               itP++
                           )
                           {

							   X1.push_back(itP->P1().x);
							   Y1.push_back(itP->P1().y);
							   X2.push_back(itP->P2().x);
							   Y2.push_back(itP->P2().y);
							   //Compute the distance between the point and the center of the image
							   double x0=aSz.x/2;
							   double y0=aSz.y/2;
							   double D=sqrt(pow(itP->P1().x-x0,2)+pow(itP->P1().y-y0,2));
							   D1.push_back(D);
							   x0=aSz.x/2;
							   y0=aSz.y/2;
							   D=sqrt(pow(itP->P2().x-x0,2)+pow(itP->P2().y-y0,2));
							   D2.push_back(D);
							   //Go looking for grey value of the point
							   double G = Reechantillonnage::biline(aData1, 1,1, itP->P1());
							   G1.push_back(G);
							   G = Reechantillonnage::biline(aData2, 1,1, itP->P2());
							   G2.push_back(G);

                     // std::cout << aNamePack  << " " << aPack.size() << "\n";
                   }
				   }
                   else
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
            }
        }
    }

	vector<vector<double> > aPtsHomol;
	aPtsHomol.push_back(D1);
	aPtsHomol.push_back(X1);
	aPtsHomol.push_back(Y1);
	aPtsHomol.push_back(D2);
	aPtsHomol.push_back(X2);
	aPtsHomol.push_back(Y2);
	aPtsHomol.push_back(G1);
	aPtsHomol.push_back(G2);
	vector<double> SZ;SZ.push_back(aSz.x);SZ.push_back(aSz.y);
	aPtsHomol.push_back(SZ);
   return aPtsHomol;
}


void Vignette_correct(string aDir,string aPatIm,double *aParam,string aDirOut){

	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
    list<string> ListIm=RegexListFileMatch(aDir,aPatIm,1,false);
    int nbIm=ListIm.size();

    for(int i=1;i<=nbIm;i++)
	{
	    string aNameIm=ListIm.front();
        ListIm.pop_front();
		string aNameOut=aDir + aDirOut + aNameIm +"_vodka.tif";

		//Reading the image and creating the objects to be manipulated
		Tiff_Im aTF= Tiff_Im::StdConvGen(aDir + aNameIm,3,false);
		Pt2di aSz = aTF.sz();

		Im2D_U_INT1  aImR(aSz.x,aSz.y);
		Im2D_U_INT1  aImG(aSz.x,aSz.y);
		Im2D_U_INT1  aImB(aSz.x,aSz.y);

		ELISE_COPY
		(
		   aTF.all_pts(),
		   aTF.in(),
		   Virgule(aImR.out(),aImG.out(),aImB.out())
		);

		U_INT1 ** aDataR = aImR.data();
		U_INT1 ** aDataG = aImG.data();
		U_INT1 ** aDataB = aImB.data();

		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					double x0=aSz.x/2;
					double y0=aSz.y/2;
					double D=pow(aX-x0,2)+pow(aY-y0,2);
					double R = aDataR[aY][aX] + 255*(aParam[0]*D+aParam[1]*pow(D,2)+aParam[2]*pow(D,3));
					double G = aDataG[aY][aX] + 255*(aParam[0]*D+aParam[1]*pow(D,2)+aParam[2]*pow(D,3));
					double B = aDataB[aY][aX] + 255*(aParam[0]*D+aParam[1]*pow(D,2)+aParam[2]*pow(D,3));
					if(R>255){aDataR[aY][aX]=255;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else{aDataB[aY][aX]=B;}
				}
		}

		 Tiff_Im  aTOut
			(
				aNameOut.c_str(),
				aSz,
				GenIm::u_int1,
				Tiff_Im::No_Compr,
				Tiff_Im::RGB
			);


		 ELISE_COPY
			 (
				 aTOut.all_pts(),
				 Virgule(aImR.in(),aImG.in(),aImB.in()),
				 aTOut.out()
			 );
	  
	}
}

double* Vignette_Solve(L2SysSurResol & aSys)
{
    bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok);

    if (Ok)
    {
        double* aData = aSol.data();
        std::cout << "Vignette parameters : " << aData[0] << " " << aData[1] << " " << aData[2] << "\n";
		return aData;
    }else{
		return 0;}
}

int  Vignette_main(int argc,char ** argv)
{
   std::cout << "Correting the vignetting effect \n";
   // Create L2SysSurResol to solve least square equation with 2 unknown

 
	std::string aFullPattern,aDirOut="Vignette/";
	bool InTxt=false;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						<< EAM(InTxt,"InTxt",true,"True if homologous points have been exported in txt (Defaut=false)")
                    );
		std::string aDir,aPatIm;
		SplitDirAndFile(aDir,aPatIm,aFullPattern);

		std::string Extension = "dat";
		if (InTxt){Extension="txt";}

	vector<vector<double> > aPtsHomol=ReadPtsHom(aDir,aPatIm,Extension);
	//aPtsHomol est l'ensemble des vecteurs D1,X1,Y1,D2,X2,Y2,G1,G2;

//For Each SIFT point

   L2SysSurResol aSys(3);
   cout<<"Total number of points used in least square : "<<aPtsHomol[0].size()<<endl;
   for(int i=0;i<int(aPtsHomol[0].size());i++){
	   {
		   double aPds[3]={(aPtsHomol[7][i]*pow(aPtsHomol[3][i],2)-aPtsHomol[6][i]*pow(aPtsHomol[0][i],2)),
						   (aPtsHomol[7][i]*pow(aPtsHomol[3][i],4)-aPtsHomol[6][i]*pow(aPtsHomol[0][i],4)),
						   (aPtsHomol[7][i]*pow(aPtsHomol[3][i],6)-aPtsHomol[6][i]*pow(aPtsHomol[0][i],6)),
						   //(aPtsHomol[7][i]*aPtsHomol[4][i]-aPtsHomol[6][i]*aPtsHomol[1][i]),
						   //(aPtsHomol[7][i]*aPtsHomol[5][i]-aPtsHomol[6][i]*aPtsHomol[2][i])
						};
				 aSys.AddEquation(1,aPds,aPtsHomol[6][i]-aPtsHomol[7][i]);
	   }
	}
	//System has 3 unknowns and nbPtsSIFT equations (significantly more than enough)

   double* aParam = Vignette_Solve(aSys);

   if (aParam==0){
	   cout<<"Could'nt compute vignette parameters"<<endl;
   }else{
	   cout<<"Correcting the images"<<endl;
	   Pt2di aSz;aSz.x=aPtsHomol[8][0];aSz.y=aPtsHomol[8][1];
	   Vignette_correct(aDir,aPatIm,aParam,aDirOut);
   }

   Vodka_Banniere();
   return 0;
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

