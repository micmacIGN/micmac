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
#include <functional>
#include <numeric>

void Arsenic_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " **********************************\n";
    std::cout <<  " *     A-utomated                 *\n";
    std::cout <<  " *     R-adiometric               *\n";
    std::cout <<  " *     S-hift                     *\n";
    std::cout <<  " *     E-qualization              *\n";
    std::cout <<  " *     N-ormalization             *\n";
    std::cout <<  " *     I-nter-images              *\n";
    std::cout <<  " *     C-orrection                *\n";
    std::cout <<  " **********************************\n\n";
}
vector<vector<double> > ReadPtsHom(string aDir,std::vector<std::string> * aSetIm,string Extension)
{

	vector<double> NbPtsCouple,G1,G2;//Elements of output (distance from SIFT pts to center for Im1 and Im2, and respective grey lvl 
	Pt2di aSz;

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

//On parcours toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		std::cout<<"Getting homologous points from: "<<(*aSetIm)[aK1]<<endl;
		    
		//Reading the image and creating the objects to be manipulated
			Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK1],1,false);
			aSz = aTF1.sz();
			Im2D_REAL16  aIm1(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF1.all_pts(),
				   aTF1.in(),
				   aIm1.out()
				);

			REAL16 ** aData1 = aIm1.data();

        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
			if (aK1!=aK2)
            {
			Tiff_Im aTF2= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK2],1,false);
			Im2D_REAL16  aIm2(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF2.all_pts(),
				   aTF2.in(),
				   aIm2.out()
				);

			REAL16 ** aData2 = aIm2.data();

			string prefixe="";
            
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
						int cpt=0;
							for 
							(
								ElPackHomologue::const_iterator itP=aPack.begin();
								itP!=aPack.end();
								itP++
							)
							{
								cpt++;
								//Go looking for grey value of the point, adjusted to ISO and Exposure time induced variations
								double Grey1 =Reechantillonnage::biline(aData1, aSz.x, aSz.y, itP->P1());
								double Grey2 =Reechantillonnage::biline(aData2, aSz.x, aSz.y, itP->P2());
							G1.push_back(Grey1);
							G2.push_back(Grey2);
							}
						NbPtsCouple.push_back(double(cpt));
				   }
                   else{
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
					  NbPtsCouple.push_back(0);
				   }
            }
        }
    }
	int nbpts=G1.size();
	vector<vector<double> > aPtsHomol;
	vector<double> SZ;
	SZ.push_back(aSz.x);SZ.push_back(aSz.y);
	aPtsHomol.push_back(NbPtsCouple);
	aPtsHomol.push_back(G1);
	aPtsHomol.push_back(G2);
	aPtsHomol.push_back(SZ);
   return aPtsHomol;
}

vector<double> Egalisation_factors(vector<vector<double> > aPtsHomol)
{
vector<double> K;

double sum1 = std::accumulate(aPtsHomol[1].begin(),aPtsHomol[1].begin()+aPtsHomol[0][0]-1,0.0);
double sum2 = std::accumulate(aPtsHomol[2].begin(),aPtsHomol[2].begin()+aPtsHomol[0][0]-1,0.0);
K.push_back(sum1/sum2);
int nbParcouru=aPtsHomol[0][0];
for (int i=1;i<int(aPtsHomol[0].size());i++){
	//cout<<aPtsHomol[1].size()<<endl;
	//cout<<"nbPoints : "<<aPtsHomol[0][i]<<endl;
	//cout<<"load from "<<nbParcouru<<" to "<<nbParcouru+aPtsHomol[0][i]-1<<endl;
	sum1 = std::accumulate(aPtsHomol[1].begin()+nbParcouru,aPtsHomol[1].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
	sum2 = std::accumulate(aPtsHomol[2].begin()+nbParcouru,aPtsHomol[2].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
	nbParcouru=nbParcouru+aPtsHomol[0][i];
	K.push_back(sum1/sum2);
}

return K;
}

void Egal_correct(string aDir,std::vector<std::string> * aSetIm,vector<double> K_used,string aDirOut)
{
	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
    int nbIm=(aSetIm)->size();
    for(int i=0;i<nbIm;i++)
	{
	    string aNameIm=(*aSetIm)[i];
		cout<<"Correcting "<<aNameIm<<endl;
		string aNameOut=aDir + aDirOut + aNameIm +"_egal.tif";

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
		
		double aCor=K_used[i];
		cout<<"Correction factor for image n"<<i<<" ="<<aCor<<endl;
		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					double x0=aSz.x/2;
					double y0=aSz.y/2;
					double D=pow(aX-x0,2)+pow(aY-y0,2);
					double R = aDataR[aY][aX] * aCor;
					double G = aDataG[aY][aX] * aCor;
					double B = aDataB[aY][aX] * aCor;
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

int  Arsenic_main(int argc,char ** argv)
{

	std::string aFullPattern,aDirOut="Egal/";
	bool InTxt=false,DoCor=false;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						//<< EAM(InVig,"InVig",true,"Input vignette parameters")
						<< EAM(InTxt,"InTxt",true,"True if homologous points have been exported in txt (Defaut=false)")
						<< EAM(DoCor,"DoCor",true,"Use the computed parameters to correct the images (Defaut=false)")
        );
		std::string aDir,aPatIm;
		SplitDirAndFile(aDir,aPatIm,aFullPattern);

		std::string Extension = "dat";
		if (InTxt){Extension="txt";}

		cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

		std::vector<std::string> aVectIm=*aSetIm;
	
		vector<vector<double> > aPtsHomol=ReadPtsHom(aDir, & aVectIm, Extension);

		cout<<"Computing equalization factors"<<endl;
		vector<double> K=Egalisation_factors(aPtsHomol);
		
		cout<<"Choosing correction to apply"<<endl;
		double Kmax = 0;
		int imMax=0;
		for(int i=0;i<int(K.size());i++)
		{
		if(K[i]>Kmax){Kmax=K[i];imMax=i;}
		cout<<K[i]<<endl;
		}
		
		int nbIm=aVectIm.size();
		vector<double> K_used;
		for(int i=0;i<int(nbIm);i++)
		{
			if(i<imMax/(nbIm-1)){K_used.push_back(K[(imMax/(nbIm-1))*(nbIm-1)+i]);}
			else if(i>imMax/(nbIm-1)){K_used.push_back(K[(imMax/(nbIm-1))*(nbIm-1)+i-1]);}
			else{K_used.push_back(1);}
		}
		if(DoCor){
			Egal_correct(aDir, & aVectIm, K_used, aDirOut);
		}
		Arsenic_Banniere();

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

