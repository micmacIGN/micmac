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
#include "StdAfx.h"
#include <algorithm>
#include "hassan/reechantillonnage.h"


// Reading Homologous point associated to a set of images


int  Luc_test_ptshom_main(int argc,char ** argv)
{
  //=====================  PARAMETRES EN DUR ==============

   std::string aDir = "C:/Users/Luc Girod/Desktop/TFE/Vignettage/vignette_sift3/";
   std::string aPatIm = ".*NEF";
   std::string Prefix = "";
   // std::string Prefix =  "_SRes" ; 
   std::string Extension = "dat";

  //===================== 

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

//On parcours toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		cout<<(*aSetIm)[aK1]<<endl;
        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
            if (aK1!=aK2)
            {
               std::string aNamePack =  aDir +  aICNM->Assoc1To2
                                        (
                                           "NKS-Assoc-CplIm2Hom@"+ Prefix + "@"+Extension,
                                           (*aSetIm)[aK1],
                                           (*aSetIm)[aK2],
                                           true
                                    );
             
               if (aK1==0) 
               {
                   bool Exist = ELISE_fp::exist_file(aNamePack);
                   if (Exist)
                   {
                      ElPackHomologue aPack = ElPackHomologue::FromFile(aNamePack);
                      if (aK2==1)
                      {
                           int aNb=0;
                           for 
                           (
                               ElPackHomologue::const_iterator itP=aPack.begin();
                               itP!=aPack.end();
                               itP++
                           )
                           {
                              if (aNb<10)
								  std::cout  << itP->P1() << itP->P2() <<"\n";
                              aNb++;
                           }
                      }
                      std::cout << aNamePack  << " " << aPack.size() << "\n";
                   }
                   else
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
               }
            }
        }
    }


	return 0;
}

void RotateImage(double alpha,Pt2dr P1, Pt2dr P2, Pt2dr P3, string aNameDir, string aNameIm)
{
	cout<<"Rotating "<<aNameIm<<endl;
	string aNameOut=aNameDir + aNameIm + "_rotated.tif";
	//Reading the image and creating the objects to be manipulated
    Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aNameIm,1,false);

	int border=10;
    Pt2di aSz = aTF.sz();
	Pt2dr P1Cor; P1Cor.x=cos(-alpha)*(P1.x-P2.x)+sin(-alpha)*(P1.y-P2.y)+P2.x; P1Cor.y=-sin(-alpha)*(P1.x-P2.x)+cos(-alpha)*(P1.y-P2.y)+P2.y;
	Pt2dr P3Cor; P3Cor.x=cos(-alpha)*(P3.x-P2.x)+sin(-alpha)*(P3.y-P2.y)+P2.x; P3Cor.y=-sin(-alpha)*(P3.x-P2.x)+cos(-alpha)*(P3.y-P2.y)+P2.y;
	Pt2di aSzOut; aSzOut.x=P3Cor.x-P1Cor.x-2*border+1; aSzOut.y=P3Cor.y-P1Cor.y-2*border+1;
	cout<< P1Cor << " " << P3Cor << " " << aSzOut <<endl;
    Im2D_U_INT1  aImR(aSz.x,aSz.y);
    //Im2D_U_INT1  aImG(aSz.x,aSz.y);
    //Im2D_U_INT1  aImB(aSz.x,aSz.y);
    Im2D_U_INT1  aImROut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImGOut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImBOut(aSzOut.x,aSzOut.y);

    ELISE_COPY
    (
       aTF.all_pts(),
       aTF.in(),
       aImR.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
    );

    U_INT1 ** aDataR = aImR.data();
    //U_INT1 ** aDataG = aImG.data();
    //U_INT1 ** aDataB = aImB.data();
    U_INT1 ** aDataROut = aImROut.data();
    U_INT1 ** aDataGOut = aImGOut.data();
    U_INT1 ** aDataBOut = aImBOut.data();

    //Parcours des points de l'image de sortie et remplissage des valeurs
    Pt2dr ptOut;
	cout<<"Image is being corrected"<<endl;
    for (int aY=P1Cor.y+border ; aY<P3Cor.y-border-1  ; aY++)
    {//cout<<aY<< " ";
        for (int aX=P1Cor.x+border ; aX<P3Cor.x-border-1  ; aX++)
		{//if(aY== 6387){cout<<aX<< " ";}
			//ptOut=aCam->DistDirecte(Pt2dr(aX,aY));
			ptOut.x=cos(-alpha)*(aX-P2.x)+sin(-alpha)*(aY-P2.y)+P2.x;
			ptOut.y=-sin(-alpha)*(aX-P2.x)+cos(-alpha)*(aY-P2.y)+P2.y;
			aDataROut[aY-((int)P1Cor.y+border)][aX-((int)P1Cor.x+border)] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
			aDataGOut[aY-((int)P1Cor.y+border)][aX-((int)P1Cor.x+border)] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
			aDataBOut[aY-((int)P1Cor.y+border)][aX-((int)P1Cor.x+border)] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);

        }
    }
	cout<<"BIM"<<endl;
    Tiff_Im  aTOut
             (
                  aNameOut.c_str(),
                  aSzOut,
                  GenIm::u_int1,
                  Tiff_Im::No_Compr,
                  Tiff_Im::RGB
             );


     ELISE_COPY
     (
         aTOut.all_pts(),
         Virgule(aImROut.in(),aImGOut.in(),aImBOut.in()),
         aTOut.out()
     );

}
int  Luc_main(int argc,char ** argv){

	std::string aFullPattern;
	//Reading the arguments
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
        LArgMain()  
    );

	std::string aDir,aPatIm;
	SplitDirAndFile(aDir,aPatIm,aFullPattern);
	
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

	std::vector<std::string> aVectIm=*aSetIm;
	int nbIm=aVectIm.size();


	Pt2dr P1,P2,P3;
	P1.x= 795 ; P1.y= 1064;
	P2.x= 7401; P2.y= 926 ;
	P3.x= 7518; P3.y= 7598;
	//cout<<(P1.y-P2.y)/(P1.x-P2.x)<<endl;
	//cout<<atan((P1.y-P2.y)/(P1.x-P2.x))<<endl;
	//cout<<(P3.x-P2.x)/(P3.y-P2.y)<<endl;
	//cout<<atan((P3.x-P2.x)/(P3.y-P2.y))<<endl;
	double aT1=atan((P1.y-P2.y)/(P1.x-P2.x));
	double aT2=atan(-(P3.x-P2.x)/(P3.y-P2.y));
	cout<<aT1<<" + "<<aT2<< " = " <<(aT1+aT2)<<endl;
	double alpha=(atan((P1.y-P2.y)/(P1.x-P2.x))+atan(-(P3.x-P2.x)/(P3.y-P2.y)))/2;
	cout<<"Alpha = "<<alpha<<endl;
	RotateImage(alpha, P1, P2, P3, aDir, aVectIm[0]);

	return 0;
}
#if (0)
#endif




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
