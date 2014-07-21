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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>

Fonc_Num Gama(Fonc_Num aF,REAL aG)
{
   return 255.0 * pow(aF/255.0,1/aG);
}

int  main(int argc,char ** argv)
{
      std::string aPref,aName1,aName2;
      REAL aZoom = 1.0;
      Pt2dr aTr(0,0);
      Pt2dr aSzMax (1000,800);

      REAL aGamaExp = 1.0;

      ElInitArgMain
      (
          argc,argv,
          LArgMain() << EAM(aPref) << EAM(aName1) << EAM(aName2),
          LArgMain() << EAM(aZoom,"Zoom",true)
	             << EAM(aTr,"Tr",true)
	             << EAM(aSzMax,"SzMax",true)
                     << EAM(aGamaExp,"Gama",true)
     );	

      aName1 = aPref + aName1;
      aName2 = aPref + aName2;
      Im2DGen aI1 = Tiff_Im::StdConv(aName1.c_str()).ReadIm();
      Im2DGen aI2 = Tiff_Im::StdConv(aName2.c_str()).ReadIm();

     Pt2dr aSz =   Pt2dr(aI1.sz()) * aZoom;
     aSz.SetInf(aSzMax);

     cElImageFlipper aFliper
     (
        aSz,
        Gama(aI1.in(0)[Virgule(FX+aTr.x,FY+aTr.y)/aZoom],aGamaExp),
        Gama(aI2.in(0)[Virgule(FX+aTr.x,FY+aTr.y)/aZoom],aGamaExp)
     );

     while (true)
     {
         INT aNb;
	 REAL aTime;
         cin >> aNb >> aTime;
	 aFliper.Flip(aNb,aTime);
     }

}

#if (0)

int main(int argc,char ** argv)
{
	string Name;
	Pt2di  Dalles(256,256);
	string Compr("Id"); // Par defaut on reprend le meme mode de compr.

	Pt2di	SzMaxVisu(700,500);
	INT     Visu = 1;


        INT Z0 = 1;
        INT Z1 = 256;
        
        string SPrefix = "";


	ElInitArgMain
	(
		argc,argv,
		LArgMain() 	<< EAM(Name) ,
		LArgMain() 	<< EAM(Dalles,"Dalles",true)
				<< EAM(Z0,"Z0",true)
				<< EAM(Z1,"Z1",true)
				<< EAM(Compr,"Compr",true)
				<< EAM(SzMaxVisu,"SzMaxVisu",true)
				<< EAM(Visu,"Visu",true)
			        << EAM(SPrefix,"NameOut",true)
	);	

        if (SPrefix=="")
           SPrefix = StdPrefix(Name);

        Z0 = ElMax(1,ElMin(2,Z0));

	Tiff_Im TIFF = Tiff_Im::StdConv(Name); 


	Tiff_Im::COMPR_TYPE  MC = TIFF.mode_compr();
	if (Compr != "Id")
		MC = Tiff_Im::mode_compr(Compr);


        Tiff_Im::PH_INTER_TYPE aPhotoInt = TIFF.phot_interp();



        for (INT aZ=Z0 ; aZ<Z1 ; aZ *=2)
        {
             cout << "A Zoom = " << aZ << "\n";
             char aBuf[512];
             sprintf(aBuf,"%sReduc%d.tif",SPrefix.c_str(),aZ);
             Pt2di aSz = TIFF.sz() / aZ;

             Tiff_Im aTOut(aBuf,aSz,GenIm::u_int1,MC,aPhotoInt);

             if (aZ==1)
             {
                ELISE_COPY(TIFF.all_pts(),TIFF.in(),aTOut.out());
             }
             else
             {
                  char aBuf2[512];
                  sprintf(aBuf2,"%sReduc%d.tif",SPrefix.c_str(),aZ/2);
                  Tiff_Im aPrec(aBuf2);
                  Im2D_REAL8 aMasq
                  (  3,3,
                     " 1 2 1 "
                     " 2 4 2 "
                     " 1 2 1 "
                   );
                   Im2D_U_INT1 anI(aSz.x*2,aSz.y*2);
                   ELISE_COPY
                   (
                        anI.all_pts(),
                          som_masq(aPrec.in(0),aMasq)
                        / Max(1.0,som_masq(aPrec.inside(),aMasq)),
                        anI.out()
                   );
                   ELISE_COPY
                   (
                       aTOut.all_pts(),
                       anI.in()[Virgule(FX*2,FY*2)],
                       aTOut.out()
                   );
             }
        }

         return 1;
}


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
