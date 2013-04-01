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
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/


#include "StdAfx.h"

#if (ELISE_X11)



using namespace NS_ParamChantierPhotogram;

#if (0)


#endif






Fonc_Num Correl(Fonc_Num aF1,Fonc_Num aF2,int aNb)
{
   Symb_FNum aM1 (Moy(aF1,aNb));
   Symb_FNum aM2 (Moy(aF2,aNb));

   Fonc_Num aEnct1 = Moy(Square(aF1),aNb) -Square(aM1);
   Fonc_Num aEnct2 = Moy(Square(aF2),aNb) -Square(aM2);


   return (Moy(aF1*aF2,aNb)  -aM1*aM2) / sqrt(Max(1e-5,aEnct1*aEnct2));
}

void AutoCorrel(const std::string & aName)
{
   Tiff_Im aTF(aName.c_str());
   Pt2di aSz = aTF.sz();
   Im2D_REAL4 anI(aSz.x,aSz.y);
   ELISE_COPY(aTF.all_pts(),aTF.in(),anI.out());

   int aNb = 2;

   Fonc_Num aF = 1.0;
   for (int aK=0 ; aK<4 ; aK++)
   {
      aF = Min(aF,Correl(anI.in(0),trans(anI.in(0),TAB_4_NEIGH[aK])*(aNb*2),aNb));
   }
  
   Tiff_Im::Create8BFromFonc
   (
       StdPrefix(aName)+"_AutoCor.tif",
       aSz,
       Min(255,Max(0,(1+aF)*128))
   );
}


Im2D_REAL4 Conv2Float(Im2DGen anI)
{
   Pt2di aSz = anI.sz();
   Im2D_REAL4 aRes(aSz.x,aSz.y);
   ELISE_COPY(anI.all_pts(),anI.in(),aRes.out());
   return aRes;
}




void TestKL()
{
   Pt2di aSZ(200,200);
   Im2D_Bits<1> aImMasqF(aSZ.x,aSZ.y,1);

   Im2D_Bits<1> aImMasqDef(aSZ.x,aSZ.y,1);
   ELISE_COPY(rectangle(Pt2di(70,0),Pt2di(130,200)),0,aImMasqDef.out());

   Im2D<U_INT2,INT> aImVal(aSZ.x,aSZ.y);
   ELISE_COPY(aImVal.all_pts(),FX,aImVal.out());

   Video_Win aW=Video_Win::WStd(aSZ,3.0);
   ELISE_COPY(aW.all_pts(),aImVal.in(),aW.ogray());
   ELISE_COPY(aW.all_pts(),aImMasqDef.in(),aW.odisc());
   getchar();


   aImVal = ImpaintL2(aImMasqDef,aImMasqF,aImVal);

   // NComplKLipsParLBas(aImMasqDef,aImMasqF,aImVal,1.0);

   ELISE_COPY(aW.all_pts(),aImVal.in(),aW.ogray());

   Tiff_Im::Create8BFromFonc("toto.tif",aSZ,aImVal.in());
   getchar();
}
#if (0)
#endif










int MPDtest_main (int argc,char** argv)
{
//    TestKL();
//    BanniereMM3D();
   // AutoCorrel(argv[1]);
   double aNan = strtod("NAN(teta01)", NULL);
   std::cout << "Nan=" << aNan << "\n";

    float aX = 6e6;
    for (int aK=0 ; aK<100000; aK++)
    {
          float aY = aX + (aK*1e-5);
          float aDif = (aX-aY);
          std::cout << "TTT " << aK << " " << aDif  << " " <<  (aK*1e-5) << "\n";

          if (aDif != 0)
             getchar();
    }

    return EXIT_SUCCESS;
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
