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



int main(int argc,char ** argv)
{
    std::string aDir,aPat;
    INT         aNivMax=1;
    INT         toSupr=0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAM(aDir) 
                    << EAM(aPat),
        LArgMain()  << EAM(aNivMax,"Niv",true)
                    << EAM(toSupr,"Supr",true)
    );


     std::list<std::string>  aList=  ListFileMatch(aDir,aPat,aNivMax);

     INT NbbSupr = 2;
     INT Masq =  ~( (1<<NbbSupr) -1);
     for (INT aK=0 ; aK< 32 ; aK++)
         cout << aK << " : " << ((Masq & (1<< aK)) != 0) << "\n";

     for 
     (
            std::list<std::string>::iterator itS = aList.begin();
            itS != aList.end();
            itS++
     )
     {
           Tiff_Im aTif = Tiff_Im::StdConv( *itS );

           if (aTif.mode_compr() != Tiff_Im::LZW_Compr)
           {


                 cout << "STRING = [" << *itS << "]"  << " Sz = " << aTif.sz() << " \n";

                 std::string aName1 = *itS + "PdsFortLZW.tif";
                 std::string aName2 = *itS + "PdsFaible.tif";

                 Tiff_Im aTOut1 = Tiff_Im 
                                  (
                                      aName1.c_str(),
                                      aTif.sz(),
                                      GenIm::u_int1,
                                      Tiff_Im::LZW_Compr,
                                      aTif.phot_interp(),
                                         Tiff_Im::Empty_ARG
                                      +   Arg_Tiff(Tiff_Im::APred(Tiff_Im::Hor_Diff))
                                  );

                 Tiff_Im aTOut2 = Tiff_Im 
                                  (
                                      aName2.c_str(),
                                      aTif.sz(),
                                      GenIm::bits4_msbf,
                                      Tiff_Im::No_Compr,
                                      aTif.phot_interp(),
                                         Tiff_Im::Empty_ARG
                                      +   Arg_Tiff(Tiff_Im::APred(Tiff_Im::No_Predic))
                                  );

                  Symb_FNum aFIn (aTif.in() );

                  ELISE_COPY
                  (
                          aTif.all_pts(),
                          Virgule(aFIn/16,aFIn%16),
                          Virgule(aTOut1.out(),aTOut2.out())
                  );

                  INT aDif;
                  ELISE_COPY 
                  ( 
                        aTif.all_pts(), 
                        Abs(aTif.in()- (16*aTOut1.in()+aTOut2.in())),
                        sigma(aDif)
                  );
                  ELISE_ASSERT(aDif==0,"Incohererence in Compr");
                  if (toSupr)
                  {
                       char Buf[1000];
                       sprintf(Buf,"rm %s",itS->c_str());
                       system(Buf);
                  }
           }
     }
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
