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
#include <cstring>


int main(int argc,char ** argv)
{
   cTplValGesInit<std::string> NoName;
   std::string aDir,aKey;
   int aBorder = 1;
   int aNbLine = 0;
   int  aForceGray = 0;
   double aScale =  1;
   std::string aRes = "Planche.tif";

   std::vector<int> aColF;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory")
                    << EAMC(aKey,"Pattern"),
        LArgMain()  << EAM(aBorder,"Border",true)
                    <<  EAM(aNbLine,"NbL",true)
                    <<  EAM(aRes,"Res",true)
                    <<  EAM(aScale,"Scale",true)
                    <<  EAM(aColF,"Fond",true)
                    <<  EAM(aForceGray,"Gray",true)
   );

   aRes = aDir + aRes;

   cInterfChantierNameManipulateur * anICM=
                 cInterfChantierNameManipulateur::StdAlloc(0,0,aDir,NoName);

   std::list<std::string>  aLS = anICM->StdGetListOfFile(aKey);

   Pt2di aSzMax;
   std::vector<Tiff_Im> aVT;
   int aNbCh = 0;
   for (std::list<std::string>::const_iterator itS=aLS.begin(); itS!=aLS.end() ; itS++)
   {
        std::cout << *itS << "\n";
        //   aVT.push_back(Tiff_Im::UnivConvStd(aDir+*itS));
        aVT.push_back(Tiff_Im::StdConvGen(aDir+*itS,aForceGray?1:-1,false,true));

        aSzMax.SetSup(aVT.back().sz());
        aNbCh = ElMax(aNbCh,aVT.back().nb_chan());
   }

   if (aColF.size()==0)
      aColF = std::vector<int>(aNbCh,128);


   if (aNbLine==0)
      aNbLine = round_up(sqrt(aVT.size()));

   int aNbCol = round_up(aVT.size()/double(aNbLine));



    Pt2di aSz (
                 round_up((aSzMax.x * aNbCol) / aScale),
                 round_up((aSzMax.y * aNbLine) / aScale)
              );

   std::cout << "SZ = " << aSz << " :: " << aNbCol << " X " << aNbLine  << "\n";

   Tiff_Im::PH_INTER_TYPE aPhI = aVT[0].phot_interp();
   if (aNbCh==3)
      aPhI = Tiff_Im::RGB;
   Tiff_Im  FileRes
            (
                aRes.c_str(),
                aSz,
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                aPhI
                
            );

   Fonc_Num aFoncFond = aColF[0];
   for (int aK=1 ; aK<int(aColF.size()) ; aK++)
       aFoncFond = Virgule(aColF[aK],aFoncFond);

   ELISE_COPY(FileRes.all_pts(),aFoncFond,FileRes.out());

   Pt2di aPB (aBorder,aBorder);

   for (int aK=0 ; aK<int(aVT.size()) ; aK++)
   {
        std::cout << "WRITE " << aVT[aK].name() << "\n";
        int aKX = aK % aNbCol;
        int aKY = aK / aNbCol;

        Pt2di aP0 (
                     (aKX*aSz.x) /aNbCol,
                     (aKY*aSz.y) /aNbLine
                  );
        Pt2di aP1 (
                     ((aKX+1)*aSz.x) /aNbCol,
                     ((aKY+1)*aSz.y) /aNbLine
                  );
         
       Fonc_Num aF0 = aVT[aK].in_proj();
       Fonc_Num aF = aF0;
       while (aF.dimf_out() < aNbCh)
              aF = Virgule(aF0,aF);

       aF = StdFoncChScale(aF,Pt2dr(-aP0.x,-aP0.y)*aScale ,Pt2dr(aScale,aScale));


        ELISE_COPY
        (
              rectangle(aP0+aPB,aP1-aPB),
              aF,
              //  Virgule(NRrandom3(255),NRrandom3(255),NRrandom3(255)),
              FileRes.out()
        );
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
