/*Header-MicMac-eLiSe-25/06/2007peroChImMM_main

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

int CPP_GCP2MeasureLine3D(int argc,char ** argv)
{
    // The argument need some global computation 
    MMD_InitArgcArgv(argc,argv);
    std::string  aNameFileGCP;
    std::string  aNameLine3GCP;

    ElInitArgMain
    (
        argc,argv,
           // Mandatory arg
        LArgMain()
                    << EAMC(aNameFileGCP,"File for GCP", eSAM_IsPatFile),
           // Mandatory arg
        LArgMain()
                    << EAM(aNameLine3GCP,"Out",true,"")
    );

    if (!EAMIsInit(&aNameLine3GCP))
    {
       aNameLine3GCP = "L3D_"+aNameFileGCP;
    }


    cSetOfMesureAppuisFlottants aGCPDic = StdGetFromPCP(aNameFileGCP,SetOfMesureAppuisFlottants);

   // Pour each image
   for (const auto & aItIm : aGCPDic.MesureAppuiFlottant1Im())
   {
      // we cast to vector 4 sort
      std::vector<cOneMesureAF1I> aVMes(aItIm.OneMesureAF1I().begin(),aItIm.OneMesureAF1I().end());
      std::sort
      (
         aVMes.begin(),
         aVMes.end(),
         [](const cOneMesureAF1I &aM1,const cOneMesureAF1I &aM2) 
         {return aM1.NamePt()<aM2.NamePt();}
      );
      int aNbM = (aVMes.size());
      if ((aNbM%2)!=0)
      {
          std::cout << "NameIm= " << aItIm.NameIm() << " " << aNbM<< "\n";
          ELISE_ASSERT(false,"Odd size of point ");
      }
      for (int aKM=0 ; aKM<aNbM ; aKM+=2)
      {
          std::string aPref1,aPref2,aPost1,aPost2;

          SplitIn2ArroundCar(aVMes[aKM].NamePt()  ,'_',aPref1,aPost1,false);
          SplitIn2ArroundCar(aVMes[aKM+1].NamePt(),'_',aPref2,aPost2,false);


          ELISE_ASSERT(aPref1==aPref2,"Prefix different");
          ELISE_ASSERT(aPost1=="1","Bad postfix");
          ELISE_ASSERT(aPost2=="2","Bad postfix");

          std::cout << aPref1 << "\n";
      }
      // std::cout << aItIm.NameIm() << "\n\n";
   }
   return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
