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

#include "Xeres.h"


int XeresTest_Main(int argc,char** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aSeq;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aSeq, "Sequence"),
         LArgMain()  // << EAM(mCalib,"OriCalib",true,"Calibration folder if any")
   );

   cAppliXeres anAppli("./",aSeq);
   anAppli.TestInteractNeigh();

   return EXIT_SUCCESS;
}




int XeresTieP_Main(int argc,char** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aSeq;
   int aSz,aNbHom=2;
   std::string aDir="./";
   std::string aNameCpleSup="";

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aSeq, "Sequence")
                     << EAMC(aSz,"Sz Tie points"),
         LArgMain()  << EAM(aDir,"Dir",true,"Folder of data, Def=./")
                     << EAM(aNbHom,"DV",true,"Delta Vois, Def=2")
                     << EAM(aNameCpleSup,"CpleSup",true,"File for additional cple")
   );

   cAppliXeres anAppli(aDir,aSeq);
   anAppli.CalculTiePoint(aSz,aNbHom,aNameCpleSup);

   return EXIT_SUCCESS;
}


int XeresMergeTieP_Main(int argc,char** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::vector<std::string> aVSeq;
   std::string aDir="./";
   std::string aPostMerge;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aVSeq, "Sequence"),
         LArgMain()  << EAM(aDir,"Dir",true,"Folder of data, Def=./")
                     << EAM(aPostMerge,"Postfix of merged folder")
   );

   if (! EAMIsInit(&aPostMerge))
   {
        aPostMerge = "Merge-" + aVSeq[0];
   }

   std::vector<cAppliXeres *> aVAp;
   for (int aK=0 ; aK<int(aVSeq.size()) ; aK++)
   {
        aVAp.push_back(new cAppliXeres (aDir,aVSeq[aK]));
   }

   cAppliXeres::FusionneHom(aVAp,aPostMerge);

   return EXIT_SUCCESS;
}

int XeresHomMatch_main(int argc,char** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aDir="./";
   std::string aSeq;
   std::string anOri;
   std::string aPat=".*";

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aSeq, "Sequence")
                     << EAMC(anOri,"Orientation"),
         LArgMain()  << EAM(aDir,"Dir",true,"Folder of data, Def=./")
                     << EAM(aPat,"Filter",true,"Filter for selection")
   );

   cElRegex * anAutomFilter = new cElRegex (aPat,10);

   cAppliXeres  anAppli(aDir,aSeq,anAutomFilter);
   StdCorrecNameOrient(anOri,aDir);
   anAppli.CalculHomMatch(anOri);

   return EXIT_SUCCESS;
}

int XeresReNameInit_main(int argc,char** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aDir="./";
   std::string aSeq;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aSeq, "Sequence"),
         LArgMain()  << EAM(aDir,"Dir",true,"Folder of data, Def=./")
   );

   std::string aCom = MM3dBinFile_quotes( "MyRename " )
                      + "\"([A-Z][0-9]{1,2})_.*\"   \"\\$1_" + aSeq + ".jpg\" Exe=1" ;

   // std::cout << aCom << "\n";
   System(aCom);



   return EXIT_SUCCESS;
}

int XeresCalibMain_main(int argc,char** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aSeq,aDir,OutCal="Calib";
   int aSz=1500;



   ElInitArgMain
   (
         argc,argv,
         LArgMain()  //  << EAMC(aSeq, "Sequence")
                     << EAMC(aDir, "Directory"),
         LArgMain()  << EAM(aSeq,"Seq",true,"Folder of data, Def=./")
                     << EAM(aSz,"Sz",true,"Sz of TieP, Def=1500")
                     << EAM(OutCal,"Out",true,"")
   );

   // std::string aCdDir = "cd " + aDir + "/";
   // System(aCdDir);
   if (!EAMIsInit(&aSeq) ) aSeq = aDir;

   cElemAppliSetFile anEASF(aDir+"/.*jpg");

   const std::vector<std::string> * aVS = anEASF.SetIm();
   int aNbIm = aVS->size();
   for (int aK=0 ; aK<aNbIm ; aK++)
   {
        const std::string & aName = (*aVS)[aK];
        ELISE_fp::MvFile(aDir+"/"+aName,aDir+"/"+aSeq+"_Calib" +ToString(aK) + ".jpg");
        std::cout << "NAME = " << aName << "\n";
   }

   std::string aStrMMD= "MicMac-LocalChantierDescripteur.xml";

   ELISE_fp::CpFile(aStrMMD,aDir+"/"+aStrMMD);


   std::string aComTiep = MM3dBinFile_quotes("Tapioca") + " All  " + aDir + "/.*jpg " + ToString(aSz);
   System(aComTiep);

   std::string aComOri =  MM3dBinFile_quotes("Tapas ") + " FraserBasic " +  aDir + "/.*jpg " + " Out=" + OutCal
                          + " RankInitPP=0 RankInitF=1 RefineAll=0";
   System(aComOri);


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
aooter-MicMac-eLiSe-25/06/2007*/
