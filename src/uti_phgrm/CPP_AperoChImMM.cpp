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

int AperoChImMM_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string  aDir,aPat,aFullDir;
    std::string AeroIn;
    std::string Out;
    std::string aPatternExport=".*";
    std::string aSetHom="";
    bool ExpTxt=0;
    bool CalPerIm=0;
    double aPenalPerIm = 0.20;
    double aTetaOpt = 0.17;
    std::string aMasq3D;
    std::vector<int>  mVecDep;
    bool OnlyVecDep=false;

    Pt2dr aFocs;
    bool ExpImSec = true;


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Dir + Pattern", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation", eSAM_IsExistDirOri),
    LArgMain()
                    << EAM(ExpTxt,"ExpTxt",true,"Use tie points in text format (def = false, use dat format)", eSAM_IsBool)
                    << EAM(Out,"Out",true,"Output destination (Def= same as Orientation-parameter)", eSAM_IsOutputFile)
                    << EAM(CalPerIm,"CalPerIm",true,"If a calibration per image was used (Def=False)", eSAM_IsBool)
                    << EAM(aPatternExport,"PatExp",true,"Pattern to limit export (Def=.* , i.e. all are exported)", eSAM_IsBool)
                    << EAM(aFocs,"Focs",true,"Interval of Focal")
                    << EAM(aPenalPerIm,"PenPerIm",true,"Penality per image, to limite size, def = 0.3")
                    << EAM(aTetaOpt,"TetaOpt",true,"Optimal angle of stereoscopy, in radian, def=0.17 (+or- 10 degree)")
                    << EAM(aMasq3D,"Masq3D",true,"Masq3D for tie points selection")
                    << EAM(mVecDep,"VecDep",true,"Fixed Vec of deplacement (adapted to video like acquisition) ")
                    << EAM(OnlyVecDep,"OnlyVecDep",true,"Only vec dep (internal purpose) ",eSAM_InternalUse)
                    << EAM(ExpImSec,"ExpImSec",true,"Export Im Sec (Def= true) ",eSAM_IsBool)
                    << EAM(aSetHom,"SH",false,"Set of Hom, Def=\"\" ")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif

   SplitDirAndFile(aDir,aPat,aFullDir);
   StdCorrecNameOrient(AeroIn,aDir);

    if (! EAMIsInit(&Out))
       Out = AeroIn;




    string aXmlName="Apero-Choix-ImSec.xml";
/*
    if (CalPerIm)
    {
        aXmlName="Apero-Choix-ImSec-PerIm.xml";
    }
*/

    std::string aCom =   MM3dBinFile("Apero")
                       + XML_MM_File(aXmlName)
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                       + std::string(" +AeroIn=-") + AeroIn
                       + std::string(" +Out=-") + Out
                       + std::string(" +PatternExport=") + QUOTE(aPatternExport) + std::string(" ")
                       + std::string(" +PenalNbIm=") + ToString(aPenalPerIm) + std::string(" ")
                       + std::string(" +TetaOpt=") + ToString(aTetaOpt) + std::string(" ")
                    ;
    if (EAMIsInit(&CalPerIm))
         aCom =  aCom + " +CalPerIm=" +ToString(CalPerIm);


    if (EAMIsInit(&aFocs))
    {
       aCom = aCom + " +FocMin=" + ToString(aFocs.x) + " +FocMax=" + ToString(aFocs.y);
    }

    if (EAMIsInit(&aMasq3D))
    {
        aCom = aCom + " +UseMasq3D=true +Masq3D=" + aMasq3D;
    }

    if (EAMIsInit(&ExpImSec))
    {
        aCom = aCom + " +ExportImSec=" + ToString(ExpImSec);
    }

    if (EAMIsInit(&aSetHom))
    {
        aCom = aCom + " +SetHom=" + aSetHom;
    }

   int aRes = EXIT_SUCCESS;
   std::cout << "Com = " << aCom << "\n";
   if (!OnlyVecDep)
   {
         aRes = system_call(aCom.c_str());
   }

    if (EAMIsInit(&mVecDep))
    {
         cElemAppliSetFile anEASF(aFullDir);
         const std::vector<std::string> & aVN = *(anEASF.SetIm());

         for (int aK=0 ; aK<int(aVN.size()) ; aK++)
         {
              cImSecOfMaster aISOM;
              aISOM.Master() = aVN[aK];
              aISOM.UsedPenal() = 1;
              cOneSolImageSec aSol;
              aSol.Coverage() = 1.0;
              aSol.Score() = 1.0;
              for (int aKV =0 ; aKV<int(mVecDep.size()) ; aKV++)
              {
                  int aV = aK + mVecDep[aKV];
                  if ((aV>=0) && (aV<int(aVN.size())))
                  {
                     aSol.Images().push_back(aVN[aV]);
                  }
              }
              aISOM.Sols().push_back(aSol);
              std::string aName = anEASF.mICNM->Assoc1To1("NKS-Assoc-ImSec@-"+Out,aVN[aK],true);
              MakeFileXML(aISOM,aName);
         }
    }

   return aRes;
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
