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
#include "StdAfx.h"


std::string ImMatchFromLastN(const std::string & aDir)
{
    std::string aNameXml = aDir+"MMLastNuage.xml";
    cXML_ParamNuage3DMaille aXN = StdGetFromSI(aNameXml,XML_ParamNuage3DMaille);
    return aXN.Image_Profondeur().Val().Image();
}

std::string ImMasqFromLastN(const std::string & aDir)
{
    std::string aNameXml = aDir+"MMLastNuage.xml";
    cXML_ParamNuage3DMaille aXN = StdGetFromSI(aNameXml,XML_ParamNuage3DMaille);
    return aXN.Image_Profondeur().Val().Masq();
}

 

class cAppliEpiBasic;  // Appli principale


class cAppliEpiBasic
{
    public :
          cAppliEpiBasic(int,char **,bool aModeTestDeep);
    private :
          void InitZoom0(const std::string & aNameIm);
          // En mode TestDeep on simule une programme qui va tourner sur petite dalle
          //    donc pas en paral (car il va deja etre appelle en //) et nettoie apres
          bool mModeTestDeep;

          std::string mDir;     // Dir of data
          std::string mDirTmpMEC ; // Dir where storing result
          std::string mIm1;     // Name of first image
          std::string mIm2;     // Name of second image
          bool        mExe;     // Do we execute or only print command
          int         mZoom0;   // Zoom first step
          int         mInc;     // Inc Px
          int         mPxMoy;     // Inc Px
          double      mRegul;
          int         mSzW;
          float       mNbPix0;
          float       mCsteInc;
          float       mRatioInc;
          int         mNbProc;
          std::string mFileExp; // Name of second image
          std::string mPyr;
};

void cAppliEpiBasic::InitZoom0(const std::string & aNameIm)
{

   std::string aFullName = mDir+aNameIm;
   Tiff_Im aTF(aFullName.c_str());
   Pt2di aSz = aTF.sz();

   double aSz1 = sqrt(double(aSz.x) * double(aSz.y));
   double aRatio =   aSz1 / mNbPix0;

   // Truncated to 512 Because XML file do not handle over
   int aZoom = ElMax(1,ElMin(512,round_up(log2(aRatio))));
   if (!EAMIsInit(&mZoom0))  // If set by users, dont use
   {
       mZoom0 = ElMax(mZoom0,1<<aZoom);  // Max of 2 images
   }
   
   if (! EAMIsInit(&mInc))
      mInc = mCsteInc + mRatioInc * aSz1;
}


cAppliEpiBasic::cAppliEpiBasic(int argc,char ** argv,bool aModeTestDeep) :
     mModeTestDeep (aModeTestDeep),
     mDirTmpMEC    ("MEC-BasicEpip/"),
     mExe          (true),
     mZoom0        (1),
     mPxMoy        (0),
     mRegul        (0.1),
     mSzW          (2),
     mNbPix0       (400),
     mCsteInc      (200),
     mRatioInc     (0.1),
     mNbProc       (aModeTestDeep ? 1 :  NbProcSys() ),
     mFileExp      ("PxBasic.tif"),
     mPyr          ("Pyr/")
{
    MMD_InitArgcArgv(argc,argv,2);

     ElInitArgMain
     (
        argc,argv,
        LArgMain()  << EAMC(mDir,"Directory of data")
                    << EAMC(mIm1,"Name Im1", eSAM_IsExistFile)
                    << EAMC(mIm2,"Name Im2", eSAM_IsExistFile),

        LArgMain()  << EAM(mExe,"Exe",true,"Execute Commands, else only print them (Def=true)", eSAM_IsBool)
                    << EAM(mDirTmpMEC,"DirMEC",true,"Name of output dir (Def=MEC-BasicEpip)")
                    << EAM(mZoom0,"Zoom0",true,"Initial zoom")
                    << EAM(mRegul,"Regul",true,"Regularisation coefficient")
                    << EAM(mSzW,"SzW",true,"Matching window size")
                    << EAM(mInc,"Inc",true,"Uncertaincy on pixel")
                    << EAM(mPxMoy,"PxMoy",true,"Average paralax")
                    << EAM(mNbProc,"NbP",true,"Number of process to allocate")
                    << EAM(mFileExp,"FileExp",true,"File 4 Exporting last result")
     );



     InitZoom0(mIm1);
     InitZoom0(mIm2);

     if (mModeTestDeep)
     {
          mPyr = mDirTmpMEC;  // Because it will be purged, muts be diff for each
     }

     // mDirTmpMEC = mDir + mDirTmpMEC;

     std::string aCom =       MMBinFile(MM3DStr)
                          +   std::string(" MICMAC ")
                          +  XML_MM_File("MM-Basic-Epip.xml ")
                          + " WorkDir="   +  mDir
                          + " +Im1="      +  mIm1
                          + " +Im2="      +  mIm2
                          + " +ZoomInit=" +  ToString(mZoom0)
						  + " +Regul="    +  ToString(mRegul)
						  + " +SzW="       +  ToString(mSzW)
                          + " +Inc="      +  ToString(mInc)
                          + " +Px1Moy="   +  ToString(mPxMoy)
                          + " +DirMEC="   +  mDirTmpMEC
                          + " +NbProc="   +  ToString(mNbProc)
                          + " +Pyr="   +     mPyr
                       ;


      if (mExe)
          System(aCom);
      else
      {
          std::cout << "COM=" << aCom << "\n";
      }

      // Save the result and purge the rest
      if (mModeTestDeep)
      {
          std::string aName =ImMatchFromLastN(mDir+mDirTmpMEC);
          ELISE_fp::MvFile(mDir+mDirTmpMEC+aName,mDir+mFileExp);

          std::string aNameMasq =ImMasqFromLastN(mDir+mDirTmpMEC);
          ELISE_fp::MvFile(mDir+mDirTmpMEC+aNameMasq,mDir+StdPrefix(mFileExp)+"_Masq.tif");

          ELISE_fp::PurgeDirRecursif(mDir+mDirTmpMEC);
          //ELISE_fp::PurgeDirRecursif(mDir+"Pyram");
          ELISE_fp::PurgeDir(mDir+mDirTmpMEC,true);
          //ELISE_fp::PurgeDir(mDir+"Pyram",true);
      }
}

int  CPP_MMBasicTestDeep (int argc,char ** argv)
{
   cAppliEpiBasic(argc,argv,true);
   return EXIT_SUCCESS;
}

int  CPP_MMBasic4IGeo (int argc,char ** argv)
{
   cAppliEpiBasic(argc,argv,false);
   return EXIT_SUCCESS;
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
