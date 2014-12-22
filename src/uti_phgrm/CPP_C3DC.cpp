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
#include <algorithm>

// ffmpeg -i MVI_0001.MOV  -ss 30 -t 20 Im%5d_Ok.png

// Im*_Ok => OK
// Im*_Nl => Image Nulle (eliminee)


// =============  ::  ===================================



/*
Debut de pipeline statue :

mm3d MMByP Statue "IMGP703[0-5].*JPG" Ori-Ori-CalPerIm/ ZoomF=2 Masq3D=AperiCloud_Ori-CalPerIm_selectionInfo.xml Purge=true Do=APMCR

*/

class cAppli_C3DC : public cAppliWithSetImage
{
     public :
         cAppli_C3DC(int argc,char ** argv);
         void DoAll();

     private :

        void ExeCom(const std::string & aCom);

         void PipelineQuickMack();
         void PipelineStatue();

         std::string mStrType;
         eTypeMMByP  mType;
         std::string mOriFull;

         std::string mArgMasq3D;
         std::string mStrImOri;
         std::string mBaseComMMByP;
         std::string mBaseComEnv;
         std::string mComMerge;
         std::string mComCatPly;


    // Param opt
         bool        mTuning;
         bool        mPurge;
         bool        mPlyCoul;
         std::string mMergeOut;
         bool        mDoPoisson;
         std::string mMasq3D;
         int         mSzNorm;
         double      mDS;
         int         mZoomF;
         std::string mStrZ0ZF;
};

cAppli_C3DC::cAppli_C3DC(int argc,char ** argv) :
   cAppliWithSetImage  (argc-2,argv+2,TheFlagDev16BGray|TheFlagAcceptProblem),
   mTuning             (MPD_MM()),
   mPurge              (!MPD_MM()),
   mPlyCoul            (true),
   mMergeOut           ("C3DC.ply"),
   mSzNorm             (2),
   mDS                 (1.0),
   mZoomF              (1)
{
    if (argc<2)
    {
        ELISE_ASSERT(false,"No arg to C3CD");
    }
    mStrType = argv[1];
    StdReadEnum(mModeHelp,mType,mStrType,eNbTypeMMByP);


   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeMMByP,"e"))
                    << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(mOriFull,"Orientation", eSAM_IsExistDirOri),
        LArgMain()
                    << EAM(mMasq3D,"Masq3D",true,"3D masq for point selection")
                    << EAM(mMergeOut,"Out",true,"final result (Def=C3DC.ply)")
                    << EAM(mSzNorm,"SzNorm",true,"Sz of param for normal evaluation (<=0 if none, Def=2 mean 5x5) ")
                    << EAM(mPlyCoul,"PlyCoul",true,"Colour in ply ? Def = true")
                    << EAM(mTuning,"Tuning",true,"Will disappear on day ...")
                    << EAM(mPurge,"Purge",true,"Purge result, def=true")
                    << EAM(mDS,"DownScale",true,"DownScale of Final result, Def depenf of mode")
                    << EAM(mZoomF,"ZoomF",true,"Zoom final, Def depenf of mode")
   );

   if (!EAMIsInit(&mDS))
   {
      if (mType==eQuickMac) mDS = 2.0;
   }
   if (!EAMIsInit(&mZoomF))
   {
      if (mType==eQuickMac) mZoomF = 4;
      if (mType==eStatue)   mZoomF = 2;
   }




   if (! EAMIsInit(&mMergeOut)) mMergeOut = "C3DC_"+ mStrType + ".ply";

   mStrImOri =  BLANK + QUOTE(mEASF.mFullName) +  BLANK + Ori() + BLANK;
   mArgMasq3D = "";
   if (EAMIsInit(&mMasq3D))
      mArgMasq3D = std::string(" Masq3D=" + mMasq3D + BLANK) ;


  //=====================================

   mBaseComMMByP =    MM3dBinFile("MMByP ")
                   +  BLANK + mStrType
                   +  mStrImOri
                   +  mArgMasq3D;


  //=====================================
   mBaseComEnv =      MM3dBinFile("TestLib MMEnvlop ")
                   +  mStrImOri
                   +  std::string(" 16 ")  + ToString(4) + " " 
                   +  mArgMasq3D
                   +  std::string(" AutoPurge=") + ToString(mPurge)
                   +  " Out=" + mStrType
                 ;

/*
   if (mTuning)
   {
      mBaseComEnv = mBaseComEnv + " DoPlyDS=true";
   }
*/

  //=====================================

  mComMerge =      MM3dBinFile("TestLib  MergeCloud ")
                +  mStrImOri + " ModeMerge=" + mStrType
                +  " DownScale=" +ToString(mDS)
               ;

  if (mSzNorm>=0)
  {
     mComMerge = mComMerge + " SzNorm=" + ToString(1+2*mSzNorm);
  }

   mComMerge +=  " PlyCoul=" + ToString(mPlyCoul);
  //=====================================
  std::string aDirFusMM = (mType==eQuickMac) ? DirFusMMInit() : DirFusStatue() ;

   mComCatPly =  MM3dBinFile("MergePly ") + QUOTE( aDirFusMM + ".*Merge.*ply") + " Out="  + mMergeOut;

   mStrZ0ZF = " Zoom0=" + ToString(mZoomF) + " ZoomF=" + ToString(mZoomF);
}


void cAppli_C3DC::ExeCom(const std::string & aCom)
{

   std::cout << aCom << "\n\n";
   if (!mTuning) System(aCom);
}

void  cAppli_C3DC::PipelineQuickMack()
{
    ExeCom(mBaseComMMByP + " Do=AMP " + mStrZ0ZF);
    ExeCom(mBaseComEnv + " DownScale=" + ToString(mDS));
    ExeCom(mComMerge);
    ExeCom(mComCatPly);
}


void  cAppli_C3DC::PipelineStatue()
{
    ExeCom(mBaseComMMByP + " Purge=" + ToString(mPurge) + " Do=APMCR ZoomF=" + ToString(mZoomF)  );
    ExeCom(mBaseComEnv + " Glob=false");
    ExeCom(mBaseComMMByP + " Purge=" +  ToString(mPurge) + " Do=F " );
    ExeCom(mComMerge);
    ExeCom(mComCatPly);
/*
*/
}



void cAppli_C3DC::DoAll()
{
    switch (mType)
    {
         case eQuickMac :
              PipelineQuickMack();
         break;
 
         case eStatue :
              PipelineStatue();
         break;

         default :
              std::cout <<  mStrType  << " : not supported for now\n";
              ELISE_ASSERT(false,"Unsupported value in C3DC");
         break;
    }

}




int C3DC_main(int argc,char ** argv)
{
    if (0)
    {
       for (int aK=0 ; aK<argc ; aK++)
          std::cout << argv[aK] << "\n";
       std::cout << "=================================\n";
    }


    cAppli_C3DC anAppli(argc,argv);

    anAppli.DoAll();

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
