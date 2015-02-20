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
         cAppli_C3DC(int argc,char ** argv,bool DoMerge);
         void DoAll();

     private :

        void ExeCom(const std::string & aCom);

         void PipelineQuickMack();
         void PipelineStatue();
         void DoMergeAndPly();

         std::string mStrType;
         eTypeMMByP  mType;
         std::string mOriFull;

         std::string mArgMasq3D;
         std::string mStrImOri0;
         std::string mStrImOriApSec;
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
         bool        mDoMerge;
         cMMByImNM * mMMIN;
		 bool		 mUseGpu;
};

cAppli_C3DC::cAppli_C3DC(int argc,char ** argv,bool DoMerge) :
   cAppliWithSetImage  (argc-2,argv+2,TheFlagDev16BGray|TheFlagAcceptProblem),
   mTuning             (MPD_MM()),
   mPurge              (true),
   mPlyCoul            (true),
   mMergeOut           ("C3DC.ply"),
   mSzNorm             (2),
   mDS                 (1.0),
   mZoomF              (1),
   mDoMerge            (DoMerge),
   mMMIN               (0),
   mUseGpu			   (false)
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
                    << EAM(mTuning,"Tuning",true,"Will disappear one day ...")
                    << EAM(mPurge,"Purge",true,"Purge result, def=true")
                    << EAM(mDS,"DownScale",true,"DownScale of Final result, Def depends of mode")
                    << EAM(mZoomF,"ZoomF",true,"Zoom final, Def depends of mode")
					<< EAM(mUseGpu,"UseGpu",false,"Use cuda (Def=false)")
   );

   if (!EAMIsInit(&mDS))
   {
      // if (mType==eQuickMac) mDS = 2.0;
   }
   if (!EAMIsInit(&mZoomF))
   {
      if (mType==eBigMac)   mZoomF = 2;
      if (mType==eMicMac)   mZoomF = 4;
      if (mType==eQuickMac) mZoomF = 8;
      if (mType==eStatue)   mZoomF = 2;
   }




   if (! EAMIsInit(&mMergeOut)) mMergeOut = "C3DC_"+ mStrType + ".ply";

   mStrImOri0  =  BLANK + QUOTE(mEASF.mFullName) +  BLANK + Ori() + BLANK;
   mStrImOriApSec = BLANK +  DirAndPatFileOfImSec() +  BLANK + Ori() + BLANK;
   mArgMasq3D = "";
   if (EAMIsInit(&mMasq3D))
      mArgMasq3D = std::string(" Masq3D=" + mMasq3D + BLANK) ;


  //=====================================

   mBaseComMMByP =    MM3dBinFile("MMByP ")
                   +  BLANK + mStrType
                   +  mStrImOri0
				   +  mArgMasq3D
				   +  " UseGpu=" + ToString(mUseGpu);


  //=====================================
   mBaseComEnv =      MM3dBinFile("TestLib MMEnvlop ")
                   +  mStrImOriApSec
                   +  std::string(" 16 ")  + ToString(mZoomF) + " " 
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
                +  mStrImOri0 + " ModeMerge=" + mStrType
                +  " DownScale=" +ToString(mDS)
               ;

  if (mSzNorm>=0)
  {
     mComMerge = mComMerge + " SzNorm=" + ToString(1+2*mSzNorm);
  }

   mComMerge +=  " PlyCoul=" + ToString(mPlyCoul);

   mMMIN = cMMByImNM::ForGlobMerge(Dir(),mDS,mStrType);

  //=====================================

  std::string aDirFusMM = mMMIN->FullDir();

   mComCatPly =  MM3dBinFile("MergePly ") + QUOTE( aDirFusMM + ".*Merge.*ply") + " Out="  + mMergeOut;

   mStrZ0ZF = " Zoom0=" + ToString(mZoomF) + " ZoomF=" + ToString(mZoomF);
   mMMIN->SetOriOfEtat(mOri);
}


void cAppli_C3DC::ExeCom(const std::string & aCom)
{

   std::cout << aCom << "\n\n";
   if (!mTuning) System(aCom);
}

void cAppli_C3DC::DoMergeAndPly()
{
    mMMIN->AddistofName(mEASF.SetIm());
    if (mDoMerge)
    {
       ExeCom(mComMerge);
       ExeCom(mComCatPly);
    }
    if (MPD_MM())
    {
        // std::cout << "KKKKey " <<  mMMIN->KeyFileLON() << "\n";
    }
}

void  cAppli_C3DC::PipelineQuickMack()
{
    ExeCom(mBaseComMMByP + " Do=AMP " + mStrZ0ZF);
    ExeCom(mBaseComEnv + " DownScale=" + ToString(mDS));
    DoMergeAndPly();
}


void  cAppli_C3DC::PipelineStatue()
{
    ExeCom(mBaseComMMByP + " Purge=" + ToString(mPurge) + " Do=APMCR ZoomF=" + ToString(mZoomF)  );
    ExeCom(mBaseComEnv + " Glob=false");
    ExeCom(mBaseComMMByP + " Purge=" +  ToString(mPurge) + " Do=F " );
    DoMergeAndPly();
/*
*/
}



void cAppli_C3DC::DoAll()
{
    switch (mType)
    {
         case eBigMac :
         case eMicMac :
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
    cAppli_C3DC anAppli(argc,argv,true);
    anAppli.DoAll();
    return EXIT_SUCCESS;
}


int MPI_main(int argc,char ** argv)
{
    cAppli_C3DC anAppli(argc,argv,false);
    anAppli.DoAll();
    return EXIT_SUCCESS;
}


//====================================================

class cChantierFromMPI
{
     public :
       cChantierFromMPI(const std::string &,double aScale,const std::string & aPat);
         
       cMMByImNM *    mMMI;
       std::string    mOri;

       std::string    mStrPat; // Pattern : def  =>KeyFileLON
       std::string    mStrImOri0; // les initiales

       std::string    mStrType;
       std::string    mFullDirPIm;
       std::string    mFullDirChantier;

};



cChantierFromMPI::cChantierFromMPI(const std::string & aStr,double aScale,const std::string & aPat) :
    mMMI               (cMMByImNM::FromExistingDirOrMatch(aStr,false,aScale)),
    mOri               (mMMI->Etat().NameOri().ValWithDef("")),
    mStrPat            (aPat=="" ? mMMI->KeyFileLON() : aPat),
    mStrImOri0         (std::string(" ") + mStrPat + " " + mOri),
    mStrType           (mMMI->NameType()),
    mFullDirPIm        (mMMI->FullDir()),
    mFullDirChantier   (mMMI->DirGlob())
{
    if (mOri=="")
    {
        std::cout << "For Name=" << aStr  << " Scale=" << aScale << "\n";
        ELISE_ASSERT(false,"Reused PIMs was not correctly terminated");
    }
}


//====================================================

class cAppli_MPI2Ply
{
     public :
         cAppli_MPI2Ply(int argc,char ** argv);
         void DoAll();

     private :
         std::string mName;
         double      mDS;
         cChantierFromMPI * mCFPI;
         std::string mMergeOut;
         std::string mComNuageMerge;
         std::string mComCatPly;
         std::string mPat;
};


cAppli_MPI2Ply::cAppli_MPI2Ply(int argc,char ** argv):
    mDS (1.0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mName,"Dir or PMI-Type (QuickMac ....)"),
        LArgMain()
                    << EAM(mDS,"DS",true,"Dowscale, Def=1.0")
                    << EAM(mMergeOut,"Out",true,"Ply File Results")
                    << EAM(mPat,"Pat",true,"Pattern for selecting images (Def=All image in files)")
    );
     
    mCFPI = new cChantierFromMPI(mName,mDS,mPat);
 
    mComNuageMerge =       MM3dBinFile("TestLib  MergeCloud ")
                  +   mCFPI-> mStrImOri0
                  + " ModeMerge=" + mCFPI->mStrType
                  + " DownScale=" +ToString(mDS)
                  + " PlyCoul=true"
               ;

   std::string aPatPly = "Nuage-Merge-" +mPat + ".*.ply";


   if (! EAMIsInit(&mMergeOut)) mMergeOut =  mCFPI->mFullDirChantier+"C3DC_"+ mCFPI->mStrType + ".ply";
   mComCatPly =  MM3dBinFile("MergePly ") + QUOTE( mCFPI->mFullDirPIm + aPatPly) + " Out="  + mMergeOut;

}

void cAppli_MPI2Ply::DoAll()
{
   System(mComNuageMerge);
   System(mComCatPly);
}

int MPI2Ply_main(int argc,char ** argv)
{
    cAppli_MPI2Ply anAppli(argc,argv);
    anAppli.DoAll();
    return EXIT_SUCCESS;
}

//====================================================

class cAppli_MPI2Mnt
{
     public :
         cAppli_MPI2Mnt(int argc,char ** argv);
         void DoAll();

     private :
         void DoMTD();
         void DoBascule();

         std::string NameBascOfIm(const std::string &);


         std::string mName;
         double      mDS;
         cChantierFromMPI * mCFPI;
         cInterfChantierNameManipulateur * mICNM;
         const std::vector<std::string> *  mSetIm;
         std::string mDirApp;
         std::string mRep;
         std::string mPat;
         std::string mStrRep;
         std::string mDirMTD;
         std::string mDirBasc;

};

std::string cAppli_MPI2Mnt::NameBascOfIm(const std::string & aNameIm)
{
    return  "Bacule" + aNameIm + ".xml" ;
}

void cAppli_MPI2Mnt::DoAll()
{
    // DoMTD();
    DoBascule();
}

void cAppli_MPI2Mnt::DoBascule()
{

    std::list<std::string> aLCom;

       

    std::cout << "DIRAP " << mDirApp << " NBI " << mSetIm->size() << "\n";

    for (int aK=0 ; aK<int(mSetIm->size()) ; aK++)
    {
         std::string aNameIm =  (*mSetIm)[aK];
         std::string aCom =      MM3dBinFile("NuageBascule ")
                             +   mCFPI->mFullDirPIm+   "Nuage-Depth-"+ aNameIm +  ".xml" + BLANK
                             +   mDirApp+mDirMTD+ TheStringLastNuageMM + BLANK
                             +   mDirApp+mDirBasc + NameBascOfIm(aNameIm) + BLANK
                             +   "Paral=0 ";

           aLCom.push_back(aCom);
    }
    cEl_GPAO::DoComInParal(aLCom);

    // SMDM

// mm3d NuageBascule "P=PIMs-MicMac/Nuage-Depth-(.*).xml" TmpPMI2Mnt/NuageImProf_STD-MALT_Etape_5.xml  "c=Bascule/Basc-\$1.xml"  Paral=0

}


void cAppli_MPI2Mnt::DoMTD()
{
    std::string aCom =      MM3dBinFile("Malt ")
                          + std::string( " UrbanMNE ")
                          + std::string(" ") + mCFPI->mStrPat
                          + std::string(" ") + mCFPI->mMMI->GetOriOfEtat()
                          + mStrRep
                          + " DoMEC=0  Purge=true ZoomI=4 ZoomF=2  IncMax=1.0 " + 
                          + " DirMEC=" + mDirMTD
                       ;


   // std::cout << "COM = " << aCom << "\n";
   System(aCom);

 // mm3d Malt UrbanMNE ./%NKS-Set-OfFile@./PIMs-MicMac/PimsFile.xml  Ori-CalPerIm/ Repere=TheCyl.xml  DoMEC=0  Purge=true ZoomI=4 ZoomF=2 DirMEC=TmpPMI2Mnt IncMax=1.0
}

cAppli_MPI2Mnt::cAppli_MPI2Mnt(int argc,char ** argv) :
    mDS       (1.0),
    mDirMTD   ("PIMs-TmpMnt/"),
    mDirBasc   ("PIMs-TmpBasc/")
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mName,"Dir or PMI-Type (QuickMac ....)"),
        LArgMain()
                    << EAM(mDS,"DS",true,"Dowscale, Def=1.0")
                    << EAM(mRep,"Repere",true,"Repair (Euclid or Cyl)")
                    << EAM(mPat,"Pat",true,"Patter, def = all existing clouds")
   );
     
   mCFPI = new cChantierFromMPI(mName,mDS,mPat);
   mDirApp = mCFPI->mFullDirChantier;
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDirApp);
   mSetIm = mICNM->Get(mCFPI->mStrPat);

   ELISE_fp::MkDirSvp(mDirApp+mDirBasc);


   if (EAMIsInit(&mRep))
       mStrRep = " Repere=" + mRep;
  // cMMByImNM *    mMMI;



}


int MPI2Mnt_main(int argc,char ** argv)
{
    cAppli_MPI2Mnt anAppli(argc,argv);
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
