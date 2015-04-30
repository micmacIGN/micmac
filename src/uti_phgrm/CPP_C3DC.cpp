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

#if (ELISE_QT_VERSION >= 4)
    #include "general/visual_mainwindow.h"
#endif

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
         void PipelineEpip();
         void DoMergeAndPly();

         void ReadType(const std::string &aType);

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
         double          mDefCor;
         double          mZReg;
         std::string     mArgSupEpip;
         std::string     mFilePair;
};

cAppli_C3DC::cAppli_C3DC(int argc,char ** argv,bool DoMerge) :
   cAppliWithSetImage  (argc-2,argv+2,TheFlagDev16BGray|TheFlagAcceptProblem),
   mTuning             (MPD_MM()),
   mPurge              (true),
   mPlyCoul            (true),
   mMergeOut           ("C3DC.ply"),
   mSzNorm             (3),
   mDS                 (1.0),
   mZoomF              (1),
   mDoMerge            (DoMerge),
   mMMIN               (0),
   mUseGpu	       (false),
   mArgSupEpip         ("")
{


#if(ELISE_QT_VERSION >= 4)

    if (MMVisualMode)
    {
        QApplication app(argc, argv);

        LArgMain LAM;
        LAM << EAMC(mStrType,"Mode",eSAM_None,ListOfVal(eNbTypeMMByP));

        std::vector <cMMSpecArg> aVA = LAM.ExportMMSpec();

        cMMSpecArg aArg = aVA[0];

        list<string> liste_valeur_enum = listPossibleValues(aArg);

        QStringList items;
        list<string>::iterator it=liste_valeur_enum.begin();
        for (; it != liste_valeur_enum.end(); ++it)
            items << QString((*it).c_str());

        setStyleSheet(app);

        bool ok = false;
        int  defaultItem = 0;

        if(argc > 1)
            defaultItem = items.indexOf(QString(argv[1]));

        QInputDialog myDialog;
        QString item = myDialog.getItem(NULL, app.applicationName(),
                                             QString (aArg.Comment().c_str()), items, defaultItem, false, &ok);

        if (ok && !item.isEmpty())
            mStrType = item.toStdString();
        else
            return;

        ReadType(mStrType);
    }
    else
    {
        ELISE_ASSERT(argc >= 2,"Not enough arg");
        ReadType(argv[1]);
    }
#else
    ELISE_ASSERT(argc >= 2,"Not enough arg");
    ReadType(argv[1]);
#endif

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeMMByP))
                    << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(mOriFull,"Orientation", eSAM_IsExistDirOri),
        LArgMain()
                    << EAM(mMasq3D,"Masq3D",true,"3D masq for point selection",eSAM_IsExistFileRP)
                    << EAM(mMergeOut,"Out",true,"final result (Def=C3DC.ply)")
                    << EAM(mSzNorm,"SzNorm",true,"Sz of param for normal evaluation (<=0 if none, Def=2 means 5x5) ")
                    << EAM(mPlyCoul,"PlyCoul",true,"Colour in ply ? (Def = true)")
                    << EAM(mTuning,"Tuning",true,"Will disappear one day ...",eSAM_InternalUse)
                    << EAM(mPurge,"Purge",true,"Purge result, (Def=true)")
                    << EAM(mDS,"DownScale",true,"DownScale of Final result, Def depends on mode")
                    << EAM(mZoomF,"ZoomF",true,"Zoom final, Def depends on mode",eSAM_IsPowerOf2)
                    << EAM(mUseGpu,"UseGpu",false,"Use cuda (Def=false)")
                    << EAM(mDefCor,"DefCor",false,"Def correlation, context depend")
                    << EAM(mZReg,"ZReg",false,"Regularisation, context depend")
                    << EAM(mFilePair,"FilePair",false,"Expicit pairs of files (as in Tapioca)")
    );

   if (MMVisualMode) return;

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
       if (mType==eForest)   mZoomF = 4;
   }

   if (EAMIsInit(&mDefCor)) mArgSupEpip +=  " DefCor=" + ToString(mDefCor);
   if (EAMIsInit(&mZReg)) mArgSupEpip +=  " ZReg=" + ToString(mZReg);

   if (! EAMIsInit(&mMergeOut)) mMergeOut = "C3DC_"+ mStrType + ".ply";

   mStrImOri0  =  BLANK + QUOTE(mEASF.mFullName) +  BLANK + Ori() + BLANK;
   mStrImOriApSec = BLANK +  DirAndPatFileMMByP() +  BLANK + Ori() + BLANK;
   mArgMasq3D = "";
   if (EAMIsInit(&mMasq3D))
       mArgMasq3D = std::string(" Masq3D=" + mMasq3D + BLANK) ;


   //=====================================

   mBaseComMMByP =    MM3dBinFile("MMByP ")
           +  BLANK + mStrType
           +  mStrImOri0
           +  mArgMasq3D
           +  " UseGpu=" + ToString(mUseGpu);
   if (EAMIsInit(&mFilePair))
       mBaseComMMByP  += " FilePair=" + mFilePair;


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

void cAppli_C3DC::ReadType(const std::string & aType)
{
    mStrType = aType;
    StdReadEnum(mModeHelp,mType,mStrType,eNbTypeMMByP);
}

void  cAppli_C3DC::PipelineQuickMack()
{  
    ExeCom(mBaseComMMByP + " Do=AMP " + mStrZ0ZF);
    ExeCom(mBaseComEnv + " DownScale=" + ToString(mDS));
    DoMergeAndPly();
}


void  cAppli_C3DC::PipelineEpip()
{
    ExeCom(mBaseComMMByP + " Purge=" + ToString(mPurge) + " Do=APMCR ZoomF=" + ToString(mZoomF) + mArgSupEpip  );
    ExeCom(mBaseComEnv + " Glob=false");
    ExeCom(mBaseComMMByP + " Purge=" +  ToString(mPurge) + " Do=F " );
    DoMergeAndPly();
/*
*/
}



void cAppli_C3DC::DoAll()
{
    if (!MMVisualMode)
    {
        switch (mType)
        {
             case eBigMac:
             case eMicMac:
             case eQuickMac:
                  PipelineQuickMack();
             break;

             case eStatue:
             case eForest:
                  PipelineEpip();
             break;

             default :
                  std::cout <<  mStrType  << " : not supported for now\n";
                  ELISE_ASSERT(false,"Unsupported value in C3DC");
             break;
        }
    }

}


int C3DC_main(int argc,char ** argv)
{
    cAppli_C3DC anAppli(argc,argv,true);
    if (!MMVisualMode) anAppli.DoAll();
    return EXIT_SUCCESS;
}


int MPI_main(int argc,char ** argv)
{
    cAppli_C3DC anAppli(argc,argv,false);
    if (!MMVisualMode) anAppli.DoAll();
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
       std::string    mPatFilter; // Pattern : def  =>KeyFileLON
       std::string    mStrImOri0; // les initiales

       std::string    mStrType;
       std::string    mFullDirPIm;
       std::string    mFullDirChantier;

};



cChantierFromMPI::cChantierFromMPI(const std::string & aStr,double aScale,const std::string & aPat) :
    mMMI               (cMMByImNM::FromExistingDirOrMatch(aStr,false,aScale)),
    mOri               (mMMI->Etat().NameOri().ValWithDef("")),
    mStrPat            (aPat=="" ? mMMI->KeyFileLON() : aPat),
    mPatFilter         (aPat=="" ? ".*" : aPat),
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
        LArgMain()  << EAMC(mName,"Dir or PMI-Type (QuickMac ....)",eSAM_None,ListOfVal(eNbTypeMMByP)),
        LArgMain()
                    << EAM(mDS,"DS",true,"Dowscale, Def=1.0")
                    << EAM(mMergeOut,"Out",true,"Ply File Results")
                    << EAM(mPat,"Pat",true,"Pattern for selecting images (Def=All image in files)",eSAM_IsPatFile)
    );

    if(MMVisualMode) return;

    mCFPI = new cChantierFromMPI(mName,mDS,mPat);

    mComNuageMerge =       MM3dBinFile("TestLib  MergeCloud ")
                  +   mCFPI-> mStrImOri0
                  + " ModeMerge=" + mCFPI->mStrType
                  + " DownScale=" +ToString(mDS)
                  + " SzNorm=3"
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
    if (!MMVisualMode) anAppli.DoAll();
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
         void DoMerge();
         void DoOrtho();

         std::string NameBascOfIm(const std::string &);


         std::string mName;
         double      mDS;
         int         mDeZoom;
         cChantierFromMPI * mCFPI;
         cInterfChantierNameManipulateur * mICNM;
         const std::vector<std::string> *  mSetIm;
         std::string mDirApp;
         std::string mRep;
         std::string mPat;
         std::string mStrRep;
         std::string mDirMTD;
         std::string mDirBasc;
         std::string mNameMerge;
         std::string mNameOriMerge;
         std::string mNameOriMasq;
         std::string              mTargetGeom;
         cXML_ParamNuage3DMaille  mParamTarget;
         bool                     mRepIsAnam;
         bool                     mDoMnt;
         bool                     mDoOrtho;
		 std::string			  mMasqImGlob;
         bool                     mDebug;
};

std::string cAppli_MPI2Mnt::NameBascOfIm(const std::string & aNameIm)
{
    return  "Bascule" + aNameIm + ".xml" ;
}



void cAppli_MPI2Mnt::DoAll()
{
    if (mDoMnt && (!mDebug) ) DoMTD();
    mParamTarget =  StdGetFromSI(mTargetGeom,XML_ParamNuage3DMaille);
    if (mDoMnt && (!mDebug) ) DoBascule();
    if (mDoMnt ) DoMerge();


    //============== Generation d'un Ori
    cXML_ParamNuage3DMaille aN =   StdGetFromSI(mDirApp+mDirBasc +mNameMerge,XML_ParamNuage3DMaille);
    cFileOriMnt  aFOM = ToFOM(aN,true);
    MakeFileXML(aFOM,mDirApp+mDirBasc +mNameOriMasq);

    double aSR = aN.SsResolRef().Val();
    int aISR = round_ni(aSR);
    ELISE_ASSERT(ElAbs(aSR-aISR)<1e-7,"cAppli_MPI2Mnt::DoAll => ToFOM");
    aFOM.NombrePixels() =  aFOM.NombrePixels()* aISR;
    aFOM.ResolutionPlani() = aFOM.ResolutionPlani() / aISR;
    aFOM.ResolutionAlti() = aFOM.ResolutionAlti() / aISR;
    MakeFileXML(aFOM,mDirApp+mDirBasc +mNameOriMerge);
    //============== Generation d'un Ori

    if (mDoOrtho) DoOrtho();
}



void cAppli_MPI2Mnt::DoOrtho()
{
     std::string aCom =       MM3dBinFile("MICMAC ")
                         +    XML_MM_File("MM-PIMs2Ortho.xml") + BLANK
                         +    " +Pat=" +  mCFPI->mStrPat       + BLANK
                         +    " +Ori=" +  mCFPI->mOri                 + BLANK
                         +    " +DeZoom=" +ToString(mDeZoom)   + BLANK
                         +    " WorkDir=" + mDirApp
                      ;
	if (EAMIsInit(&mMasqImGlob)) aCom +=  " +UseGlobMasqPerIm=1  +GlobMasqPerIm="+mMasqImGlob;
	 
    if (EAMIsInit(&mRep))
    {
           aCom +=  " +Repere="+mRep;
           if (mRepIsAnam)
              aCom += " +RepereIsAnam=true";
           else
              aCom += " +RepereIsCart=true";
    }

    if (mDebug)
        std::cout << "COMORTHO= " << aCom << "\n";
    else
        System(aCom);

}

void cAppli_MPI2Mnt::DoMerge()
{

    std::string aCom =       MM3dBinFile("SMDM ")
                         +   QUOTE(mDirApp+mDirBasc + NameBascOfIm(mCFPI->mPatFilter)) + BLANK
                         +   "Out=" + mNameMerge + BLANK
                         // +   "TargetGeom=" +   mTargetGeom + BLANK

                      ;

    if (mDebug)
       std::cout << aCom << "\n";
    else
        System(aCom);

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
                             +   mTargetGeom + BLANK
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
                          + " ZoomF=" + ToString(mDeZoom)
                       ;

   System(aCom);
}

cAppli_MPI2Mnt::cAppli_MPI2Mnt(int argc,char ** argv) :
    mDS       (1.0),
    mDeZoom   (2),
    mDirMTD   ("PIMs-TmpMnt/"),
    mDirBasc   ("PIMs-TmpBasc/"),
    mNameMerge ("PIMs-Merged.xml"),
    mNameOriMerge ("PIMs-ZNUM-Merged.xml"),
    mNameOriMasq ("PIMs-Merged_Masq.xml"),
    mRepIsAnam   (false),
    mDoMnt       (true),
    mDoOrtho     (false),
    mMasqImGlob (""),
    mDebug       (false)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mName,"Dir or PMI-Type (QuickMac ....)",eSAM_None,ListOfVal(eNbTypeMMByP)),  //pas gerable par les vCommandes...
        LArgMain()
                    << EAM(mDS,"DS",true,"Downscale, Def=1.0")
                    << EAM(mRep,"Repere",true,"Repair (Euclid or Cyl)",eSAM_IsExistFileRP)
                    << EAM(mPat,"Pat",true,"Pattern, def = all existing clouds", eSAM_IsPatFile)
                    << EAM(mDoMnt,"DoMnt",true," Compute DTM , def=true (use false to return only ortho)")
                    << EAM(mDoOrtho,"DoOrtho",true,"Generate ortho photo,  def=false")
                    << EAM(mMasqImGlob,"MasqImGlob",true,"Global Masq for ortho: if used, give full name of masq (e.g. MasqGlob.tif) ")
                    << EAM(mDebug,"Debug",true,"Debug !!!",eSAM_InternalUse)
   );

   if (MMVisualMode) return;

   mCFPI = new cChantierFromMPI(mName,mDS,mPat);
   mDirApp = mCFPI->mFullDirChantier;
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDirApp);
   mSetIm = mICNM->Get(mCFPI->mStrPat);

   if (EAMIsInit(&mRep))
   {
       bool IsOrthoXCSte=false;
       bool IsAnamXCsteOfCart=false;
       mRepIsAnam = RepereIsAnam(mDirApp+mRep,IsOrthoXCSte,IsAnamXCsteOfCart);
   }

   ELISE_fp::MkDirSvp(mDirApp+mDirBasc);


   if (EAMIsInit(&mRep))
       mStrRep = " Repere=" + mRep;
  // cMMByImNM *    mMMI;


   mTargetGeom = mDirApp+mDirMTD+ TheStringLastNuageMM ;

}


int MPI2Mnt_main(int argc,char ** argv)
{
    cAppli_MPI2Mnt anAppli(argc,argv);
    if (!MMVisualMode) anAppli.DoAll();


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
