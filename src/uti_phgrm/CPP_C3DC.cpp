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

#if ELISE_QT
    #include "general/visual_mainwindow.h"
#endif

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
         
         std::string mSetHom;


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
         double      mDefCor;
         double      mZReg;
		 bool		 mExpTxt;
         std::string mArgSupEpip;
         std::string mFilePair;
         bool        mDebugMMByP;
         bool        mBin;
         bool        mExpImSec;
         Pt3dr       mOffsetPly;
         int         mSzW;
         bool        mNormByC;
         double      mTetaOpt;
         double      mResolTerrain;

};

cAppli_C3DC::cAppli_C3DC(int argc,char ** argv,bool DoMerge) :
   cAppliWithSetImage  (argc-2,argv+2,TheFlagDev16BGray|TheFlagAcceptProblem),
   mSetHom             (""),
   mTuning             (false),
   mPurge              (true),
   mPlyCoul            (true),
   mMergeOut           ("C3DC.ply"),
   mSzNorm             (3),
   mDS                 (1.0),
   mZoomF              (1),
   mDoMerge            (DoMerge),
   mMMIN               (0),
   mUseGpu	       (false),
   mExpTxt	       (false),
   mArgSupEpip         (""),
   mDebugMMByP         (false),
   mBin                (true),
   mExpImSec           (true),
   mSzW                (1),
   mNormByC            (false),
   mTetaOpt             (0.17)
{


#if ELISE_QT

    if (MMVisualMode)
    {
/*
// MPD : je comprends pas a quoi cela sert, et cela fait planter, sur les tests cela
// marche sans
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
*/
    }
    else
    {
        ELISE_ASSERT(argc >= 2,"Not enough arg");
        ReadType(argv[1]);
    }
#else 
    ELISE_ASSERT(argc >= 2,"Not enough arg");
    ReadType(argv[1]);
#endif // ELISE_QT
	
	//C3DC call case : general case
    if(mDoMerge)
    {
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
						<< EAM(mDefCor,"DefCor",true,"Def correlation, context depend")
						<< EAM(mZReg,"ZReg",true,"Regularisation, context depend")
						<< EAM(mExpTxt,"ExpTxt",false,"Use txt tie points for determining image pairs")
						<< EAM(mFilePair,"FilePair",true,"Explicit pairs of images (as in Tapioca)", eSAM_IsExistFileRP)
						<< EAM(mDebugMMByP,"DebugMMByP",true,"Debug MMByPair ...")
						<< EAM(mBin,"Bin",true,"Generate Binary or Ascii (Def=true, Binary)")
						<< EAM(mExpImSec,"ExpImSec",true,"Export Images Secondair, def=true")
						<< EAM(mOffsetPly,"OffsetPly",true,"Ply offset to overcome 32 bits problem")
						<< EAM(mSetHom,"SH",true,"Set of Hom, Def=\"\"")
                        << EAM(mNormByC,"NormByC",true,"Replace normal with camera position in ply (Def=false)")
                        << EAM(mTetaOpt,"TetaOpt",true,"For the choice of secondary images: Optimal angle of stereoscopy, in radian, def=0.17 (+or- 10 degree)")

		);
	}
	//Pims call case : no need to have all export .ply options in the command display (source of confusion)
	else
	{
		ElInitArgMain
		(
			argc,argv,
			LArgMain()  << EAMC(mStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeMMByP))
						<< EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)", eSAM_IsPatFile)
						<< EAMC(mOriFull,"Orientation", eSAM_IsExistDirOri),
			LArgMain()
						<< EAM(mMasq3D,"Masq3D",true,"3D masq for point selection",eSAM_IsExistFileRP)
						<< EAM(mTuning,"Tuning",true,"Will disappear one day ...",eSAM_InternalUse)
						<< EAM(mPurge,"Purge",true,"Purge result, (Def=true)")
						<< EAM(mZoomF,"ZoomF",true,"Zoom final, Def depends on mode",eSAM_IsPowerOf2)
						<< EAM(mUseGpu,"UseGpu",false,"Use cuda (Def=false)")
						<< EAM(mDefCor,"DefCor",true,"Def correlation, context depend")
						<< EAM(mZReg,"ZReg",true,"Regularisation, context depend")
						<< EAM(mExpTxt,"ExpTxt",false,"Use txt tie points for determining image pairs")
						<< EAM(mFilePair,"FilePair",true,"Explicit pairs of images (as in Tapioca)", eSAM_IsExistFileRP)
						<< EAM(mDebugMMByP,"DebugMMByP",true,"Debug MMByPair ...")
						<< EAM(mExpImSec,"ExpImSec",true,"Export Images Secondair, def=true")
						<< EAM(mOffsetPly,"OffsetPly",true,"Ply offset to overcome 32 bits problem")
                        << EAM(mSzW,"SzW",true,"Correlation Window Size (Def=1 means 3x3)")
                        << EAM(mSetHom,"SH",true,"Set of Hom, Def=\"\"")
                        << EAM(mTetaOpt,"TetaOpt",true,"For the choice of secondary images: Optimal angle of stereoscopy, in radian, def=0.17 (+or- 10 degree)")
                    );

	}
	

   if (MMVisualMode) return;

   if (mNormByC == true)
   {
       mSzNorm = -1;
   }

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

   //if (EAMIsInit(&mDefCor)) mArgSupEpip +=  " DefCor=" + ToString(mDefCor);
   //if (EAMIsInit(&mZReg)) mArgSupEpip +=  " ZReg=" + ToString(mZReg);
   //if (EAMIsInit(&mExpTxt)) mArgSupEpip +=  " ExpTxt=" + ToString(mExpTxt);
   if (! EAMIsInit(&mMergeOut)) mMergeOut = mEASF.mDir+"C3DC_"+ mStrType + ".ply";

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
           +  " UseGpu=" + ToString(mUseGpu)
           +  " ExpImSec=" + ToString(mExpImSec)
           +  " SH=" + mSetHom;

   if (EAMIsInit(&mTetaOpt)) mBaseComMMByP+=" TetaOpt=" + ToString(mTetaOpt) + " ";
   if (EAMIsInit(&mDefCor)) mBaseComMMByP +=  " DefCor=" + ToString(mDefCor);
   if (EAMIsInit(&mZReg)) mBaseComMMByP +=  " ZReg=" + ToString(mZReg);
   if (EAMIsInit(&mExpTxt)) mBaseComMMByP +=  " ExpTxt=" + ToString(mExpTxt);

   if (mDebugMMByP)
      mBaseComMMByP = mBaseComMMByP + " DebugMMByP=true";

   if (EAMIsInit(&mFilePair))
       mBaseComMMByP  += " FilePair=" + mFilePair;


   //=====================================
   mBaseComEnv =      MM3dBinFile("TestLib MMEnvlop ")
           +  mStrImOriApSec
           +  std::string(" 16 ")  + ToString(mZoomF) + " "
           +  mArgMasq3D
           +  std::string(" AutoPurge=") + ToString(mPurge)
           +  " Out=" + mStrType
           +  " SzW=" + ToString(mSzW)
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
           +  " SH=" + mSetHom
           + " NormByC=" + (mNormByC ? "true" : "false")
           ;

   if (EAMIsInit(&mOffsetPly))
   {
        mComMerge = mComMerge + " OffsetPly=" + ToString(mOffsetPly);
   }

   if (mSzNorm>=0)
   {
       mComMerge = mComMerge + " SzNorm=" + ToString(1+2*mSzNorm);
   }

   mComMerge +=  " PlyCoul=" + ToString(mPlyCoul);

   mMMIN = cMMByImNM::ForGlobMerge(Dir(),mDS,mStrType);

/*
   if (MPD_MM())
   {
       std::cout << "TESTING EXPORT PREP CARVING \n";


       getchar();
       exit(0);
   }
*/

   //=====================================

   std::string aDirFusMM = mMMIN->FullDir();
   
   if(mBin)
		mComCatPly =  MM3dBinFile("MergePly ") + QUOTE( aDirFusMM + ".*Merge.*ply") + " Out="  + mMergeOut;
   else
		mComCatPly =  MM3dBinFile("MergePly ") + QUOTE( aDirFusMM + ".*Merge.*ply") + " Out="  + mMergeOut + " Bin=0";

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
    ExeCom(mBaseComMMByP + " Do=AMP " + mStrZ0ZF + " ExpTxt=" + ToString(mExpTxt));
    ExeCom(mBaseComEnv + " DownScale=" + ToString(mDS));
    DoMergeAndPly();
}


void  cAppli_C3DC::PipelineEpip()
{
    ExeCom(mBaseComMMByP + " Purge=" + ToString(mPurge) + " Do=APMCR ZoomF=" + ToString(mZoomF) + mArgSupEpip  );
    if (mDebugMMByP) 
    {
       exit(EXIT_SUCCESS);
    }


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
       std::string    mCFPIStrImOri0; // les initiales

       std::string    mStrType;
       std::string    mFullDirPIm;
       std::string    mFullDirChantier;

};



cChantierFromMPI::cChantierFromMPI(const std::string & aStr,double aScale,const std::string & aPat) :
    mMMI               (cMMByImNM::FromExistingDirOrMatch(aStr,false,aScale,"./",true)),
    mOri               (mMMI->Etat().NameOri().ValWithDef("")),
    mStrPat            (aPat=="" ? mMMI->KeyFileLON() : aPat),
    mPatFilter         (aPat=="" ? ".*" : aPat),
    mCFPIStrImOri0     (std::string(" ") + mStrPat + " " + mOri),
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
         bool 		 mBin;
         Pt3dr       mOffsetPly;
	 bool DoublePrec = false;
         bool        mDebug;

};


cAppli_MPI2Ply::cAppli_MPI2Ply(int argc,char ** argv):
    mDS (1.0),
    mDebug(0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mName,"Dir or PMI-Type (QuickMac ....)",eSAM_None,ListOfVal(eNbTypeMMByP)),
        LArgMain()
                    << EAM(mDS,"DS",true,"Dowscale, Def=1.0")
                    << EAM(mMergeOut,"Out",true,"Ply File Results")
                    << EAM(mPat,"Pat",true,"Pattern for selecting images (Def=All image in files)",eSAM_IsPatFile)
                    << EAM(mOffsetPly,"OffsetPly",true,"Ply offset to overcome 32 bits problem")
                    << EAM(DoublePrec,"64B",true,"To generate 64 Bits ply, Def=false, WARN = do not work properly with meshlab or cloud compare")
                    << EAM(mDebug,"Debug",true,"debug mode, def false")
    );

    if(MMVisualMode) return;

    mCFPI = new cChantierFromMPI(mName,mDS,mPat);

    mComNuageMerge =       MM3dBinFile("TestLib  MergeCloud ")
                  +   mCFPI-> mCFPIStrImOri0
                  + " ModeMerge=" + mCFPI->mStrType
                  + " DownScale=" +ToString(mDS)
                  + " SzNorm=3"
                  + " PlyCoul=true"

               ;
   if (EAMIsInit(&mOffsetPly))
   {
        mComNuageMerge = mComNuageMerge + " OffsetPly=" + ToString(mOffsetPly);
   }
	
   if (DoublePrec)
   {
       mComNuageMerge = mComNuageMerge + " 64B=true";
   }
   else
   {
      mComNuageMerge = mComNuageMerge + " 64B=false";
   }

   std::string aPatPly = "Nuage-Merge-" +mPat + ".*.ply";


   if (! EAMIsInit(&mMergeOut)) mMergeOut =  mCFPI->mFullDirChantier+"C3DC_"+ mCFPI->mStrType + ".ply";
   mComCatPly =  MM3dBinFile("MergePly ") + QUOTE( mCFPI->mFullDirPIm + aPatPly) + " Out="  + mMergeOut;
}

void cAppli_MPI2Ply::DoAll()
{
if (mDebug)
{
   std::cout <<  "-----DEBUG MODE-----";
   std::cout <<    mCFPI-> mCFPIStrImOri0 << "\n";
   std::cout <<  "cAppli_MPI2Ply::DoAllcAppli_MPI2Ply::DoAll \n\n";
   std::cout <<  mComNuageMerge << "\n\n";
   std::cout <<  mComCatPly << "\n";
   std::cout <<  "Type something to restart processing..";
   getchar();
}
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

         std::string NameBascOfIm(const std::string &,bool ModePattern);


         std::string mName;
         double      mDS;
         int         mDeZoom;
         double      mResolIm;

         cChantierFromMPI * mCFPI;
         cInterfChantierNameManipulateur * mICNM;
         const std::vector<std::string> *  mSetIm;
         std::string mDirApp;
         std::string mRep;
         std::string mPat;
         std::string mStrRep;
         std::string mDirMTD;
         std::string mDirOrtho;
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
         bool                     mPurge;
         bool        mUseTA;
         void ExeCom(const std::string & aCom);
         double      mZReg;
         double      mSeuilE;
         double      mResolTerrain;
};

void cAppli_MPI2Mnt::ExeCom(const std::string & aCom)
{
   if (mDebug)
      std::cout << aCom << "\n\n";
   else
      System(aCom);
}

std::string cAppli_MPI2Mnt::NameBascOfIm(const std::string & aNameIm,bool ModePattern)
{
    return  "Bascule" + aNameIm + (ModePattern ? ".*" : "") + ".xml" ;
}



void cAppli_MPI2Mnt::DoAll()
{
    if (mDoMnt  ) DoMTD();
    mParamTarget =  StdGetFromSI(mTargetGeom,XML_ParamNuage3DMaille);
    if (mDoMnt  ) DoBascule();
    if (mDoMnt ) DoMerge();


    //============== Generation d'un Ori

    cXML_ParamNuage3DMaille aN =   StdGetFromSI(mDirApp+mDirBasc +mNameMerge,XML_ParamNuage3DMaille);

    if (mDebug)
    {
       std::cout  << "NUAGE MERGE " << mDirApp+mDirBasc +mNameMerge << "\n";
    }
    cFileOriMnt  aFOM = ToFOM(aN,true);
    if (mDebug) std::cout  << "toto\n";
    MakeFileXML(aFOM,mDirApp+mDirBasc +mNameOriMasq);

    double aSR = aN.SsResolRef().Val();
    int aISR = round_ni(aSR);

    ELISE_ASSERT(ElAbs(aSR-aISR)<1e-7,"cAppli_MPI2Mnt::DoAll => ToFOM");
    aFOM.NombrePixels() =  aFOM.NombrePixels()* aISR;
    aFOM.ResolutionPlani() = aFOM.ResolutionPlani() / aISR;
    // aFOM.ResolutionAlti() = aFOM.ResolutionAlti() / aISR;  MPD croit c'est inutile ???
    MakeFileXML(aFOM,mDirApp+mDirBasc +mNameOriMerge);
    //============== Generation d'un Ori

    if (mDoOrtho) DoOrtho();
}



void cAppli_MPI2Mnt::DoOrtho()
{
     std::string aCom =       MM3dBinFile("MICMAC ")
                         +    XML_MM_File("MM-PIMs2Ortho.xml") + BLANK
                         +    " +Pat=" + PATTERN_QUOTE(mCFPI->mStrPat) + BLANK
                         +    " +Ori=" +  mCFPI->mOri                 + BLANK
                         +    " +DeZoom=" +ToString(mDeZoom)   + BLANK
                         +    " WorkDir=" + mDirApp
                         +    " +DirOrthoF=" + mDirOrtho
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
/*
if (MPD_MM())
{
   std::cout << "EExxxxxxxxxxxxxxxxxxxxxxxxxxxitt \n";
   std::cout << aCom << "\n";
   exit(EXIT_SUCCESS);
}
*/

    ExeCom(aCom);

}

void cAppli_MPI2Mnt::DoMerge()
{

    std::string aCom =       MM3dBinFile("SMDM ")
                         +   QUOTE(mDirApp+mDirBasc + NameBascOfIm(mCFPI->mPatFilter,true)) + BLANK
                         +   "Out=" + mNameMerge + BLANK
                         // +   "TargetGeom=" +   mTargetGeom + BLANK

                      ;

        ExeCom(aCom);

}


void cAppli_MPI2Mnt::DoBascule()
{

    std::list<std::string> aLCom;



    // std::cout << "DIRAP " << mDirApp << " NBI " << mSetIm->size() << "\n";

    for (int aK=0 ; aK<int(mSetIm->size()) ; aK++)
    {
         std::string aNameIm =  (*mSetIm)[aK];
         std::string aNameBascInput = mCFPI->mFullDirPIm+   "Nuage-Depth-"+ aNameIm +  ".xml";

         if (ELISE_fp::exist_file(aNameBascInput))
         {
             std::string aCom =      MM3dBinFile("NuageBascule ")
                             // +   mCFPI->mFullDirPIm+   "Nuage-Depth-"+ aNameIm +  ".xml" + BLANK
                             +   aNameBascInput + BLANK
                             +   mTargetGeom + BLANK
                             +   mDirApp+mDirBasc + NameBascOfIm(aNameIm,false) + BLANK
                             +   "Paral=0 ";

              if (EAMIsInit(&mSeuilE))
                  aCom = aCom+ " SeuilE=" + ToString(mSeuilE) + std::string(" ");
               aLCom.push_back(aCom);
               if (mDebug &&(aK<2)) 
                  std::cout << aCom << "\n\n";
          }
    }
    if (mDebug)
    {
    }
    else
    {
       cEl_GPAO::DoComInParal(aLCom);
    }

    // SMDM

// mm3d NuageBascule "P=PIMs-MicMac/Nuage-Depth-(.*).xml" TmpPMI2Mnt/NuageImProf_STD-MALT_Etape_5.xml  "c=Bascule/Basc-\$1.xml"  Paral=0

}


void cAppli_MPI2Mnt::DoMTD()
{
    std::string aCom =      MM3dBinFile("Malt ")
                          + std::string( " UrbanMNE ")
                          + PATTERN_QUOTE(mCFPI->mStrPat)
                          + std::string(" ") + mCFPI->mMMI->GetOriOfEtat()
                          + mStrRep
                          + " DoMEC=0 Purge=true "
                          + " ZoomI=" + ToString(mDeZoom * 2)
                          + " ZoomF=" + ToString(mDeZoom)
                          + " IncMax=1.0 "
                          + " DirMEC=" + mDirMTD
                          + " UseTA=" + ToString(mUseTA)
                          + " ZoomF=" + ToString(mDeZoom)
                          + " RRI=" + ToString(mDeZoom *  mResolIm)
                          + " Regul=" + ToString(mZReg)
                          + " NbVI=2" 
                          + " EZA=1 "
                       ;
    if (mResolTerrain > 0. )
        aCom += " ResolTerrain=" + ToString(mResolTerrain);
    
    ExeCom(aCom);

/*
if (MPD_MM())
{
    std::cout << aCom << "\n";
    getchar();
}
*/

}

cAppli_MPI2Mnt::cAppli_MPI2Mnt(int argc,char ** argv) :
    mDS       (1.0),
    mDeZoom   (2),
    mResolIm  (1),
    mDirMTD   ("PIMs-TmpMnt/"),
    mDirOrtho  ("PIMs-ORTHO/"),
    mDirBasc   ("PIMs-TmpBasc/"),
    mNameMerge ("PIMs-Merged.xml"),
    mNameOriMerge ("PIMs-ZNUM-Merged.xml"),
    mNameOriMasq ("PIMs-Merged_Masq.xml"),
    mRepIsAnam   (false),
    mDoMnt       (true),
    mDoOrtho     (false),
    mMasqImGlob (""),
    mDebug       (false),
    mPurge       (true),
    mUseTA       (false),
    mZReg        (0.02),
    mSeuilE      (5.0),
    mResolTerrain(0.0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mName,"Dir or PIM-Type (QuickMac ....)",eSAM_None,ListOfVal(eNbTypeMMByP)),  //pas gerable par les vCommandes...
        LArgMain()
                    << EAM(mDS,"DS",true,"Downscale, Def=1.0")
                    << EAM(mZReg,"ZReg",true,"Regularisation, context depend")
                    << EAM(mRep,"Repere",true,"Repair (Euclid or Cyl)",eSAM_IsExistFileRP)
                    << EAM(mPat,"Pat",true,"Pattern, def = all existing clouds", eSAM_IsPatFile)
                    << EAM(mDoMnt,"DoMnt",true," Compute DTM , def=true (use false to return only ortho)")
                    << EAM(mDoOrtho,"DoOrtho",true,"Generate ortho photo,  def=false")
                    << EAM(mMasqImGlob,"MasqImGlob",true,"Global Masq for ortho: if used, give full name of masq (e.g. MasqGlob.tif) ",eSAM_IsExistFileRP)
                    << EAM(mDebug,"Debug",true,"Debug !!!",eSAM_InternalUse)
                    << EAM(mUseTA,"UseTA",true,"Use TA as filter when exist (Def=false)",eSAM_InternalUse)
                    << EAM(mResolIm,"RI",true,"Resol Im, def=1 ",eSAM_InternalUse)
                    << EAM(mSeuilE,"SeuilE",true,"Seuil d'etirement des triangle, Def=5")
                    << EAM(mResolTerrain,"ResolTerrain",true,"Ground Resol (Def automatically computed)", eSAM_NoInit)
                    << EAM(mDeZoom, "ZoomF", true, "ZoomF, Def=2")
                    << EAM(mDirMTD, "DirMTD", true, "Subdirectory where the temporary results will be stored, Def=PIMs-TmpMnt/")
                    << EAM(mDirOrtho, "DirOrtho", true, "Subdirectory for ortho images, Def=PIMs-ORTHO/")
                    << EAM(mDirBasc, "DirBasc", true, "Subdirectory for surface model, Def=PIMs-TmpBasc/")
                    << EAM(mNameMerge, "NameMerge", true, "BaseName of the surface model (*.xml), Def=PIMs-Merged.xml")
                    << EAM(mDebug, "Debug", true, "Debug mode, def false")
   );

   mResolIm  /= mDeZoom;

   if (mDoOrtho && (!EAMIsInit(&mDoMnt))) mDoMnt = mDoOrtho;

   if (MMVisualMode) return;


   mCFPI = new cChantierFromMPI(mName,mDS,mPat);
   mDirApp = mCFPI->mFullDirChantier;
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDirApp);
   mSetIm = mICNM->Get(mCFPI->mStrPat);

// Probleme d'incoherence et pas purgee !!!
   if ((mPurge) && (!mDebug))
   {
      ELISE_fp::PurgeDirRecursif(mDirApp+mDirMTD);
      ELISE_fp::PurgeDirRecursif(mDirApp+mDirBasc);
   }

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
    MMD_InitArgcArgv(argc,argv);

    cAppli_MPI2Mnt anAppli(argc,argv);
    if (!MMVisualMode) anAppli.DoAll();

    // MPD : pourquoi 2 fois  ??  if (!MMVisualMode) anAppli.DoAll();

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
