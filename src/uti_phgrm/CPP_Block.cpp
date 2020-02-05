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



int Blinis_main(int argc,char ** argv)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);

    std::string  aDir,aPat,aFullDir;
    std::string AeroIn;
    std::string KeyIm2Bloc;
    std::string aFileOut;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(KeyIm2Bloc,"Key for computing bloc structure")
                    << EAMC(aFileOut,"File for destination"),
        LArgMain()
    );

    if (!MMVisualMode)
    {
    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);

    cStructBlockCam aBlockName = StdGetFromPCP(Basic_XML_MM_File("Stereo-Bloc_Naming.xml"),StructBlockCam);
    aBlockName.KeyIm2TimeCam()  = KeyIm2Bloc;
    ELISE_fp::MkDirSvp(aDir+"Tmp-MM-Dir/");
    MakeFileXML(aBlockName,aDir+"Tmp-MM-Dir/Stereo-Bloc_Naming.xml");
    std::string aCom =   MM3dBinFile_quotes( "Apero" )
                       + Basic_XML_MM_File("Stereo-Apero-EstimBlock.xml ")
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" +PatternIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +Ori=") + AeroIn
                       + std::string(" +Out=") + aFileOut
                    ;


    std::cout << "Com = " << aCom << "\n";
    int aRes = System(aCom.c_str(),false,true,true);
    return aRes;

    }
    else return EXIT_SUCCESS;
}

/***********************************************************************/

class cAppli_Block
{
   public :
        CamStenope * CamS(const std::string & aName) {return  mICNM->StdCamStenOfNamesSVP(aName,mOriIn);}
        typedef std::pair<std::string,std::string>  t2Str;
        t2Str  TimGrp(const std::string & aName)
        {
           return mICNM->Assoc2To1(mKeyBl,aName,true);
        }
   protected :
        void Compile();

    // === Input parameters
        std::string mPatIm;  // Liste of all images
        std::string mOriIn;  // input orientation
        std::string mNameBlock;  // name of the block-blini

    // === Computed parameters
        std::string     mKeyBl;  // key for name compute of bloc
        cStructBlockCam mBlock;  // structure of the rigid bloc
        cElemAppliSetFile mEASF; // Structure to extract pattern + name manip
        const std::vector<std::string> *   mVN;    // list of images
        std::string                        mDir; // Directory of data
        cInterfChantierNameManipulateur *  mICNM;  // Name manip
};

void cAppli_Block::Compile()
{
        //  =========== Naming and name stuff ====
    mEASF.Init(mPatIm); // Compute the pattern
    mICNM = mEASF.mICNM;  // Extract name manip
    mVN = mEASF.SetIm();  // Extract list of input name
    mDir = mEASF.mDir;
    StdCorrecNameOrient(mOriIn,mDir);

        //  =========== Block stuff ====
    mBlock = StdGetFromPCP(mNameBlock,StructBlockCam); // Init Block
    mKeyBl =  mBlock.KeyIm2TimeCam(); // Store key for facilitie
}


class cAOFB_Im
{
     public :
        cAOFB_Im(const std::string & aName,CamStenope* aCamInit,const std::string& aNameCalib) :
          mName      (aName),
          mCamInit   (aCamInit),
          mNameCalib  (aNameCalib),
          mDone      (mCamInit!=nullptr),
          mR_Cam2W   (mDone ? mCamInit->Orient().inv()  : ElRotation3D::Id),
          mTime      (mDone ? mCamInit->GetTime() : 0.0)
        {
        }

        std::string   mName;     // Name of image
        CamStenope  * mCamInit;  // Possible initial camera (pose+Cal)
        std::string   mNameCalib; // Name of calibration
        bool          mDone;
        ElRotation3D  mR_Cam2W;  // Orientation Cam -> Word
        double        mTime;
};

class cAppli_OriFromBlock : public  cAppli_Block
{
    public :
        cAppli_OriFromBlock (int argc,char ** argv);
    private :
        // Extract Time Stamp and Grp from Compiled image
        t2Str TimGrp(cAOFB_Im * aPtrI) {return  cAppli_Block::TimGrp(aPtrI->mName);}

        std::string mNameCalib;  // input calibration (required for non oriented image)
        std::string mOriOut;     // Output orientation

        std::vector<cAOFB_Im*>             mVecI;  // list of "compiled"
        std::map<t2Str,cAOFB_Im*>          mMapI;  // map Time+Grp -> compiled image
    
        bool        mCPI;        // Calibration Per Image

};

cAppli_OriFromBlock::cAppli_OriFromBlock (int argc,char ** argv) :
   mCPI (false)
{
    MMD_InitArgcArgv(argc,argv);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mPatIm,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(mOriIn,"Input Orientation folder", eSAM_IsExistDirOri)
                    << EAMC(mNameCalib,"Calibration folder", eSAM_IsExistDirOri)
                    << EAMC(mNameBlock,"File for block")
                    << EAMC(mOriOut,"Output Orientation folder"),
        LArgMain()
                    << EAM(mCPI,"CPI",true,"Calib Per Im")
    );

    cAppli_Block::Compile();
    
    StdCorrecNameOrient(mNameCalib,mDir);
    // Create structur of "compiled" cam
    for (int aKIm=0 ; aKIm<int(mVN->size()) ; aKIm++)
    {
        const std::string & aName = (*mVN)[aKIm];
        // CamStenope * aCamIn = mICNM->StdCamStenOfNamesSVP(aName,mOriIn);  // Camera may not 
        CamStenope * aCamIn = CamS(aName);
        std::string  aNameCal = mICNM->StdNameCalib(mNameCalib,aName);
        mICNM->GlobCalibOfName(aName,mNameCalib,true);  // Calib should exist
        cAOFB_Im * aPtrI = new cAOFB_Im(aName,aCamIn,aNameCal);  // Create compiled
        mVecI.push_back(aPtrI);  // memo all compiled
        mMapI[TimGrp(aPtrI)] = aPtrI; // We will need to access to pose from Time & Group
        // std::pair<std::string,std::string> aPair=mICNM->Assoc2To1(mKeyBl,aPtrI->mName,true);
    }

    // Try to compile pose non existing
    for (const auto & aPtrI : mVecI)
    {
        // If cam init exits nothing to do
        if (! aPtrI->mCamInit)
        {
           int  aNbInit = 0;
           double aSumTime = 0.0;
           // Extract time stamp & name of this block
           t2Str aPair= TimGrp(aPtrI);
           std::string aNameTime = aPair.first; // get a time stamp
           std::string aNameGrp  = aPair.second;// get a cam name
           
           cParamOrientSHC * OrInBlock = POriFromBloc(mBlock,aNameGrp,false); // Extrac orientaion in block

           // In the case there is several head oriented, we store multiple solutio,
           std::vector<ElRotation3D>  aVecOrient;
           std::vector<double>        aVecPds;

           // Parse the block of camera
           for (const auto & aLiaison :  mBlock.LiaisonsSHC().Val().ParamOrientSHC())
           {
              std::string aNeighGrp = aLiaison.IdGrp();
              if (aNameGrp != aNeighGrp)  // No need to try to init on itself
              {
                 // Extract neighboor in block from time-stamp & group
                 cAOFB_Im* aNeigh = mMapI[t2Str(aNameTime,aNeighGrp)];
                 if (aNeigh && aNeigh->mCamInit) // If neighbooring exist and has orientation init
                 {
                    cParamOrientSHC * aOriNeigh = POriFromBloc(mBlock,aNeighGrp,false);

                    ElRotation3D aOri_Neigh2World =  aNeigh->mCamInit->Orient().inv(); // Monde->Neigh
                    ElRotation3D aOriBlock_This2Neigh = RotCam1ToCam2(*OrInBlock,*aOriNeigh);
                    ElRotation3D aOri_This2Word = aOri_Neigh2World * aOriBlock_This2Neigh;
                    aVecOrient.push_back(aOri_This2Word);
                    aVecPds.push_back(1.0);
                    aSumTime += aNeigh->mTime;
                    aNbInit++;
                    std::cout << aPtrI->mName << " From " << aNeigh->mName << "\n";
                 }
              }
           }
           if (aNbInit)
           {
              aPtrI->mR_Cam2W = AverRotation(aVecOrient,aVecPds);
              aPtrI->mDone = true;
              aPtrI->mTime = aSumTime / aNbInit;
           }
           else
           {
           }
        }
    }

    // Now export 
    for (const auto & aPtrI : mVecI)
    {
        if (aPtrI->mDone)
        {
            // Case calibration by image, we export directly the results
            //  std::string aNameFI =   mICNM->StdNameCalib(mNameCalib,aName);
            std::string aNameFI =   mICNM->StdNameCalib(mOriOut,aPtrI->mName);
            CamStenope  * aCamCal = mICNM->GlobCalibOfName(aPtrI->mName,mNameCalib,true);
            aCamCal->SetOrientation(aPtrI->mR_Cam2W.inv());
            aCamCal->SetTime(aPtrI->mTime);
            aCamCal->StdExport2File(mICNM,mOriOut,aPtrI->mName,mCPI ? "" : aNameFI);
            if (!mCPI)
            {
                std::string aNameCal =   mICNM->StdNameCalib(mNameCalib,aPtrI->mName);
                ELISE_fp::CpFile(aNameCal,aNameFI);
            }
        }
    }

    // cElemAppliSetFile anEASF(mPatIm);
}

int OrientFromBlock_main(int argc,char ** argv)
{
    cAppli_OriFromBlock anAppli(argc,argv);

    return EXIT_SUCCESS;
}


/***********************************************************************/

class cOneImBrion
{
    public :
    private :
        std::string  mName; 
        CamStenope   * mCalib;
};

class cAppli_Brion   // Block Rigid Initialisation des Orientation Normale
{
    public :
    private :
       cStructBlockCam  mBlock;
};


/***********************************************************/
/*                                                         */
/*               Graphe Stereopolis                        */
/*                                                         */
/***********************************************************/

class cGS_Cam;
class cGS_1BlC; //  1 Bloc Camera
class cGS_Appli;  //  Application


class cGS_Cam //  1 Camera
{
    public :
        cGS_Cam(CamStenope * aCamS, const std::string &aName, const std::string &aGrp) :
            mCamS (aCamS),
            mName (aName),
            mGrp  (aGrp)
        {
        }
        cGS_Cam(const cGS_Cam&) = delete;
        CamStenope * mCamS;  // Camera
        std::string  mName;  // Name Im
        std::string  mGrp;   // Groupe in the block
};


class cGS_1BlC //  1 Bloc Camera
{
    public :
      double DistLine(const cGS_1BlC & aBl2) const;
      double Time() const {return mCamC->mCamS->GetTime();}
      cGS_Cam * CamOfGrp(const std::string & aGrp) const;
      bool  ValidForCross(const cGS_1BlC&,const cGS_SectionCross &) const;

      cGS_1BlC(cGS_Appli&,const std::vector<std::string> &);
      cGS_1BlC(const cGS_1BlC&) = delete;

      cGS_Appli& mAppli;   // Application  
      std::vector<cGS_Cam *> mVCams;  // Vector of all cam/image
      cGS_Cam *              mCamC;   // Centrale camera (generaly INS)
      Pt3dr                  mP3;     // Center
      Pt2dr                  mP2;     // P for indextion in QT
      Pt2dr                  mV2;     // 2D speed
      Seg2d                  mSeg;    // line  PCur -> Next
      int                    mNum;    // Numerotation, can be usefull ?
      double                 mAbsCurv;  // Curvilinear abscisse
};

//extern Box2dr aBox;
//ElQT<cGS_1BlC * ,Pt2dr,Pt2dr (*)(cGS_1BlC *)> aQT([](cGS_1BlC *aBl){return aBl->mP2;},aBox,20,20);

// (aGetP1,aBox,20,20);



class cGS_Appli : public cAppli_Block //  Application
{
     public :
         cGS_Appli(int,char**);
         bool AddPair(std::string aS1,std::string aS2);
         const std::string & NameGrpC() const {return mParam.NameGrpC();}
     private :
         void AddAllBloc(const cGS_1BlC&,const cGS_1BlC &);
         double  mDistStd;
         typedef ElQT<cGS_1BlC * ,Pt2dr,Pt2dr (*)(cGS_1BlC *)> tQtSom;
         typedef ElQT<cGS_1BlC * ,Seg2d,Seg2d (*)(cGS_1BlC *)> tQtArc;

         cXml_ParamGraphStereopolis mParam;
         int                      mLevelMsg;
         int AdaptIndex(int aKBl) {return  ElMax(0,ElMin(mNbBloc,aKBl));}

         std::vector<cGS_1BlC *>  mVBlocs;
         int                      mNbBloc;
         std::set<t2Str>          mSetStr;
         tQtSom *                 mQtSom;
         tQtArc *                 mQtArc;
         cPlyCloud                mPlyCross;
         std::string              mNameSave;
};

    // ===========   cGS_1BlC =============

cGS_1BlC::cGS_1BlC(cGS_Appli& anAppli,const std::vector<std::string> & aVN) :
   mAppli (anAppli),
   mCamC  (nullptr)
{
    for (const auto  & aName : aVN)
    {
        // Compute new Cam with name & grp
        CamStenope * aCam = mAppli.CamS(aName);
        std::string  aNameGrp =  mAppli.TimGrp(aName).second;
        cGS_Cam * aGS_Cam = new cGS_Cam(aCam,aName,aNameGrp);
        // save it
        mVCams.push_back(aGS_Cam);
        // Chek if it is the central one
        if (aNameGrp==mAppli.NameGrpC())
           mCamC   = aGS_Cam;
    }
    if (mCamC)
    {
        // Compute localisation
        mP3 = mCamC->mCamS->PseudoOpticalCenter();
        mP2 = Pt2dr(mP3.x,mP3.y);
    }
}
cGS_Cam * cGS_1BlC::CamOfGrp(const std::string & aGrp) const
{
    for (const auto& aCam : mVCams)
        if (aCam->mGrp==aGrp)
           return aCam;
    return nullptr;
}

double cGS_1BlC::DistLine(const cGS_1BlC & aBl2) const
{
    SegComp aDr1(mSeg);
    return ElMax
           (
                aDr1.dist_droite(aBl2.mSeg.p0())
               ,aDr1.dist_droite(aBl2.mSeg.p1())
           );
}


bool  cGS_1BlC::ValidForCross(const cGS_1BlC& aBl2,const cGS_SectionCross & aSC) const
{
   // Handle only one way +  Avoid ajacence
   if (mNum>=aBl2.mNum-1) 
      return false;
   // If absc curv too close we are just in the same trajectory, not a cross
   if (ElAbs(mAbsCurv-aBl2.mAbsCurv) < aSC.DistCurvMin())
      return false;
   
   if (ElAbs(angle_de_droite_nor(mV2,aBl2.mV2))  < aSC.AngleMinSpeed())
      return false;

   if  (      (DistLine(aBl2) <aSC.DistMinTraj())
           || ( aBl2.DistLine(*this) <aSC.DistMinTraj())
       )
      return false;

    return true;
}


    // ==========   cGS_Appli ==========

bool cGS_Appli::AddPair(std::string aS1,std::string aS2)
{
    if (aS1==aS2) return false; // juste in case
    if (aS1>aS2) ElSwap(aS1,aS2);  // store only oneway

    return mSetStr.insert(t2Str(aS1,aS2)).second;
}

void cGS_Appli::AddAllBloc(const cGS_1BlC& aBl1,const cGS_1BlC & aBl2)
{
   const cGS_SectionCross&  aSC  = mParam.GS_SectionCross().Val();
   for (const auto & aN1 : aSC.ListCam())
   {
      for (const auto & aN2 : aSC.ListCam())
      {
          cGS_Cam * aCam1 =  aBl1.CamOfGrp(aN1);
          cGS_Cam * aCam2 =  aBl2.CamOfGrp(aN2);
          if (aCam1 && aCam2)
          {
             AddPair(aCam1->mName,aCam2->mName);
          }
      }
   }
}



cGS_Appli::cGS_Appli (int argc,char ** argv)  :
   mLevelMsg (1),
   mQtSom    (nullptr),
   mQtArc    (nullptr),
   mNameSave ("GrapheStereropolis.xml")
{
    bool mDoPlyCros = true;

    MMD_InitArgcArgv(argc,argv);

    std::string aNameParam;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mPatIm,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(mOriIn,"Input Orientation folder", eSAM_IsExistDirOri)
                    << EAMC(mNameBlock,"File for block")
                    << EAMC(aNameParam,"Name Grp Centrale Camera"),
        LArgMain()
                    << EAM(mLevelMsg,"LevMsg",true,"Level of Message, def=1")
                    << EAM(mNameSave,"Out",true,"Name 4 save, def=GrapheStereropolis.xml")
    );
    mParam = StdGetFromPCP(aNameParam,Xml_ParamGraphStereopolis);

    cAppli_Block::Compile();
    if (mLevelMsg>=1)  std::cout << "=== DONE READ ARGS\n";

    // Map  Name of Time => Vecteur of name images at this time
    std::map<std::string,std::vector<std::string> > aMapBl;

    // Group name by timing
    for (const auto & aName : *mVN)
    {
        t2Str  aTimG =  TimGrp(aName);
        aMapBl[aTimG.first].push_back(aName);
    }
    if (mLevelMsg>=1)  std::cout << "=== DONE COMPUTE Groups, NbInit=" << aMapBl.size() << "\n";

    // Compute bloc
    Pt2dr aPInf(1e20,1e20);
    Pt2dr aPSup(-1e20,-1e20);
    for (const auto & aBloc : aMapBl)
    {
        cGS_1BlC * aGSB = new cGS_1BlC(*this,aBloc.second);
        if (aGSB->mCamC!=nullptr)
        {
           mVBlocs.push_back(aGSB);
           aPInf = Inf(aPInf,aGSB->mP2);
           aPSup = Sup(aPSup,aGSB->mP2);
        }
    }

    mNbBloc = mVBlocs.size();
    if (mLevelMsg>=1)  std::cout << "=== DONE COMPUTE READ CAMS; NbBloc=" << mNbBloc << "\n";
    // Speed computation would fail if not enough blocs
    ELISE_ASSERT(mVBlocs.size()>=2,"Not enough bloc");

    // Sort by time
    std::sort
    (
        mVBlocs.begin(),
        mVBlocs.end(),
        [](const cGS_1BlC * aBl1,const cGS_1BlC * aBl2) 
        {
           return aBl1->Time() < aBl2->Time() ;
        }
    );
    if (mLevelMsg>=1)  std::cout << "=== DONE SORTED BY TIME\n";

    // Computation of speed, absic 
    for (int aKBl=0 ; aKBl<mNbBloc ; aKBl++)
    {
         cGS_1BlC & aCur  = *(mVBlocs.at(aKBl));
         cGS_1BlC & aPrec = *(mVBlocs.at(ElMax(0,aKBl-1)));
         cGS_1BlC & aNext = *(mVBlocs.at(ElMin(mNbBloc-1,aKBl+1)));

         aCur.mNum = aKBl;
         aCur.mV2 = (aNext.mP2-aPrec.mP2) / (aNext.Time()-aPrec.Time());
         // Compute abs curv
         if (aKBl==0)
         {
            aCur.mAbsCurv = 0.0;
         }
         else
         {
            aCur.mAbsCurv = aPrec.mAbsCurv + euclid(aCur.mP2-aPrec.mP2);
         }
         if (aKBl==mNbBloc-1)
         {
            // To have a non degenerat seg, in case it creates problem
            aCur.mSeg    = Seg2d(aCur.mP2,aCur.mP2+aCur.mV2*1e-5);
         }
         else
         {
            aCur.mSeg    = Seg2d(aCur.mP2,aNext.mP2);
         }
    }
    if (mLevelMsg>=1)  std::cout << "=== DONE COMPUTED SPEED\n";

    // Compute Quod Tree for spatial indexation
    {
       // Get a rough estimation of a standard dist using average between succesive position
       mDistStd = mVBlocs.back()->mAbsCurv / (mNbBloc-1);
       Pt2dr aRab(mDistStd,mDistStd);
       mQtSom= new tQtSom
                   (
                        [](cGS_1BlC *aBl){return aBl->mP2;},
                        Box2dr(aPInf-aRab,aPSup+aRab),
                        10,
                        mDistStd*5.0
                    );
       mQtArc= new tQtArc
                   (
                        [](cGS_1BlC *aBl){return aBl->mSeg;},
                        Box2dr(aPInf-aRab,aPSup+aRab),
                        10,
                        mDistStd*5.0
                    );

        // Put submit in Qt
        for (auto & aBl : mVBlocs)
            mQtSom->insert(aBl);
    }
    if (mLevelMsg>=1)  std::cout << "=== DONE COMPUTED Quod-Tree\n";
  
    int aNbLine = 0;
    //  Add successive from linear analysis, generally only same camera + block
    if (mParam.GS_SectionLinear().IsInit())
    {
       for (int aKBl1=0 ; aKBl1<mNbBloc ; aKBl1++)
       {
          cGS_1BlC & aGrp1 = *(mVBlocs.at(aKBl1));
          for (const auto & aSLin : mParam.GS_SectionLinear().Val().GS_OneLinear())
          {
             if ((aGrp1.mNum % aSLin.Period().Val())==0) //  Generally, only used when Delta ={0}
             {
                // Compute limits of index taking into account overfow
                int aKB2Min = AdaptIndex(aKBl1+aSLin.DeltaMin()); 
                int aKB2Max = AdaptIndex(aKBl1+aSLin.DeltaMax());
                for (int aKBl2=aKB2Min ; aKBl2<aKB2Max ; aKBl2++)
                {
                   cGS_1BlC & aGrp2 = *(mVBlocs.at(aKBl2));
                   // Parse pairs and get associated camera
                   for (const auto & aCple : aSLin.CpleGrp())
                   {
                      cGS_Cam * aCam1 = aGrp1.CamOfGrp(aCple.N1());
                      cGS_Cam * aCam2 = aGrp2.CamOfGrp(aCple.N2());
                      if (aCam1 && aCam2)  // They may not exist, bloc are not all full
                      {
                         if (AddPair(aCam1->mName,aCam2->mName))
                         {
                            aNbLine++;
                         }
                      }
                   }
                }
             }
          }
       }
    }
    if (mLevelMsg>=1)  std::cout << "=== DONE Added Linear\n";
    // Process crosses, may be important to limit drift
    int aNbCross = 0;
    if (mParam.GS_SectionCross().IsInit())
    {
        const cGS_SectionCross&  aSC  = mParam.GS_SectionCross().Val();
        for (auto & aBl1 : mVBlocs)
        {
           set<cGS_1BlC *> aSNeigh;
           mQtSom->RVoisins(aSNeigh,aBl1->mP2,aSC.DistMax());
           for (auto & aBl2 : aSNeigh)
           {
               if (aBl1->ValidForCross(*aBl2,aSC))
               {
                  AddAllBloc(*aBl1,*aBl2);
                  aNbCross ++;
                  if (mDoPlyCros)
                  {
                     Pt3di aColCr(255,128,0);
                     mPlyCross.AddSeg(aColCr,aBl1->mP3,aBl2->mP3,200);
                  }
               }
           }
        }
    }
    if (mDoPlyCros)
    {
       mPlyCross.PutFile("Cross.ply");
    }
    std::cout << "NB  VAL , FOR LINE: " <<  aNbLine  << " FOR CROSS: " << aNbCross << "\n";

    // Save to file
    cSauvegardeNamedRel aRel;
    for (const auto & aPair : mSetStr)
    {
        aRel.Cple().push_back(cCpleString(aPair.first,aPair.second));
    }
    MakeFileXML(aRel,mDir+mNameSave);
}

int GrapheStereopolis_main(int argc,char ** argv)
{
    cGS_Appli(argc,argv);

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
