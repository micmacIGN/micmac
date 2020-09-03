#ifndef _ELISE_BLOCK_CAM_H_
#define _ELISE_BLOCK_CAM_H_

/*
   This file, who should have exist far before, contains some devlopment common to Tapas & Campari
*/

class cAppli_Block
{
   public :
        // CamStenope * CamSOfName(const std::string & aName) {return  mICNM->StdCamStenOfNames(aName,mOriIn);}
        CamStenope * CamSOfName(const std::string & aName) {return  mICNM->StdCamStenOfNamesSVP(aName,mOriIn);} // Corrige une possible regression qui generait une erreur si non existe
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


class cGS_Cam;
class cGS_1BlC; //  1 Bloc Camera
class cGS_Appli;  //  Application


class cGS_Cam //  1 Camera
{
    public :
        cGS_Cam(CamStenope * aCamS, const std::string &aName, const std::string &aGrp,cGS_1BlC * aBl) :
            mCamS  (aCamS),
            mName  (aName),
            mGrp   (aGrp),
            mBlock (aBl)
        {
        }
        cGS_Cam(const cGS_Cam&) = delete;
        CamStenope * mCamS;  // Camera
        std::string  mName;  // Name Im
        std::string  mGrp;   // Groupe in the block
        cGS_1BlC *   mBlock;
};


class cGS_1BlC //  1 Bloc Camera
{
    public :
      double DistLine(const cGS_1BlC & aBl2) const;
      double Time() const {return mCamC->mCamS->GetTime();}
      cGS_Cam * CamOfGrp(const std::string & aGrp) const;
      bool  ValidForCross(const cGS_1BlC&,const cGS_SectionCross &) const;

      cGS_1BlC(cGS_Appli&,const std::string & aTimeId,const std::vector<std::string> &);
      cGS_1BlC(const cGS_1BlC&) = delete;

      cGS_Appli& mAppli;   // Application  
      std::string            mTimeId; // Identifier of timing
      std::vector<cGS_Cam *> mVCams;  // Vector of all cam/image
      cGS_Cam *              mCamC;   // Centrale camera (generaly INS)
      cGS_Cam *              mCamSec;   // Used for apply comparing 3 traj
      Pt3dr                  mP3;     // Center
      Pt2dr                  mP2;     // P for indextion in QT
      Pt2dr                  mV2;     // 2D speed
      Seg2d                  mSeg;    // line  PCur -> Next
      int                    mNum;    // Numerotation, can be usefull ?
      double                 mAbsCurv;  // Curvilinear abscisse
};

// Poor desing,  a huge class with several command, would be better to make some inheritance
// will see later

class cGS_Appli : public cAppli_Block //  Application
{
     public :
        typedef enum
        {
            eGraphe,
            eCheckRel,    // Check Relative Orientation
            eComputeBlini // Analyse Traj et Blini
        } eModeAppli;

         cGS_Appli(int,char**,eModeAppli);
         bool AddPair(std::string aS1,std::string aS2);
         const std::string & NameGrpC() const {return mNameGrpC;}
         cGS_Cam* & CamOfName(const std::string & aName) {return mDicoCam[aName];}
         virtual int Exe();
     protected :
         void  DoGraphe();
         void  DoCheckMesures();
         void  SauvRel();
         void AddAllBloc(const cGS_1BlC&,const cGS_1BlC &);
         double  mDistStd;
         typedef ElQT<cGS_1BlC * ,Pt2dr,Pt2dr (*)(cGS_1BlC *)> tQtSom;
         typedef ElQT<cGS_1BlC * ,Seg2d,Seg2d (*)(cGS_1BlC *)> tQtArc;

         eModeAppli                 mMode;
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
         int                      mNbPairByF;
         int                      mNbFile;
         cSauvegardeNamedRel      mRel;
         Pt2dr                    mPInf;
         Pt2dr                    mPSup;
         bool                     mDoPlyCros;
         std::string              mNameParam;
         std::string              mNameGrpC;

         cSetOfMesureAppuisFlottants mSMAF;
         std::string              mNamePointe;
         std::map<std::string,cGS_Cam*> mDicoCam;
         std::set<std::string>          mSetBloc;
         std::string                    mDirInc; // Folder 4 Inc readin
         std::string                    mNameMasq3D;  // For eliminating part of trajectory
         Pt2dr                          mSig0Incert;  // Sigma to transform Inc in weight
};




#endif //  _ELISE_BLOCK_CAM_H_
