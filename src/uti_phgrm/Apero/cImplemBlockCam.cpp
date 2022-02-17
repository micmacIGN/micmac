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
#include "Apero.h"


// ModifDIST ==> Tag Provisoire des modifs a rajouter si dist

// Return the cParamOrientSHC of a given name
cParamOrientSHC * POriFromBloc(cStructBlockCam & aBloc,const std::string & aName,bool SVP)
{
    for ( auto & aPOS : aBloc.ParamOrientSHC())
    {
        if (aPOS.IdGrp() == aName)
           return & aPOS;
    }
    if (!SVP)
    {
        ELISE_ASSERT(false ,"Cannot get POriFromBloc");
    }
    return nullptr;
}
 
// Return the Rotation that transformate from Cam Coord to Block coordinates (in fact coord of "first" cam)
ElRotation3D  RotCamToBlock(const cParamOrientSHC & aPOS)
{
    return  ElRotation3D(aPOS.Vecteur(),ImportMat(aPOS.Rot()),true);
}

// Return the Rotation that transformate from Cam1 Coord to Cam2 Coord
ElRotation3D  RotCam1ToCam2(const cParamOrientSHC & aPOS1,const cParamOrientSHC & aPOS2)
{
    return  RotCamToBlock(aPOS2).inv() * RotCamToBlock(aPOS1);
}

/****************************************************************************/
/*                                                                          */
/*        Rigid Block, distance equation                                    */
/*                                                                          */
/****************************************************************************/

#define NUMPIVOT 0 // Numero , arbirtraire, de la camera "pivot" dans un bloc

// IBC : suffix for Implement Block Cam

//class cIBC_ImsOneTime;  //  regroupe les pose acquise au meme temps T
//class cIBC_OneCam;      //  contient les info partagee par la meme tete de camera
                        //  par exemple la rotation (ou le point) inconnue du bloc de camera


class cImplemBlockCam;  //  contient toutes les infos  relatives au meme bloc rigide
class cEqObsBlockCam ;  //  implemante l'equation  Ri Rj(-1) = Li Lj(-1)
class cEqOBCDistRef;    //  implemante la conservation des distances par rapport a une reference, a une similitude pres
// cStructBlockCam      //  XML structure du bloc  camera
// cLiaisonsSHC         //  XML structure, incluse dans cStructBlockCam, contient dans la geometrie
// cParamOrientSHC       //  XML structure, incluse dans cLiaisonsSHC, contient la geometrie d'une camera
// cBlockGlobalBundle   //  XML structure de ParamApero.xml,  indique le rappel / a une calibration du bloc





extern bool AllowUnsortedVarIn_SetMappingCur;

#define HandleGL 1

/*     ==== MODELISATION MATHEMATIQUE-1 ====================
  
   Pour les notations on reprend la terminologie LR (Left-Right) et Time (date de prise de vue)

      Block Inc :
          LinkL  LinkR ...

      Pour un bloc a N Camera, il y a N rotation inconnue (donc une de plus que necessaire) qui lie la camera a
   un systeme de reference; en pratique le systeme de reference  se trouve etre de la "premiere" camera (c'est ce qui
   est fait lors de l'estimation. c'est ce qui est maintenu pour lever l'arbitraire des equations, mais a part cela
   la "premiere" camera ne joue aucun role particulier);

      Les valeurs stockees dans LinkL (t.q. stockees dans LiaisonsSHC, voir cImplemBlockCam.cpp  CONV-ORI)

          LinkL =   L  -> Ref
          LinkR =   R  -> Ref

      Time i : CamLi  CamRi   ...  ;  CamLi : Li  -> Monde
      Time j : CamLj  CamRj   ...
      Time k : CamLk  CamRk   ...


       Les equation sont du type :

          L to R =      CamRi-1  CamLi   =   CamRk -1 CamLk , qqs k,i (et L,R)

     On ecrit cela 
 
             CamRi-1  CamLi = LinkR-1 LinkL  ,  qqs i, L , R  (Eq1) 

*/

/*
     Soit L = (Tl,Wl)  avec WL matrice rotation et Tl centre de projection   L(P) = Wl* P + Tl  (Cam->Monde)
     Et R-1 (P) = Wr-1*P - Wr-1 * Tr = tWr * P - Tr  ;   R-1 = (Wr-1,- Wr-1* P)
    

      CamRi-1  CamLi  = (Wr-1,- Wr-1* Pr) X (Wl,Pl) = (Wr-1*Wl,-  Pr +  Wr-1* Pl)
*/







// Compute the parameter of the transformation of Point in L coordinate
// to point in R coordinates
void CalcParamEqRel
     (
          Pt3d<Fonc_Num> & aTr, 
          ElMatrix<Fonc_Num> & aMat, 
          cRotationFormelle & aRotR,
          cRotationFormelle & aRotL,
          int                 aRNumGl,
          int                 aLNumGl
      )
{
#if (HandleGL != 1)
    ELISE_ASSERT(! aRotR.IsGL(),"Guimbal lock in Eq Rig still unsupported");
    ELISE_ASSERT(! aRotL.IsGL(),"Guimbal lock in Eq Rig still unsupported");
    aRNumGl=-1;
    aLNumGl=-1;
#endif

    ElMatrix<Fonc_Num> aRMat = aRotR.MatFGLComplete(aRNumGl);
    ElMatrix<Fonc_Num> aLMat = aRotL.MatFGLComplete(aLNumGl);
    ElMatrix<Fonc_Num> aRMatInv = aRMat.transpose();

    //  L to R = CamRi-1  CamLi (Left2Right=Left2Monde * Monde2Right)
    aMat = aRMatInv * aLMat;
    // vector de Translation entre 2 centre optique s'exprime en "coordonne du monde"
    aTr = aRMatInv * (aRotL.COpt() - aRotR.COpt());

}



/***********************************************************/
/*                                                         */
/*          cEqObsBlockCam                                 */
/*                                                         */
/***********************************************************/

class cEOBC_ModeRot
{
    public :

       cEOBC_ModeRot() :
          mMatR0("GL_MK0",3,3),
          mMatL0("GL_MK1",3,3),
          mMatR1("GL_MK2",3,3),
          mMatL1("GL_MK3",3,3)
       {
       }

       cMatr_Etat_PhgrF   mMatR0;
       cMatr_Etat_PhgrF   mMatL0;
       cMatr_Etat_PhgrF   mMatR1;
       cMatr_Etat_PhgrF   mMatL1;
};

class cEqObsBlockCam  : public cNameSpaceEqF,
                        public cObjFormel2Destroy
{
     public :
        friend class   cSetEqFormelles;
        const std::vector<double> &  AddObsRot(const double & aPdsTr,const double & aPdsMatr);
        double  AddObsDist(const double & aPdsDist);
        void  DoAMD(cAMD_Interf * anAMD);

        // Pas utile a priori pour l'equation, mais apparu quand on le met dans Tapas pour verifier que c'est initialise
        void SetCams(cGenPoseCam *,cGenPoseCam*,cGenPoseCam*,cGenPoseCam*); 
        bool CamIsInit() const;
     private  :

         cEqObsBlockCam
         (
             cRotationFormelle & aRotRT0,
             cRotationFormelle & aRotLT0,
             cRotationFormelle & aRotRT1,
             cRotationFormelle & aRotLT1,
             bool                doGenerateCode,
             bool                ModeDistance
         );

          void GenerateCode();
          cEqObsBlockCam(const cEqObsBlockCam &); // Non Implemanted

          cSetEqFormelles *   mSet;
          bool                mModeDistance;
          cRotationFormelle * mRotRT0;
          cRotationFormelle * mRotLT0;
          cRotationFormelle * mRotRT1;
          cRotationFormelle * mRotLT1;
          cIncListInterv      mLInterv;
          std::string         mNameType;
          cElCompiledFonc*    mFoncEqResidu;
          std::vector<cGenPoseCam *> mVGP;
#if  (HandleGL)
          cEOBC_ModeRot *     mModeR;
#endif
};


bool cEqObsBlockCam::CamIsInit() const
{
    for (const auto & aPC : mVGP)
       if (aPC && (!aPC->RotIsInit()))
          return false;
    return true;
}


void cEqObsBlockCam::SetCams(cGenPoseCam * aCam1,cGenPoseCam* aCam2,cGenPoseCam* aCam3,cGenPoseCam* aCam4)
{
    mVGP.push_back(aCam1);
    mVGP.push_back(aCam2);
    mVGP.push_back(aCam3);
    mVGP.push_back(aCam4);
}

cEqObsBlockCam::cEqObsBlockCam
(
    cRotationFormelle & aRotRT0,
    cRotationFormelle & aRotLT0,
    cRotationFormelle & aRotRT1,
    cRotationFormelle & aRotLT1,
    bool                doGenerateCode,
    bool                ModeDistance
) :
    mSet       (aRotRT0.Set()),
    mModeDistance (ModeDistance),
    mRotRT0    (&aRotRT0),
    mRotLT0    (&aRotLT0),
    mRotRT1    (&aRotRT1),
    mRotLT1    (&aRotLT1),
    mNameType     (mModeDistance ? "cCodeDistBlockCam" : "cCodeBlockCam"),
    mFoncEqResidu  (0),
#if  (HandleGL)
    mModeR (ModeDistance ? 0  : new cEOBC_ModeRot)
#endif

{

   AllowUnsortedVarIn_SetMappingCur = true;

   ELISE_ASSERT(mSet==mRotRT0->Set(),"Different sets incEqObsBlockCam");
   ELISE_ASSERT(mSet==mRotLT0->Set(),"Different sets incEqObsBlockCam");
   ELISE_ASSERT(mSet==mRotRT1->Set(),"Different sets incEqObsBlockCam");
   ELISE_ASSERT(mSet==mRotLT1->Set(),"Different sets incEqObsBlockCam");


   mRotRT0->IncInterv().SetName("OriR0");
   mRotLT0->IncInterv().SetName("OriL0");
   mRotRT1->IncInterv().SetName("OriR1");
   mRotLT1->IncInterv().SetName("OriL1");

   mLInterv.AddInterv(mRotRT0->IncInterv());
   mLInterv.AddInterv(mRotLT0->IncInterv());
   mLInterv.AddInterv(mRotRT1->IncInterv());
   mLInterv.AddInterv(mRotLT1->IncInterv());
   
   if (doGenerateCode)
   {
      GenerateCode();
      return;
   }

   mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
   ELISE_ASSERT(mFoncEqResidu!=0,"Cannot allocate cEqObsBlockCam");
   mFoncEqResidu->SetMappingCur(mLInterv,mSet);

#if  (HandleGL)
   if (!mModeDistance)
   {
       mModeR->mMatR0.InitAdr(*mFoncEqResidu);
       mModeR->mMatL0.InitAdr(*mFoncEqResidu);
       mModeR->mMatR1.InitAdr(*mFoncEqResidu);
       mModeR->mMatL1.InitAdr(*mFoncEqResidu);
   }
#endif
   mSet->AddFonct(mFoncEqResidu);
}


void  cEqObsBlockCam::DoAMD(cAMD_Interf * anAMD)
{
    std::vector<int> aNums;

    aNums.push_back(mRotRT0->IncInterv().NumBlocAlloc());
    aNums.push_back(mRotLT0->IncInterv().NumBlocAlloc());
    aNums.push_back(mRotRT1->IncInterv().NumBlocAlloc());
    aNums.push_back(mRotLT1->IncInterv().NumBlocAlloc());

    for (int aK1=0 ; aK1 <int(aNums.size()) ; aK1++)
        for (int aK2=aK1 ; aK2 <int(aNums.size()) ; aK2++)
            anAMD->AddArc(aNums[aK1],aNums[aK2],true);

}

cEqObsBlockCam * cSetEqFormelles::NewEqBlockCal
                 ( 
                         cRotationFormelle & aRotRT0,
                         cRotationFormelle & aRotLT0,
                         cRotationFormelle & aRotRT1,
                         cRotationFormelle & aRotLT1,
                         bool                doGenerateCode,
                         bool                ModeDistance
                 )
{
     ELISE_ASSERT(aRotRT0.Set() == this,"cSetEqFormelles::NewEqBlockCal");

     cEqObsBlockCam * aRes = new cEqObsBlockCam(aRotRT0,aRotLT0,aRotRT1,aRotLT1,doGenerateCode,ModeDistance);

     AddObj2Kill(aRes);
     return aRes;
}

void cEqObsBlockCam::GenerateCode()
{
    std::vector<Fonc_Num> aVF;

    if (mModeDistance)
    {
         Pt3d<Fonc_Num> aPRL0 = mRotRT0->COpt() - mRotLT0->COpt();
         Pt3d<Fonc_Num> aPRL1 = mRotRT1->COpt() - mRotLT1->COpt();

         Fonc_Num aF = euclid(aPRL0) - euclid(aPRL1);
         aVF.push_back(aF);
    }
    else
    {
        Pt3d<Fonc_Num>     aTrT0;
        ElMatrix<Fonc_Num> aMatT0(3,3);
        CalcParamEqRel(aTrT0,aMatT0,*mRotRT0,*mRotLT0,0,1);

        Pt3d<Fonc_Num>     aTrT1;
        ElMatrix<Fonc_Num> aMatT1(3,3);
        CalcParamEqRel(aTrT1,aMatT1,*mRotRT1,*mRotLT1,2,3);


        Pt3d<Fonc_Num> aResTr = aTrT1-aTrT0;
        ElMatrix<Fonc_Num> aResMat = aMatT1-aMatT0;

        aVF.push_back(aResTr.x);
        aVF.push_back(aResTr.y);
        aVF.push_back(aResTr.z);
        for (int aKi=0 ; aKi<3 ; aKi++)
        {
            for (int aKj=0 ; aKj<3 ; aKj++)
            {
               aVF.push_back(aResMat(aKi,aKj));
            }
        }
    }

    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aVF,  // expressions formelles 
        mLInterv  // intervalle de reference
    );

}

double  cEqObsBlockCam::AddObsDist(const double & aPdsDist)
{
  ELISE_ASSERT(mModeDistance,"cEqObsBlockCam::AddObsRot in mode Distance");
  return mSet->AddEqFonctToSys(mFoncEqResidu,aPdsDist,false);
}

const std::vector<double> &  cEqObsBlockCam::AddObsRot(const double & aPdsTr,const double & aPdsMatr)
{
  ELISE_ASSERT(!mModeDistance,"cEqObsBlockCam::AddObsRot in mode Distance");
#if  (HandleGL)
   mModeR->mMatR0.SetEtat(mRotRT0->MGL());
   mModeR->mMatL0.SetEtat(mRotLT0->MGL());
   mModeR->mMatR1.SetEtat(mRotRT1->MGL());
   mModeR->mMatL1.SetEtat(mRotLT1->MGL());
#endif
   //    mGPS.SetEtat(aGPS);
   std::vector<double> aVPds;
   for (int aK=0 ; aK<3; aK++) 
       aVPds.push_back(aPdsTr);

   for (int aK=0 ; aK<9; aK++) 
       aVPds.push_back(aPdsMatr);

  // Compute the values and its derivative (in class cCodeBlockCam here)
  // Link it to the real index of the variable
  // fill the covariance stuff taking into account these values
  return mSet->VAddEqFonctToSys(mFoncEqResidu,aVPds,false,NullPCVU);
}

void GenerateCodeBlockCam(bool ModeDist)
{
   cSetEqFormelles aSet;

   ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
   cRotationFormelle * aRotRT0 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);
   cRotationFormelle * aRotLT0 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);
   cRotationFormelle * aRotRT1 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);
   cRotationFormelle * aRotLT1 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);


  cEqObsBlockCam * aEOBC = aSet.NewEqBlockCal (*aRotRT0,*aRotLT0,*aRotRT1,*aRotLT1,true,ModeDist);
  DoNothingButRemoveWarningUnused(aEOBC);
}

void GenerateCodeBlockCam()
{
    GenerateCodeBlockCam(false);
    GenerateCodeBlockCam(true);
}

/***********************************************************/
/*                                                         */
/*                                                         */
/*                                                         */
/***********************************************************/

cTypeCodageMatr ExportMatr(const ElMatrix<double> & aMat)
{
    cTypeCodageMatr  aCM;

    aMat.GetLig(0, aCM.L1() );
    aMat.GetLig(1, aCM.L2() );
    aMat.GetLig(2, aCM.L3() );
    aCM.TrueRot().SetVal(true);

    return aCM;
}

ElMatrix<double> ImportMat(const cTypeCodageMatr & aCM)
{
   ElMatrix<double> aMat(3,3);

   SetLig(aMat,0,aCM.L1());
   SetLig(aMat,1,aCM.L2());
   SetLig(aMat,2,aCM.L3());

   return aMat;
}



/***********************************************************/
/*                                                         */
/*                  cIBC_ImsOneTime                        */
/*                  cIBC_OneCam                            */
/*                  cImplemBlockCam                        */
/*                                                         */
/***********************************************************/


class cIBC_ImsOneTime
{
    public :
        cIBC_ImsOneTime(int aNbCam,const std::string& aNameTime) ;
        void  AddPose(cPoseCam *, int aNum);
        cPoseCam * Pose(int aKP);
        void SetNum(int aNum);
        const std::string & NameTime() const {return mNameTime;}

    private :

        std::vector<cPoseCam *> mVCams;
        std::string             mNameTime;
        int                     mNums;
};


class cIBC_OneCam
{
      public :
          cIBC_OneCam(const std::string & ,int aNum);
          const int & CamNum() const;
          const std::string & NameCam() const;
          const bool & V0Init() const;
          // L'initiation ne peut etre faite que quand toute les poses ont ete lues, donc pas
          // complete dans le constructeur
          void Init0(const cParamOrientSHC & aPSH,cSetEqFormelles &,const cBlockGlobalBundle *);
          void AddContraintes(bool Stricte);
          cRotationFormelle & RF();
          cRotationFormelle * PtrRF();
      private :
          std::string mNameCam;
          int               mCamNum;  // Numero arbitraire (0,1,2...) , 0=PIVOT
          bool              mV0Init;  // Est ce que la position dans le bloc a une valeur initiale 
          Pt3dr             mC0;      // Valeur initiale du centre
          ElMatrix<double>  mMat0;    // Valeur initiale de l'orientation
          cSetEqFormelles *   mSetEq;
          cRotationFormelle * mRF;       // Rotation inconnue
          bool                mHasCstr;  // => L'image PIVOT dans un bloc, ou le premier bloc "virtuel"
          bool                mStricteCstr;  // 
};





class cImplemBlockCam
{
    public :
         // static cImplemBlockCam * AllocNew(cAppliApero &,const cStructBlockCam,const std::string & anId);
         cImplemBlockCam(cAppliApero & anAppli,const cStructBlockCam,const cBlockCamera & aBl,const std::string & anId );

         void EstimCurOri(const cXml_EstimateOrientationInitBlockCamera &);
         void Export(const cExportBlockCamera &);

         void DoCompensation(const cObsBlockCamRig &);
         void DoCompensationRot
              (
                  const cObsBlockCamRig & anObs,
                  const cRigidBlockWeighting & aRBW,
                  const std::vector<cEqObsBlockCam*>& aVecEq
              );
         void DoCompensationDist
              (
                  const cObsBlockCamRig & anObs,
                  const cRigidBlockWeighting & aRBW,
                  const std::vector<cEqObsBlockCam*>& aVecEq
              );
         void DoAMD(cAMD_Interf * anAMD);
         void DoAMD(cAMD_Interf * anAMD,const std::vector<cEqObsBlockCam* > & aVec);
         void AddContraintes(bool Stricte);

         const cStructBlockCam & SBC() const;
    private :
         void InitRF();

         cAppliApero &               mAppli;
         cStructBlockCam             mSBC;
         cLiaisonsSHC *              mLSHC;
         cStructBlockCam             mEstimSBC;
         std::string                 mId;

         std::map<std::string,cIBC_OneCam *>   mName2Cam;
         std::vector<cIBC_OneCam *>            mNum2Cam;
         int                                   mNbCam;
         int                                   mNbTime;

         std::map<std::string,cIBC_ImsOneTime *> mName2ITime;
         std::vector<cIBC_ImsOneTime *>          mNum2ITime;
         bool                                    mDoneIFC;
         const cUseForBundle *                   mUFB;
         const cBlockGlobalBundle *              mBlGB;
         bool                                    mHasCstrGlob;
         bool                                    mRelTB;
         bool                                    mRelDistTB;
         bool                                    mGlobDistTB;
         std::vector<cEqObsBlockCam* >           mVecEqGlob;
         std::vector<cEqObsBlockCam* >           mVecEqRel;
         std::vector<cEqObsBlockCam* >           mVecEqDistGlob;

         std::vector<cRotationFormelle *>        mRF0;
};

    // =================================
    //              cIBC_ImsOneTime
    // =================================

class  cCmp_IOT_Ptr
{
   public :
       bool operator () (cIBC_ImsOneTime * aT1,cIBC_ImsOneTime * aT2)
       {
          return  aT1->NameTime() < aT2->NameTime();
       }
};

static cCmp_IOT_Ptr TheIOTCmp;


cIBC_ImsOneTime::cIBC_ImsOneTime(int aNb,const std::string & aNameTime) :
       mVCams     (aNb),
       mNameTime (aNameTime)
{
}

void cIBC_ImsOneTime::SetNum(int aNum)
{
    for (auto & aPCam : mVCams)
    {
        if (aPCam)
           aPCam->SetNumTimeBloc(aNum);
    }
}

void  cIBC_ImsOneTime::AddPose(cPoseCam * aPC, int aNum) 
{
    cPoseCam * aPC0 =  mVCams.at(aNum);
    if (aPC0 != 0)
    {
         std::cout <<  "For cameras " << aPC->Name() <<  "  and  " << aPC0->Name() << "\n";
         ELISE_ASSERT(false,"Conflicting name from KeyIm2TimeCam ");
    }
    
    mVCams[aNum] = aPC;
}


cPoseCam * cIBC_ImsOneTime::Pose(int aKP)
{
   return mVCams.at(aKP);
}
    // =================================
    //              cIBC_OneCam 
    // =================================

cIBC_OneCam::cIBC_OneCam(const std::string & aNameCam ,int aNum) :
    mNameCam (aNameCam ),
    mCamNum  (aNum),
    mV0Init  (false),
    mMat0    (1,1),
    mSetEq   (0),
    mRF      (0),
    mHasCstr     (false),
    mStricteCstr (false)
{
}

cRotationFormelle & cIBC_OneCam::RF()
{
    ELISE_ASSERT(mRF!=0,"cIBC_OneCam::RF_Nn");
    return *mRF;
}

cRotationFormelle * cIBC_OneCam::PtrRF()
{
    return mRF;
}

const int & cIBC_OneCam::CamNum() const {return mCamNum;}
const std::string & cIBC_OneCam::NameCam() const { return mNameCam; }
const bool & cIBC_OneCam::V0Init() const {return mV0Init;}


/*
    aPOS => geometrie de la calibration
    aBGB => rappel / a cette geometrie
*/
void cIBC_OneCam::Init0(const cParamOrientSHC & aPOS,cSetEqFormelles & aSet,const cBlockGlobalBundle * aBGB)
{
    
    mV0Init = true;
    mC0   = aPOS.Vecteur();
    mMat0 = ImportMat(aPOS.Rot());

    mSetEq = & aSet;

    // Si il y a une valeur de reference
    if (aBGB)
    {
       bool aInitSigm   = aBGB->SigmaV0().IsInit();
       bool aInitStrict = aBGB->V0Stricte().ValWithDef(false);
       ELISE_ASSERT(!(aInitSigm&&aInitStrict),"Both Stritct && Sigma in BlockGlobalBundle");
       // ModifDIST
       mRF = aSet.NewRotation(cNameSpaceEqF::eRotFigee,ElRotation3D(mC0,mMat0,true));

       if (mCamNum==NUMPIVOT)  // C'est la "premiere",  donc on la fige a sa valeur initiale (en general Id)
       {
          mStricteCstr = true;
          mHasCstr     = true;
       }
       else // Sinon il y a peut etre un rappel a la V0, 
       {    // Si elle existe, elle est SOIT stricte , SOIT ponderee par un sigma
          if (aInitSigm)
          {
              mHasCstr  = true;
              mStricteCstr = false;
              const cXml_SigmaRot & aSigm = aBGB->SigmaV0().Val();
              mRF->SetTolAng(aSigm.Ang());
              mRF->SetTolCentre(aSigm.Center());
          }
          if (aInitStrict)
          {
              mHasCstr  = true;
              mStricteCstr = true;
          }
       }
    }

    // std::cout << "FFffffFFfffff  " <<  mStricteCstr  << " HAS " << mHasCstr << "\n";
    // getchar();
}

void cIBC_OneCam::AddContraintes(bool Stricte)
{
   if (! mRF) return;
   
   // std::cout << "AddContraintesAddContraintes " << Stricte << " " << mStricteCstr  << " HAS " << mHasCstr << "\n";
   if (mHasCstr)
   {
      if (Stricte == mStricteCstr)
          mSetEq->AddContrainte(mRF->StdContraintes(),Stricte);
   }
}

    // =================================
    //       cImplemBlockCam
    // =================================


void cImplemBlockCam::InitRF()
{

    for 
    (
        std::list<cParamOrientSHC>::const_iterator itPOS=mLSHC->ParamOrientSHC().begin();
        itPOS !=mLSHC->ParamOrientSHC().end();
        itPOS++
    )
    {
        cIBC_OneCam * aCam = mName2Cam[itPOS->IdGrp()];
        if (aCam==0)
        {
            std::cout << "For group [" << itPOS->IdGrp() << "]\n";
            ELISE_ASSERT(false,"Cannot get cam from IdGrp");
        }
        ELISE_ASSERT(! aCam->V0Init(),"Multiple Init For IdGrp");
        aCam->Init0(*itPOS,mAppli.SetEq(),mBlGB);
    }

    for (int aKC=0 ; aKC<mNbCam ; aKC++)
    {
        ELISE_ASSERT(mNum2Cam[aKC]->V0Init(),"Camera non init in grp");
    }

    // std::cout << "TESSTTT CAM UKNOWWnnnnn \n";
    // getchar();
}

cImplemBlockCam::cImplemBlockCam
(
     cAppliApero & anAppli,
     const cStructBlockCam aSBC,
     const cBlockCamera &  aParamCreateBC,
     const std::string & anId
) :
      mAppli          (anAppli),
      mSBC            (aSBC),
      mEstimSBC       (aSBC),
      mId             (anId),
      mDoneIFC        (false),
      mUFB            (0),
      mBlGB           (0),
      mHasCstrGlob    (false),
      mRelTB          (false),
      mRelDistTB      (false),
      mGlobDistTB     (false)
{
    const std::vector<cPoseCam*> & aVP = mAppli.VecAllPose();
    std::string aMasterGrp = aSBC.MasterGrp().ValWithDef("");

    // On initialise les camera
    for (int anIter = 0 ; anIter<2 ; anIter++)  // Deux iter pour forcer le groupe maitre eventuellement
    {

        for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
        {
            cPoseCam * aPC = aVP[aKP];
            std::string aNamePose = aPC->Name();
            std::pair<std::string,std::string> aPair =   mAppli.ICNM()->Assoc2To1(mSBC.KeyIm2TimeCam(),aNamePose,true);
            std::string aNameCam = aPair.second;
            // At first iter, we do it if  master, at second if not master
            bool Doit = (anIter==0) == (aNameCam==aMasterGrp);
            if (Doit && (! DicBoolFind(mName2Cam,aNameCam))) // si aNameCam se trouve dans mName2Cam
            {
                cIBC_OneCam *  aCam = new cIBC_OneCam(aNameCam, (int)mNum2Cam.size()); // (name & index dans mNum2Cam)
                mName2Cam[aNameCam] = aCam;
                mNum2Cam.push_back(aCam); 
            }
        }
    }
    mNbCam  = (int)mNum2Cam.size();
    mUFB = aParamCreateBC.UseForBundle().PtrVal();
    mLSHC = mSBC.LiaisonsSHC().PtrVal();
    // mGlobCmp = mForCompens && 

    // Initialiser les parametres & camera pour block compensation
    if (mUFB)   // UFB=UseForBundle
    {
        ELISE_ASSERT(mLSHC!=0,"Compens without LiaisonsSHC");
        mBlGB = mUFB->BlockGlobalBundle().PtrVal();
        mHasCstrGlob = (mBlGB!=0) && (mBlGB->SigmaV0().IsInit() || mBlGB->V0Stricte().ValWithDef(false));
        mRelTB = mUFB->RelTimeBundle();
        mGlobDistTB = mUFB->GlobDistTimeBundle().Val();
        mRelDistTB = mUFB->RelDistTimeBundle().Val();
        ELISE_ASSERT(!mRelDistTB,"RelDistTimeBundle not handled for now");
        InitRF();
    }
    
    // On regroupe les images prises au meme temps
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
          cPoseCam * aPC = aVP[aKP];
          std::string aNamePose = aPC->Name();
          std::pair<std::string,std::string> aPair =   mAppli.ICNM()->Assoc2To1(mSBC.KeyIm2TimeCam(),aNamePose,true);
          std::string aNameTime = aPair.first; // get a time stamp
          std::string aNameCam  = aPair.second;// get a cam name

          cIBC_ImsOneTime * aIms =  mName2ITime[aNameTime];
          if (aIms==0)  // check if there exist this time stamp in database
          {
               // If not, create a cIBC_ImsOneTime to group all img with same time stamp
               aIms = new cIBC_ImsOneTime(mNbCam,aNameTime);    // cIBC_ImsOneTime contains a vector bool cPoseCam size mNbCam => flag Pose cam with same time stamp
               mName2ITime[aNameTime] = aIms;
               mNum2ITime.push_back(aIms);
          }
          cIBC_OneCam * aCam = mName2Cam[aNameCam]; // structure map, get cIBC_OneCam from cam name
          aIms->AddPose(aPC,aCam->CamNum());   // if exist => add image to this time stamp
    }
    mNbTime = (int)mNum2ITime.size();
    std::sort(mNum2ITime.begin(),mNum2ITime.end(),TheIOTCmp); // sort by time stamp
    for (int aK=0 ; aK<mNbTime ; aK++)
        mNum2ITime[aK]->SetNum(aK);


// ## 
//    On peut avoir equation / a calib et  I/I+1 (pour model derive)

    if (mUFB) // UFB=UseForBundle
    {
       if (mBlGB) // BlGB=BlockGlobalBundle - global with no attachement to known value ?
       {
          for (int aKTime=0 ; aKTime<mNbTime ; aKTime++)    // scan all time stamp
          {
               cIBC_ImsOneTime * aTim =  mNum2ITime[aKTime];
               for (int aKCam=0 ; aKCam<mNbCam ; aKCam++)   // scan all camera in chantier
               {
                   if (aKCam != NUMPIVOT)   // if it is not a 1er cam (1er time stamp in the Right)
                   {
                       // Obs: Pour chaque cliches: 1 img L + 1 img R
                       // 1 img R de 1er cliches + 1 img L de 1er cliches
                       cIBC_OneCam *  aCamR =  mNum2Cam[NUMPIVOT]; // mNum2Cam[0]
                       cIBC_OneCam *  aCamL =  mNum2Cam[aKCam];
                       cPoseCam * aPcR1 = aTim->Pose(NUMPIVOT);
                       cPoseCam * aPcL1 = aTim->Pose(aKCam);
                       if (aCamR && aCamL && aPcR1 &&aPcL1)
                       {
                          // Quelle est model ? Quelle est inconnu ?
                          cEqObsBlockCam * anEq = mAppli.SetEq().NewEqBlockCal
                                                      (
                                                         aCamR->RF(),   // get rotation Cam->Monde
                                                         aCamL->RF(),
                                                         aPcR1->RF(),
                                                         aPcL1->RF(),
                                                         false,
                                                         false
                                                      );

                          anEq->SetCams(nullptr,nullptr,aPcR1,aPcL1);
                          mVecEqGlob.push_back(anEq);
                       }
                   }
               }
          }
          // ModifDIST 
       }
       if (mGlobDistTB) // Mode time relative ?
       {
          for (int aKTime=0 ; aKTime<mNbTime ; aKTime++)
          {
               cIBC_ImsOneTime * aTim =  mNum2ITime[aKTime];
               for (int aKCam1=0 ; aKCam1<mNbCam ; aKCam1++)
               {
                   for (int aKCam2=aKCam1+1 ; aKCam2<mNbCam ; aKCam2++)
                   {
                       // Obs:
                       cIBC_OneCam *  aCamR =  mNum2Cam[aKCam1];
                       cIBC_OneCam *  aCamL =  mNum2Cam[aKCam2];
                       cPoseCam * aPcR1 = aTim->Pose(aKCam1);
                       cPoseCam * aPcL1 = aTim->Pose(aKCam2);
                       if (aCamR && aCamL && aPcR1 &&aPcL1)
                       {
                          cEqObsBlockCam * anEq = mAppli.SetEq().NewEqBlockCal
                                                      (
                                                         aCamR->RF(),
                                                         aCamL->RF(),
                                                         aPcR1->RF(),
                                                         aPcL1->RF(),
                                                         false,
                                                         true
                                                      );

                          anEq->SetCams(nullptr,nullptr,aPcR1,aPcL1);
                          mVecEqDistGlob.push_back(anEq);
                       }
                   }
               }
          }
          // ModifDIST 
       }
       if (mRelTB) // global with attachement to known value
       {
           for (int aKTime=1 ; aKTime<mNbTime ; aKTime++)
           {
               cIBC_ImsOneTime * aT0 =  mNum2ITime[aKTime-1];
               cIBC_ImsOneTime * aT1 =  mNum2ITime[aKTime];
               for (int aKCam=0 ; aKCam<mNbCam ; aKCam++)
               {
                   if (aKCam != NUMPIVOT)
                   {   // Obs couple: Chaque temps (cliches): 1 image L + 1 image R
                       // Prendre 2 cliches consecutives pour 1 equation
                       cPoseCam * aPcR0 = aT0->Pose(NUMPIVOT);
                       cPoseCam * aPcL0 = aT0->Pose(aKCam);
                       cPoseCam * aPcR1 = aT1->Pose(NUMPIVOT);
                       cPoseCam * aPcL1 = aT1->Pose(aKCam);
                       if (aPcR0 && aPcL0 && aPcR1 &&aPcL1)
                       {
                          cEqObsBlockCam * anEq = mAppli.SetEq().NewEqBlockCal
                                                      (
                                                         aPcR0->RF(),
                                                         aPcL0->RF(),
                                                         aPcR1->RF(),
                                                         aPcL1->RF(),
                                                         false,
                                                         false
                                                      );
                          anEq->SetCams(aPcR0,aPcL0,aPcR1,aPcL1);

                           mVecEqRel.push_back(anEq);
                       }
                   }
               }
           }
       }
// Creer les equation dans  cIBC_ImsOneTime 
    }

}

// Rajouter structure compens dans SectionObservation

void cImplemBlockCam::DoCompensationRot
     (
            const cObsBlockCamRig & anObs,
            const cRigidBlockWeighting & aRBW,
            const std::vector<cEqObsBlockCam*>& aVecEq
     )
{
    std::cout << "ADR APLI " 
                 << mAppli.PdsAvIter() 
                 << " " <<  mAppli.RBW_PdsTr(aRBW) 
                 << " " << mAppli.RBW_PdsRot(aRBW)
                 << " PDS " << mAppli.PdsAvIter()
                 << "\n";

    double aGlobEcMat = 0;
    double aGlobEcPt = 0;
    for (int aKE=0 ; aKE<int(aVecEq.size()) ; aKE++)
    {
          cEqObsBlockCam &  anEQ = *(aVecEq[aKE]) ;
          if (anEQ.CamIsInit())
          {
              //  std::cout << "HHHHH " << & anEQ << "\n";
              const std::vector<double> & aResidu = anEQ.AddObsRot(mAppli.RBW_PdsTr(aRBW),mAppli.RBW_PdsRot(aRBW));
              // const std::vector<double> & aResidu = anEQ.AddObs(aRBW.PondOnTr(),aRBW.PondOnRot());
              double aSomEcartMat = 0;
              double aSomEcartPt = 0;

              for (int aK=0 ; aK<3 ; aK++)
                  aSomEcartPt += ElSquare(aResidu[aK]);

              for (int aK=3 ; aK<12 ; aK++)
                  aSomEcartMat += ElSquare(aResidu[aK]);

              aGlobEcMat += aSomEcartMat;
              aGlobEcPt += aSomEcartPt;
       
              if (anObs.Show().Val())   // comment activer cette option ?
              {
                  std::cout << "    " << aKE << "   EcMat  " <<  sqrt(aSomEcartMat)   
                       << " XYZ " <<  sqrt(aSomEcartPt) <<" \n";
              }
          }
    }
    std::cout << " GlobMat    " <<  sqrt(aGlobEcMat/aVecEq.size())   
              << " GlobXYZ    " <<  sqrt(aGlobEcPt/aVecEq.size()) <<" \n";
}

void cImplemBlockCam::DoCompensationDist
     (
            const cObsBlockCamRig & anObs,
            const cRigidBlockWeighting & aRBW,
            const std::vector<cEqObsBlockCam*>& aVecEq
     )
{
    double aGlobEcDist = 0;
    for (int aKE=0 ; aKE<int(aVecEq.size()) ; aKE++)
    {
          cEqObsBlockCam &  anEQ = *(aVecEq[aKE]) ;
          if (anEQ.CamIsInit())
          {
             double  aResidu = anEQ.AddObsDist(mAppli.RBW_PdsTr(aRBW));
             aGlobEcDist += ElSquare(aResidu);
          }
    }
    aGlobEcDist /= aVecEq.size();
    std::cout << "Comp Dist = " << sqrt(aGlobEcDist) << "\n";
}


void cImplemBlockCam::DoCompensation(const cObsBlockCamRig & anObs)
{
    if (anObs.GlobalPond().IsInit()) 
    {
       ELISE_ASSERT(mBlGB,"Required BlockGlobalBundle, not specify at creation");
       DoCompensationRot(anObs,anObs.GlobalPond().Val(),mVecEqGlob);
    }
    if (anObs.RelTimePond().IsInit()) 
    {
       ELISE_ASSERT(mRelTB,"Required RelTimeBundle, not specify at creation");
       DoCompensationRot(anObs,anObs.RelTimePond().Val(),mVecEqRel);
    }
    if (anObs.GlobalDistPond().IsInit()) 
    {
       ELISE_ASSERT(mGlobDistTB,"Required GlobDistTimeBundle, not specify at creation");
       DoCompensationDist(anObs,anObs.GlobalDistPond().Val(),mVecEqDistGlob);

    }
}

void cImplemBlockCam::Export(const cExportBlockCamera & aEBC)
{
    MakeFileXML(mEstimSBC,mAppli.ICNM()->Dir()+aEBC.NameFile());
}


void  cImplemBlockCam::DoAMD(cAMD_Interf * anAMD,const std::vector<cEqObsBlockCam* > & aVec)
{
      for (int aKE=0 ; aKE<int(aVec.size()) ; aKE++)
      {
          aVec[aKE]->DoAMD(anAMD);
      }
}

void  cImplemBlockCam::DoAMD(cAMD_Interf * anAMD)
{
    DoAMD(anAMD,mVecEqRel);
    DoAMD(anAMD,mVecEqGlob);
    DoAMD(anAMD,mVecEqDistGlob);
}


const cStructBlockCam & cImplemBlockCam::SBC() const {return mSBC;}


void cImplemBlockCam::EstimCurOri(const cXml_EstimateOrientationInitBlockCamera & anEOIB)
{
   cLiaisonsSHC aLSHC;
   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       if (anEOIB.Show().Val())
          std::cout << "=================================================\n";
       cIBC_OneCam * aCam  = mNum2Cam[aKC];

       bool ValueKnown = false;
       ElRotation3D aRMoy = ElRotation3D::Id;
       if (aCam->PtrRF())
       {
          ValueKnown = true;
          aRMoy = aCam->PtrRF()->CurRot();
       }
       else
       {
          Pt3dr aSomTr(0,0,0);
          double aSomP=0;
          ElMatrix<double> aSomM(3,3,0.0);
          for (int aKT=0 ; aKT<mNbTime ; aKT++)
          {
               cIBC_ImsOneTime *  aTime =  mNum2ITime[aKT];
               cPoseCam * aP0 = aTime->Pose(0);
               cPoseCam * aP1 = aTime->Pose(aKC);
               if (aP0 && aP1)
               {
                   ElRotation3D  aR0toM = aP0->CurCam()->Orient().inv(); // CONV-ORI
                   ElRotation3D  aR1toM = aP1->CurCam()->Orient().inv();

                   ElRotation3D aR1to0 = aR0toM.inv() * aR1toM;  //  CONV-ORI

                   if (anEOIB.Show().Val())
                   {
                       std::cout << "  EstimCurOri " << aP0->Name() <<  " " << aP1->Name() << "\n";
                       std::cout << "    " <<  aR1to0.ImAff(Pt3dr(0,0,0)) 
                                         << " " << aR1to0.teta01() 
                                         << " " << aR1to0.teta02() 
                                         << " " << aR1to0.teta12() 
                                         << "\n";
                   }
                   aSomTr = aSomTr+ aR1to0.tr();
                   aSomM += aR1to0.Mat();
                   aSomP++;
               }
          }
          if (aSomP)
          {
             ValueKnown = true;
             aSomTr = aSomTr / aSomP;
             aSomM *=  1.0/aSomP;
             aSomM = NearestRotation(aSomM);
             aRMoy = ElRotation3D(aSomTr,aSomM,true);
          }
       }

       if (ValueKnown)
       {
           Pt3dr aSomTr = aRMoy.tr();
           ElMatrix<double> aSomM = aRMoy.Mat();
           double aSomP = 0.0;
           
           double aSomEcP = 0.0;
           double aSomEcM = 0.0;
           for (int aKT=0 ; aKT<mNbTime ; aKT++)
           {
               cIBC_ImsOneTime *  aTime =  mNum2ITime[aKT];
               cPoseCam * aP0 = aTime->Pose(0);
               cPoseCam * aP1 = aTime->Pose(aKC);
               if (aP0 && aP1)
               {
                   ElRotation3D  aR0toM = aP0->CurCam()->Orient().inv(); // CONV-ORI
                   ElRotation3D  aR1toM = aP1->CurCam()->Orient().inv();
                   ElRotation3D aR1to0 = aR0toM.inv() * aR1toM;  //  CONV-ORI
                   Pt3dr aTr =  aR1to0.tr();
                   ElMatrix<double>       aMatr=  aR1to0.Mat();

                   aSomEcP += euclid(aTr-aSomTr);
                   aSomEcM += aMatr.L2(aSomM);
                   aSomP++;
               }
           }
           aSomEcP /= aSomP;
           aSomEcM = sqrt(aSomEcM) / aSomP;

           std::cout << "  ==========  AVERAGE =========== \n";
           std::cout << "    " <<  aRMoy.ImAff(Pt3dr(0,0,0))
                               << " tetas " << aRMoy.teta01() 
                               << "  " << aRMoy.teta02() 
                               << "  " << aRMoy.teta12() 
                               << "\n";
           std::cout << "    DispTr=" << aSomEcP << " DispMat=" << aSomEcM << "\n";

           cParamOrientSHC aP;
           aP.IdGrp() = aCam->NameCam();
           aP.Vecteur() = aRMoy.ImAff(Pt3dr(0,0,0));
           aP.Rot() = ExportMatr(aSomM);
           aLSHC.ParamOrientSHC().push_back(aP);
       }
   }
   
   mEstimSBC.LiaisonsSHC().SetVal(aLSHC);
}

void cImplemBlockCam::AddContraintes(bool Stricte)
{
    for (int aKC=0 ; aKC<mNbCam ; aKC++)
    {
        mNum2Cam[aKC]->AddContraintes(Stricte);
    }

}


    // =================================
    //       cAppliApero
    // =================================

void  cAppliApero::AMD_AddBlockCam()
{
   for 
   (
        std::map<std::string,cImplemBlockCam *>::iterator itB = mBlockCams.begin();
        itB != mBlockCams.end();
        itB++
   )
   {
       itB->second->DoAMD(mAMD);
   }

}

void cAppliApero::InitBlockCameras()
{
  for 
  (
        std::list<cBlockCamera>::const_iterator itB= mParam.BlockCamera().begin();
        itB!=mParam.BlockCamera().end();
        itB++
  )
  {
       std::string anId = itB->Id().ValWithDef(itB->NameFile());
       cStructBlockCam aSB = StdGetObjFromFile<cStructBlockCam>
                             (
                                 mICNM->Dir() + itB->NameFile(),
                                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 "StructBlockCam",
                                 "StructBlockCam"
                             );
       cImplemBlockCam * aIBC = new cImplemBlockCam(*this,aSB,*itB,anId);
       mBlockCams[anId] = aIBC;
       mHasBlockCams = true;
  }
}

ElRotation3D  cAppliApero::SVPGetRotationBloc(const std::string & aNameBloc,const std::string& aNamePose,std::string & aNameTimeResult)
{
   static std::map<std::string,cStructBlockCam> aMap;
   if (! DicBoolFind(aMap,aNameBloc))
   {
       aMap[aNamePose] =  StdGetFromPCP(aNameBloc,StructBlockCam);
   }
   const cStructBlockCam &  aSBC = aMap[aNamePose];

   ElRotation3D aRes = ElRotation3D::Id;
   aNameTimeResult = "";


   std::pair<std::string,std::string> aPair =   mICNM->Assoc2To1(aSBC.KeyIm2TimeCam(),aNamePose,true);
   std::string aNameTime = aPair.first; // get a time stamp
   std::string aNameGrp  = aPair.second;// get a cam name

   for (const auto  & aPO :   aSBC.LiaisonsSHC().Val().ParamOrientSHC() )
   {
        if (aPO.IdGrp() == aNameGrp)
        {
            aNameTimeResult = aNameTime;
            Pt3dr aTr = aPO.Vecteur();
            ElMatrix<double> aMat = ImportMat(aPO.Rot());
            aRes = ElRotation3D(aTr,aMat,true);
            return aRes;
        }
   }
     // std::pair<std::string,std::string> Assoc2To1(const tKey &,const std::string & aName,bool isDir);


   return aRes;
}


ElRotation3D  cAppliApero::GetUnikRotationBloc(const std::string & aNameBloc,const std::string& aNamePose)
{
   std::string aNameTR;
   ElRotation3D aRes = SVPGetRotationBloc(aNameBloc,aNamePose,aNameTR);

   ELISE_ASSERT(aNameTR!="","cAppliApero::GetUnikRotationBloc");
   return aRes;
}

  //==============================================

cPreCB1Pose::cPreCB1Pose(const ElRotation3D & aRot) :
   mRot (aRot)
{
}

cPreCompBloc::cPreCompBloc(const std::string  & aNameBloc) :
   mNameBloc (aNameBloc)
{
}

void cAppliApero::PreInitBloc(const std::string & aNameBloc)
{
  // Lecture du bloc
  static std::map<std::string,cStructBlockCam *>  MapBloc;
  if (! DicBoolFind(MapBloc,aNameBloc))
     MapBloc[aNameBloc] = new cStructBlockCam(StdGetFromPCP(aNameBloc,StructBlockCam));
  const cStructBlockCam & aStrBloc =  *(MapBloc[aNameBloc]);


  static std::map<std::string,cPreCompBloc *>  MapTime2PCB;
  for (auto & aPC : mVecPose)
  {
      cPreCompBloc * aPCB = aPC->GetPreCompBloc(true);
      if (aPCB==0)
      {
          std::pair<std::string,std::string> aPair = mICNM->Assoc2To1(aStrBloc.KeyIm2TimeCam(),aPC->Name(),true);
          std::string aNameTime = aPair.first; // get a time stamp
          std::string aNameGrp  = aPair.second;// get a cam name

          aPCB = MapTime2PCB[aNameTime];
          if (aPCB ==0)
          {
              aPCB = new cPreCompBloc(aNameBloc);
              MapTime2PCB[aNameTime] = aPCB;
          }
          aPCB->mGrp.push_back(aPC);
          ElRotation3D aRot = ElRotation3D::Id;
          int NbGotGrp = 0;
          for (const auto  & aPO :   aStrBloc.LiaisonsSHC().Val().ParamOrientSHC() )
          {
               if (aPO.IdGrp() == aNameGrp)
               {
                   Pt3dr aTr = aPO.Vecteur();
                   ElMatrix<double> aMat = ImportMat(aPO.Rot());
                   aRot = ElRotation3D(aTr,aMat,true);
                   NbGotGrp++;
               }
          }
          ELISE_ASSERT(NbGotGrp,"None or multiple group in bloc");
          
          aPC->SetPreCompBloc (aPCB); 
          aPC->SetPreCB1Pose(new cPreCB1Pose(aRot));
      }
      else
      {
         if (aPCB->mNameBloc!=aNameBloc)
         {
            std::cout << "For cam" << aPC->Name() << " Bl1=[" << aPCB->mNameBloc << "] Bl2=[" << aNameBloc <<"]\n";
            ELISE_ASSERT(false,"Cam bellong to multiple bloc");
         }
      }
  }
}



void  cAppliApero::BlocContraintes(bool Stricte)
{
   for 
   (
        std::map<std::string,cImplemBlockCam *>::iterator itB = mBlockCams.begin();
        itB != mBlockCams.end();
        itB++
   )
   {
       itB->second->AddContraintes(Stricte);
   }
}



cImplemBlockCam * cAppliApero::GetBlockCam(const std::string & anId)
{
   cImplemBlockCam* aRes = mBlockCams[anId];
   ELISE_ASSERT(aRes!=0,"cAppliApero::GetBlockCam");

   return aRes;
}

void  cAppliApero::EstimateOIBC(const cXml_EstimateOrientationInitBlockCamera & anEOIB)
{ 
    cImplemBlockCam * aBlock = GetBlockCam(anEOIB.Id());
    aBlock->EstimCurOri(anEOIB);
}


void cAppliApero:: ExportBlockCam(const cExportBlockCamera & aEBC)
{
    cImplemBlockCam * aBlock = GetBlockCam(aEBC.Id());
    if (aEBC.Estimate().IsInit())
    {
       aBlock->EstimCurOri(aEBC.Estimate().Val());
    }
    aBlock->Export(aEBC);
}

void cAppliApero::AddObservationsRigidBlockCam
     (
         const cObsBlockCamRig & anOBCR,
         bool IsLastIter,
         cStatObs & aSO
     )
{
    cImplemBlockCam * aBlock = GetBlockCam(anOBCR.Id());
    aBlock->DoCompensation(anOBCR);
}


/***********************************************************/
/*                                                         */
/*   BRAS DE LEVIER                                        */
/*                                                         */
/***********************************************************/

cAperoOffsetGPS::cAperoOffsetGPS(const cGpsOffset & aParam,cAppliApero & anAppli) :
    mAppli (anAppli),
    mParam (aParam),
    mBaseUnk (mAppli.SetEq().NewBaseGPS(mParam.ValInit()))
{
}


const cGpsOffset & cAperoOffsetGPS::ParamCreate() const {return mParam;}
cBaseGPS *         cAperoOffsetGPS::BaseUnk()           {return mBaseUnk;}


/***********************************************************/
/*                                                         */
/*               cCompiledObsRelGPS                        */
/*                                                         */
/***********************************************************/

class cCmpPtrPoseTime
{
    public :
      bool operator () (cPoseCam * aPC1,cPoseCam * aPC2)
      {
          return aPC1->Time() < aPC2->Time();
      }
};

cCompiledObsRelGPS::cCompiledObsRelGPS
(
    cAppliApero &     anAppli,
    cDeclareObsRelGPS aXML
)   :
    mXML  (aXML),
    mAppli (&anAppli)
{
   cElRegex  anAutom(mXML.PatternSel(),10);

   const std::vector<cPoseCam*> & aVP = mAppli->VecAllPose();

   for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
   {
        cPoseCam * aPose = aVP[aKP];
        if (anAutom.Match(aPose->Name()))
        {
           mVOrderedPose.push_back(aPose);
        }
   }

   cCmpPtrPoseTime aCmp;
   std::sort(mVOrderedPose.begin(),mVOrderedPose.end(),aCmp);

   for (int aKP=1 ; aKP<int(mVOrderedPose.size()) ; aKP++)
   {
      cPoseCam * aPC1 = mVOrderedPose[aKP-1];
      cPoseCam * aPC2 = mVOrderedPose[aKP];

      mVObs.push_back(mAppli->SetEq().NewEqRelativeGPS(aPC1->RF(),aPC2->RF()));
        // std::cout << "TTTT " << mVOrderedPose[aKP]->Name() << " " << mVOrderedPose[aKP]->Time() -  mVOrderedPose[aKP-1]->Time() << "\n";
   }
}

const cDeclareObsRelGPS &             cCompiledObsRelGPS::XML() const {return mXML;}
const std::vector<cPoseCam *> &       cCompiledObsRelGPS::VOrderedPose() const {return mVOrderedPose;}
const std::vector<cEqRelativeGPS *> & cCompiledObsRelGPS::VObs() const {return mVObs;}




void cAppliApero::InitObsRelGPS()
{
   for 
   (
       std::list<cDeclareObsRelGPS>::const_iterator itD=mParam.DeclareObsRelGPS().begin();
       itD != mParam.DeclareObsRelGPS().end() ;
       itD++
   )
   {
       cCompiledObsRelGPS * anObs = new cCompiledObsRelGPS(*this,*itD);
       const std::string  & anId = itD->Id();
       ELISE_ASSERT(mMCORelGps[anId] ==0,"Multiple Id in DeclareObsRelGPS");
       mMCORelGps[anId] = anObs;
   }
}


void  cAppliApero::AddOneObservationsRelGPS(const cObsRelGPS & aXMLObs)
{
     cCompiledObsRelGPS * aCObs = mMCORelGps[aXMLObs.Id()];
     const cGpsRelativeWeighting & aPond = aXMLObs.Pond();
     ELISE_ASSERT(aCObs!=0,"cAppliApero::AddObservationsRelGPS cannot find id");
     const std::vector<cPoseCam *> &  aVP = aCObs->VOrderedPose();
     const std::vector<cEqRelativeGPS *> & aVR = aCObs->VObs();

     for (int aKR = 0 ; aKR < int(aVR.size()) ; aKR++)
     {
         cPoseCam * aPC1 = aVP[aKR];
         cPoseCam * aPC2 = aVP[aKR+1];
         cEqRelativeGPS * anObs =  aVR[aKR];

         Pt3dr aC1 = aPC1->ObsCentre();
         Pt3dr aC2 = aPC2->ObsCentre();
         Pt3dr aDif21 = aC2-aC1;

         Pt3dr aResidu = anObs->Residu(aDif21);
         if ((! aPond.MaxResidu().IsInit()) || (euclid(aResidu) < aPond.MaxResidu().Val()))
         {
             double aT1 = aPC1->Time();
             double aT2 = aPC2->Time();
             double aSigma = aPond.SigmaMin() + aPond.SigmaPerSec() * ElAbs(aT1-aT2);
             Pt3dr aPds(1/ElSquare(aSigma),1/ElSquare(aSigma),1/ElSquare(aSigma));
             anObs->AddObs(aDif21,aPds);

             std::cout << "RELGPS " << aPC1->Name() << " " << aResidu 
                      << " D=" << euclid(aResidu) 
                      << " Sig0 " << aSigma<< "\n";
         }
     }
}

void  cAppliApero::AddObservationsRelGPS(const std::list<cObsRelGPS> & aLO)
{
    for (std::list<cObsRelGPS>::const_iterator itO=aLO.begin() ; itO!=aLO.end() ; itO++)
       AddOneObservationsRelGPS(*itO);
}


void cAppliApero::AddObservationsPlane(const cDataObsPlane & aDObs)
{
    for (const auto & anObs : aDObs.Data().Obs1Im())
    {
        const  cXml_ObsPlaneOnPose & aObs1Im =  anObs.second;
        cPoseCam * aPC = PoseFromName(aObs1Im.NameIm());

        aPC->AddObsPlaneOneCentre(aObs1Im,aDObs.Weight().Val());
        // std::cout << aObs1Im.NameIm()  << " " <<  aPC->Name() << "\n";
    }
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
Footer-MicMac-eLiSe-25/06/2007*/
