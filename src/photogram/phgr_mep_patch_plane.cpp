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

/*
    Temps pour 1000 SVD :       0.040899
    Temps pour 1000 Solve(8)  : 0.00645113
    Temps pour 1000000  AddEq  : 0.113502

Le parametre de taille n'a quasiment pas d'influence dans OneIterNbSomGlob 
    OneIterNbSomGlob(aHom,20,P8COL::green,Show);

Par contre, si on supprime TestEvalHomographie, le temps es / par 60.

Conclusion, pour optimiser les chances il faut tenter bcp de triangle, en faisant varier
les tailles d'initialisation. Pour chaque taille on prend un approch progressive, mais 
on ne fait qu'un seul test de TestEvalHomographie,


  A faire rajouter une observation.
  Mesure les temps de calcul des différentes briques :

     * calcul de SVD . Si calcul de SVD rapide, dq on fait une estim hom, on tente sa chance sur residu 3D:q
     * inversion
     * ajout une obs


  Algo :

    - a partir de germe (4 point)
    - file d'attente (heap) sur le critere de l'ecart / homogr courante

*/

#include "StdAfx.h"

double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH);



   //===============================================================
   //            
   //            Graphe "PRIMAL"
   //            
   //===============================================================

static const double TheDefautEcH = 1e20;

class cAttrSomOPP
{
    public :

       cAttrSomOPP() {}

       cAttrSomOPP(const Pt2dr & aP1,const Pt2dr & aP2,double aPds,int aNum) :
          mP1 (aP1),
          mP2 (aP2),
          mPds (aPds),
          mEcHom (TheDefautEcH),
          mNum (aNum),
          mCptF (0)
       {
       }

       ElCplePtsHomologues Cple() {return ElCplePtsHomologues(mP1,mP2,1.0);}

       Pt2dr mP1;
       Pt2dr mP2;
       double mPds;
       double mEcHom;
       int    mNum;   // identifiant
       int    mCptF;  // Compteur de face
};

class cAttrSomDualOPP;
class cAttrArcDualOPP;

class cAttrArcSomOPP
{
     public :
         cAttrArcSomOPP()  :
              mFInt (0)
         {
         }
         ElSom <cAttrSomDualOPP,cAttrArcDualOPP> * mFInt;
};



typedef  ElSom<cAttrSomOPP ,cAttrArcSomOPP>      tSomOPP;
typedef  ElArc<cAttrSomOPP,cAttrArcSomOPP>       tArcOPP;
typedef  ElGraphe<cAttrSomOPP,cAttrArcSomOPP>    tGrOPP;
typedef  ElSubGraphe<cAttrSomOPP,cAttrArcSomOPP> tSubGrBaseOPP;
typedef  ElSomIterator<cAttrSomOPP,cAttrArcSomOPP> tItSOPP;
typedef  ElArcIterator<cAttrSomOPP,cAttrArcSomOPP> tItAOPP;
typedef tSomOPP * tSomOPPPtr;


typedef cSubGrFlagSom<tSubGrBaseOPP>  tSubGrFlagOpp;

class cSubGrapheOPPPlan : public tSubGrBaseOPP
{
    public :
        Pt2dr  pt(tSomOPP & aS) 
        {
            return aS.attr().mP1;
        }
    
};

Pt2dr POfSomPtr(const tSomOPP * aSom) {return aSom->attr().mP1;}



class cCmpAttrSomOPP
{
     public :
           bool operator () (const tSomOPPPtr & aPS1, const tSomOPPPtr & aPS2)
           {
                 return aPS1->attr().mEcHom < aPS2->attr().mEcHom;
           }
};

typedef ElHeap<tSomOPPPtr,cCmpAttrSomOPP> tHeapSPPP;


   //===============================================================
   //            
   //            Graphe "DUAL"
   //            
   //===============================================================


class cAttrSomDualOPP
{
    public :
          cAttrSomDualOPP() :
              mHom            (cElHomographie::Id()),
              mRot            (ElRotation3D::Id),
              mRotPReSel      (ElRotation3D::Id)
           {}

           double DistRotPres(const cAttrSomDualOPP & anA2) const;

          cAttrSomDualOPP(const  ElSubFilo<tArcOPP *> & aFA,int aNum) :
              mEcHom (TheDefautEcH),
              mDejaInH  (false),
              mNum      (aNum),
              mOwnSc0   (1e20),
              mDoneHom        (false),
              mBestIncludeSc0 (1e10),
              mNumBestIS0     (-1),
              mPreselH        (false),
              mHom            (cElHomographie::Id()),
              mRot            (ElRotation3D::Id),
              mRotPReSel      (ElRotation3D::Id),
              mScore3D        (1e10)
          {
              ELISE_ASSERT(aFA.nb()==3,"cSomDualOPP");
              mSoms[0] = &(aFA[0]->s1());
              mSoms[1] = &(aFA[1]->s1());
              mSoms[2] = &(aFA[2]->s1());

              mC = (mSoms[0]->attr().mP1 + mSoms[1]->attr().mP1 + mSoms[2]->attr().mP1)/3.0;
          }

          tSomOPPPtr   mSoms[3];
          Pt2dr        mC;
          double       mEcHom;
          bool         mDejaInH ;  // Cpt in Hom
          int          mNum;
          double       mOwnSc0;
          bool         mDoneHom;
          double       mBestIncludeSc0;
          int          mNumBestIS0;
          bool         mPreselH;
          cElHomographie  mHom;
          ElRotation3D  mRot;
          ElRotation3D  mRotPReSel;
          Pt3dr        mMedRP;
          double       mScore3D;
          double       mMoyDist;
          double       mMaxDist;
};

double  cAttrSomDualOPP::DistRotPres(const cAttrSomDualOPP & anA2) const
{
    double aBSH = (ElAbs(1/mMedRP.z)  + ElAbs(1/anA2.mMedRP.z)) / 2.0;
    return DistRot(mRotPReSel,anA2.mRotPReSel,aBSH);
}


class cAttrArcDualOPP
{
     public :
         cAttrArcDualOPP()  
         {
         }
     private :
};


typedef  ElSom   <cAttrSomDualOPP,cAttrArcDualOPP>        tSomDualOPP;
typedef  ElArc   <cAttrSomDualOPP,cAttrArcDualOPP>        tArcDualOPP;
typedef  ElGraphe<cAttrSomDualOPP,cAttrArcDualOPP>        tGrDualOPP;
typedef  ElSubGraphe<cAttrSomDualOPP,cAttrArcDualOPP>     tSubGrDualOPP;
typedef  ElSomIterator<cAttrSomDualOPP,cAttrArcDualOPP>   tItDualSOPP;
typedef  ElArcIterator<cAttrSomDualOPP,cAttrArcDualOPP>   tItDualAOPP;

typedef tSomDualOPP * tSomDPtr;

class cCmpEcHomAttrFaceOPP
{
     public :
           bool operator () (const tSomDPtr & aPF1, const tSomDPtr & aPF2)
           {
                 return aPF1->attr().mEcHom < aPF2->attr().mEcHom;
           }
};

class cCmpSc3DAttrFaceOPP
{
     public :
           bool operator () (const tSomDPtr & aPF1, const tSomDPtr & aPF2)
           {
                 return aPF1->attr().mScore3D < aPF2->attr().mScore3D;
           }
};




typedef ElHeap<tSomDPtr,cCmpEcHomAttrFaceOPP> tHeapFPP;


 
   //===============================================================
   //            
   //            Planar Patch
   //            
   //===============================================================


class cOriPlanePatch
{
    public :
        cOriPlanePatch(   double aFoc,
                          const ElPackHomologue & aPack,
                          const ElPackHomologue & aPack150,
                          const ElPackHomologue & aPack30,
                          Video_Win * aW,
                          Pt2dr       aP0W,
                          double      aScaleW
                      );
         void operator()(tSomOPP*,tSomOPP*,bool);  // Delaunay call back


         void TestPt();
         void TestHomogrDual();
         void CalcAllHomAndSel();
         void TestHomogr();
         void ResetHom();
         void SolveHom();
         void OneIterNbSomGlob(int aNbGlobIn,int aNbGlobOut,int aNbCoul,bool Show);

         cInterfBundle2Image *  IBIOfNum(int aK);
         ElRotation3D * Sol() const ;
    private  :

         double EcHom(const cAttrSomOPP & anAtr) const
         {
             return square_euclid(mCurHom.Direct(anAtr.mP1)-anAtr.mP2);
         }

         double MoyenneEcStd(const std::vector<tSomOPP *> &) const;
         // 0 Lin ; 1 Angle ; 2 Bundle
         void  TestEvalHomographie(tSomDualOPP  &,bool Show,int aNumBundle,int aNbIter);
         void  TestEvalHomographie(const cElHomographie & aHom,tSomDualOPP & aFace,bool Show,int aNumBundle,int aNbIter);
         void AddHom(tSomOPP*,double aPds);
         void ReinitSom(tSomOPP* aSom);
         void ShowPoint(const Pt2dr &,double aRay,int coul) const;
         void ShowPoint(const tSomOPP &,double aRay,int coul) const;
         void ShowSeg(const Pt2dr & aP1,const Pt2dr& aP2,int aCoul) const;
         Pt2dr ToW(const Pt2dr & aP) const;
         Pt2dr FromW(const Pt2dr & aP) const;
         tSomOPP * GetSom(int aCoul);
         tSomDualOPP * GetFace(int aCoul);

         void LocalRestimateHomogr();
         void LocalAmelRot(const ElRotation3D & aR);

   // Appremment empire le resultat sur les rotation !!
         void  EstimStatHom(const cElHomographie & aHom,const  std::vector<tSomOPP *> & aVS0,double &aMaxDist,double & aMoyDist);
         cElHomographie ReestimHom(const cElHomographie & aHom, const double & aSeuilDist,double & aSigmaDist);


   //  Aproche par les faces
         bool MakeHomogrInitDual(std::vector<tSomDualOPP *>,bool aModeAffine,bool aShow);
         bool MakeHomogrInitDual(tSomDualOPP *,bool aShow);
         void ResetMakeHomDual(bool aModeAff);
         void InitHeapDual(const std::vector<tSomDualOPP *> & aVInit);
         void DualRecalculHom();

         void FaceEstimateEcH(tSomDualOPP * aF,bool Force);
         void AddFace2EstimHom(tSomDualOPP * aF,int aDelta);
         void InsertVoisDual(tSomDualOPP *);

         void SetExploredFace(tSomDualOPP * aF);
         void SetSelectedFace(tSomDualOPP * aF);
         void AddSetFaceSom(tSomDualOPP * aF,int aFlagFace,std::vector<tSomDualOPP *> & aVFace,int aFlagSom,std::vector<tSomOPP*>& aVSom);

         void  ReinitAll();
         void  ReinitFace(bool DejaInH,bool  EcH, bool Flag );
         void  ReinitSol(bool Cpt,bool  EcH, bool Flag );




         double                 mFoc;
         const ElPackHomologue&       mPack;
         const ElPackHomologue&       mPack150;
         const ElPackHomologue&       mPack30;
         Video_Win *            mW;
         Pt2dr                  mP0W;
         double                 mScaleW;
         std::vector<tSomOPP *> mVSom;
         std::vector<tSomOPP *> mVExploredSom;
         std::vector<tSomOPP *> mVSelectedSom;
         int                    mNbSom;
         tGrOPP                 mGrOPP;
         cSubGrapheOPPPlan      mSubGrFull;
         int                    mFlagVisitSomH;
         int                    mFlagSelectedSomH;
         tSubGrFlagOpp          mSubGrFlagVH;
         L2SysSurResol          mSysHom;
         bool                   mModeAff;
         cInterfBundle2Image *  mIBI_Lin;
         cInterfBundle2Image *  mIBI_Ang;
         cInterfBundle2Image *  mIBI_Bund;
         cElHomographie         mCurHom;
         cCmpAttrSomOPP         mCmpPPP;
         tHeapSPPP              mHeap;
         tGrDualOPP             mGrDual;
         tSubGrDualOPP          mSubGrDualFull;
         std::vector<tSomDualOPP *> mVFace;
         std::vector<tSomDualOPP *>  mExploredFaceH;
         std::vector<tSomDualOPP *>  mSelectedFaceH;
         cCmpEcHomAttrFaceOPP    mCmpF;
         tHeapFPP                mHeapF;
         int                    mFlagVisitFaceH;
         int                    mFlagSelectedFaceH;
         tSomDualOPP *          mRes;

};


    //  ==============  Graphisme   ================

Pt2dr cOriPlanePatch::ToW(const Pt2dr & aP) const { return (aP-mP0W) *mScaleW; }
Pt2dr cOriPlanePatch::FromW(const Pt2dr & aP) const { return aP/mScaleW + mP0W; }
void cOriPlanePatch::ShowPoint(const Pt2dr & aP,double aRay,int aCoul) const
{
   if (mW && (aCoul>=0)) 
      mW->draw_circle_abs(ToW(aP),aRay,mW->pdisc()(aCoul));
}
void  cOriPlanePatch::ShowPoint(const tSomOPP & aS,double aRay,int aCoul) const
{
    ShowPoint(aS.attr().mP1,aRay,aCoul);
}

void cOriPlanePatch::ShowSeg(const Pt2dr & aP1,const Pt2dr& aP2,int aCoul) const
{
    if (mW) mW->draw_seg(ToW(aP1),ToW(aP2),mW->pdisc()(aCoul) );
}
   // ==========================================================


void cOriPlanePatch::operator()(tSomOPP* aS1,tSomOPP* aS2,bool)
{
    ShowSeg(aS1->attr().mP1,aS2->attr().mP1,P8COL::red);

    if (! mGrOPP.arc_s1s2(*aS1,*aS2))
    {
        cAttrArcSomOPP anAttr;
        mGrOPP.add_arc(*aS1,*aS2,anAttr,anAttr);
    }
}



tSomOPP * cOriPlanePatch::GetSom(int aCoul)
{
   Clik aCl = mW->clik_in();
   Pt2dr aP = FromW(aCl._pt);
   double aDistMin = 1e20;
   tSomOPP * aRes = 0;
   
   for (int aK=0 ; aK<int(mVSom.size()) ; aK++)
   {
       double aD = euclid(aP,mVSom[aK]->attr().mP1);
       if (aD<aDistMin)
       {
           aDistMin = aD;
           aRes = mVSom[aK];
       }
   }

   ShowPoint(*aRes,3.0,P8COL::white);
   return aRes;
}

tSomDualOPP * cOriPlanePatch::GetFace(int aCoul)
{
   Clik aCl = mW->clik_in();
   Pt2dr aP = FromW(aCl._pt);
   double aDistMin = 1e20;
   tSomDualOPP * aRes = 0;
   
   for (int aK=0 ; aK<int(mVFace.size()) ; aK++)
   {
       double aD = euclid(aP,mVFace[aK]->attr().mC);
       if (aD<aDistMin)
       {
           aDistMin = aD;
           aRes = mVFace[aK];
       }
   }

   ShowPoint(aRes->attr().mC,3.0,P8COL::white);
   return aRes;
}





// (a + b x1 + c y1) = x2 (1+g x1 + h y1)
// (d + e x1 + f y1) = y2 (1+g x1 + h y1)


void cOriPlanePatch::ReinitSom(tSomOPP* aSom)
{
    aSom->flag_set_kth_false(mFlagVisitSomH);
    aSom->attr().mEcHom = TheDefautEcH;
}

void cOriPlanePatch::AddHom(tSomOPP* aSom,double aPds)
{
     static double aCoeff[8];


     const cAttrSomOPP & anAttr = aSom->attr();
     double aX1 =  anAttr.mP1.x;
     double aY1 =  anAttr.mP1.y;
     double aX2 =  anAttr.mP2.x;
     double aY2 =  anAttr.mP2.y;
    
     aCoeff[0] = 1;
     aCoeff[1] = aX1;
     aCoeff[2] = aY1;
     aCoeff[3] = aCoeff[4] = aCoeff[5] = 0;
     if ( mModeAff)
     {
         aCoeff[6] =  aCoeff[7] = 0.0;
     }
     else
     {
        aCoeff[6]  = -aX2 * aX1;
        aCoeff[7] =  -aX2 * aY1;
     }
     mSysHom.GSSR_AddNewEquation(1.0,aCoeff,aX2,0);


     aCoeff[0] = aCoeff[1] = aCoeff[2] = 0;
     aCoeff[3] = 1;
     aCoeff[4] = aX1;
     aCoeff[5] = aY1;
     if (mModeAff)
     {
         aCoeff[6] =  aCoeff[7] = 0.0;
     }
     else
     {
        aCoeff[6]  = -aY2 * aX1;
        aCoeff[7]  = -aY2 * aY1;
     }
     mSysHom.GSSR_AddNewEquation(1.0,aCoeff,aY2,0);
}


void cOriPlanePatch::ResetHom()
{
    mSysHom.GSSR_Reset(true);
    mSysHom.SetPhaseEquation(0);
    for (int aK=0 ; aK< mNbSom ; aK++)
    {
        mVSom[aK]->flag_set_kth_false(mFlagVisitSomH);
        mVSom[aK]->attr().mEcHom = TheDefautEcH;
    }
}


void cOriPlanePatch::LocalAmelRot(const ElRotation3D & aR)
{
}

void  cOriPlanePatch::TestEvalHomographie(tSomDualOPP & aFace,bool Show,int aNumBundle,int aNbIter)
{
      TestEvalHomographie(aFace.attr().mHom,aFace,Show,aNumBundle,aNbIter);
}

void  cOriPlanePatch::TestEvalHomographie(const cElHomographie & aHom,tSomDualOPP & aFace,bool Show,int aNumBundle,int aNbIter)
{
     // const cElHomographie & aHom =  aFace.attr().mHom ;
     if (0)
     {
         for (int aK=0 ; aK<int(mVExploredSom.size()) ; aK++)
             std::cout << "DHHHH  " << sqrt(EcHom(mVExploredSom[aK]->attr())) *mFoc << "\n";
     }


     cResMepRelCoplan aRMC =  ElPackHomologue::MepRelCoplan(1.0,aHom,tPairPt(Pt2dr(0,0),Pt2dr(0,0)));
     const std::list<cElemMepRelCoplan>  & aLSolPl = aRMC.LElem();

     ElRotation3D aBestR = ElRotation3D::Id;
     double aBestScore = 1e20;
     for (std::list<cElemMepRelCoplan>::const_iterator itS = aLSolPl.begin() ; itS != aLSolPl.end() ; itS++)
     {
        if ( itS->PhysOk())
        {
             ElRotation3D aR = itS->Rot();
             aR = aR.inv();
             double aScore = ProjCostMEP(mPack,aR,0.1);
             if (Show)
             {
                std::cout << "SC " <<  aScore * mFoc 
                       << " PR " << ProjCostMEP(mPack,aR,0.1) * mFoc  
                       << " Pv " << PVCostMEP(mPack,aR,0.1) * mFoc  
                       << "\n";
             }
/*
*/
             if (aScore<aBestScore)
             {
                 aBestScore = aScore;
                 aBestR = aR;
             }
        }
     }
     ElRotation3D aBestR0 = aBestR;
     cInterfBundle2Image * anIBI =  IBIOfNum(aNumBundle);

     double aPrCostIn = (Show ?   ProjCostMEP(mPack,aBestR,0.1) : 0.0);
     double anEr = anIBI->ErrInitRobuste(aBestR);
     anEr =  anIBI->ResiduEq(aBestR,anEr);
     double anEr0 = anEr;
     for (int aK=0 ; aK<aNbIter ; aK++)
     {
           aBestR = anIBI->OneIterEq(aBestR,anEr);
     }
     if (Show)
     {
        std::cout << "COST Hom " << anEr0*mFoc 
                  << " => " << anEr*mFoc 
                  << " Proj "  << aPrCostIn*mFoc 
                  << " => " << ProjCostMEP(mPack,aBestR,0.1) *mFoc << "\n";
     }

     if (anEr< aFace.attr().mScore3D)
     {
         aFace.attr().mScore3D = ProjCostMEP(mPack,aBestR,0.1) *mFoc;
         aFace.attr().mRot     = aBestR;
     }
}
 

void  cOriPlanePatch::SolveHom()
{ 
    if (mModeAff)
    {
        static double aCoeff[8];
        for (int aCstr=6 ; aCstr<8 ; aCstr++)
        {
            for (int aK=0 ; aK< 8 ; aK++)
                 aCoeff[aK] = (aK==aCstr);
            mSysHom.GSSR_AddNewEquation(1.0,aCoeff,0,0);
        }
    }
    Im1D_REAL8 aSol =      mSysHom.GSSR_Solve (0);
    double * aDS = aSol.data();

    cElComposHomographie aHX(aDS[1],aDS[2],aDS[0]);
    cElComposHomographie aHY(aDS[4],aDS[5],aDS[3]);
    cElComposHomographie aHZ(aDS[6],aDS[7],     1);

    mCurHom =  cElHomographie(aHX,aHY,aHZ);
}


void  cOriPlanePatch::EstimStatHom(const cElHomographie & aHom,const  std::vector<tSomOPP *> & aVS0,double &aMaxDist,double & aMoyDist)
{
    aMaxDist = 0.0;
    aMoyDist = 0.0;
    for (int aK=0 ; aK<int(aVS0.size()) ; aK++)
    {
        const cAttrSomOPP & anAtr =   aVS0[aK]->attr();
        double aDist = euclid(aHom.Direct(anAtr.mP1) - anAtr.mP2);
        ElSetMax(aMaxDist,aDist);
        aMoyDist += aDist;
    }

    aMoyDist /= aVS0.size();
}

cElHomographie cOriPlanePatch::ReestimHom(const cElHomographie & aHom, const double & aSeuilDist,double & aSigmaDist)
{
    ResetHom();
    for (int aK=0 ; aK<int(mVSom.size()) ; aK++)
    {
        const cAttrSomOPP & anAtr =   mVSom[aK]->attr();
        double aDist = euclid(aHom.Direct(anAtr.mP1) - anAtr.mP2);

        if (aDist < aSeuilDist)
        {
            double aPds = 1/(1+ElSquare(aDist/aSigmaDist));
            AddHom(mVSom[aK],aPds);
        }
    }

    SolveHom();
    return mCurHom;
}

/*
    double aSigma0 = aMoyDist * 2.0;
    double aSeuil = aMaxDist * 1.5;
*/



     //=============================================
     //
     //     Approche "duale"
     //
     //=============================================

void cOriPlanePatch::AddFace2EstimHom(tSomDualOPP * aF,int aDelta)
{
    cAttrSomDualOPP &  anAF = aF->attr();

    if (anAF.mDejaInH && (aDelta==1)) return;
    anAF.mDejaInH  = (aDelta==1);

    for (int aK=0 ; aK<3 ; aK++)
    {
         tSomOPPPtr aS = anAF.mSoms[aK];
         // Cas entrant 
         if (aDelta==1)
         {
             if (aS->attr().mCptF==0)  // passe 0 a 1
             {
                 AddHom(aS,1.0);
             }
         }
         else if (aS->attr().mCptF==1) // cas sortant, passe de 1 a 0
         {
            AddHom(aS,-1.0);
         }
         aS->attr().mCptF += aDelta;
    }
}

void cOriPlanePatch::AddSetFaceSom
     (
           tSomDualOPP * aF,
           int                        aFlagFace,
           std::vector<tSomDualOPP *> & aVFace,
           int                        aFlagSom,
           std::vector<tSomOPP *>     & aVSom
     )
{
    if (! aF->flag_kth(aFlagFace))
    {
        aF->flag_set_kth_true(aFlagFace);
        aVFace.push_back(aF);
        cAttrSomDualOPP &  anAF = aF->attr();
        for (int aK=0 ; aK<3 ; aK++)
        {
             tSomOPPPtr aS = anAF.mSoms[aK];
             if (! aS->flag_kth(aFlagSom))
             {
                 aVSom.push_back(aS);
                 aS->flag_set_kth_true(aFlagSom);
             }
        }
    }
}
void cOriPlanePatch::SetExploredFace(tSomDualOPP * aF)
{
     AddSetFaceSom(aF,mFlagVisitFaceH,mExploredFaceH,mFlagVisitSomH,mVExploredSom);
}

void cOriPlanePatch::SetSelectedFace(tSomDualOPP * aF)
{
     AddSetFaceSom(aF,mFlagSelectedFaceH,mSelectedFaceH,mFlagSelectedSomH,mVSelectedSom);
}




void cOriPlanePatch::FaceEstimateEcH(tSomDualOPP * aF,bool Force)
{
    cAttrSomDualOPP &  anAF = aF->attr();
    if ((anAF.mEcHom != TheDefautEcH)  && (! Force))return;

    anAF.mEcHom = 0;

    for (int aK=0 ; aK<3 ; aK++)
    {
         cAttrSomOPP & anAS = anAF.mSoms[aK]->attr();
         if ((anAS.mEcHom == TheDefautEcH) || Force)
         {
             anAS.mEcHom = EcHom(anAS);
         }
         ElSetMax(anAF.mEcHom,anAS.mEcHom);
    }
}



void  cOriPlanePatch::ResetMakeHomDual(bool aModeAff)
{
    mModeAff = aModeAff;
    mExploredFaceH.clear();
    mSelectedFaceH.clear();
    mVExploredSom.clear();
    mVSelectedSom.clear();
    mSysHom.GSSR_Reset(true);
    mSysHom.SetPhaseEquation(0);
    mHeapF.clear();
}

void  cOriPlanePatch::ReinitFace(bool DejaInH,bool  EcH, bool Flag )
{
    for (int aKF=0 ; aKF<int(mExploredFaceH.size()) ; aKF++)
    {
         tSomDualOPP * aFace = mExploredFaceH[aKF];
         cAttrSomDualOPP &  anAF =aFace->attr();
         if (DejaInH) anAF.mDejaInH= false;
         if (EcH)  anAF.mEcHom = TheDefautEcH;
         if (Flag)
         {
             aFace->flag_set_kth_false(mFlagVisitFaceH);
             aFace->flag_set_kth_false(mFlagSelectedFaceH);
         }
    }
}

void  cOriPlanePatch::ReinitSol(bool Cpt,bool  EcH, bool Flag )
{
    for (int aKS=0 ; aKS<int( mVExploredSom.size()) ; aKS++)
    {
         tSomOPP * aSom = mVExploredSom[aKS];
         cAttrSomOPP & anAS = aSom->attr();
         if ( Cpt) anAS.mCptF = 0;
         if (EcH)  anAS.mEcHom = TheDefautEcH;
         if (Flag)
         {
             aSom->flag_set_kth_false(mFlagVisitSomH);
             aSom->flag_set_kth_false(mFlagSelectedSomH);
         }
    }
}

void  cOriPlanePatch::ReinitAll()
{
     ReinitFace(true,true,true);
     ReinitSol(true,true,true);
}


// AFinitte transformant ak en bk


void cOriPlanePatch::InitHeapDual(const std::vector<tSomDualOPP *> & aVInit)
{
   for (int aKF=0 ; aKF<int(aVInit.size()) ; aKF++)
   {
       AddFace2EstimHom(aVInit[aKF],1);
   }

   if (aVInit.size()==1)
   {
      cAttrSomDualOPP &  anAF =aVInit[0]->attr();
      ElAffin2D anAfinity = ElAffin2D::FromTri2Tri
                       (
                           anAF.mSoms[0]->attr().mP1, anAF.mSoms[1]->attr().mP1, anAF.mSoms[2]->attr().mP1,
                           anAF.mSoms[0]->attr().mP2, anAF.mSoms[1]->attr().mP2, anAF.mSoms[2]->attr().mP2
                       );
      mCurHom = anAfinity.ToHomographie();
   }
   else
   {
      SolveHom();
   // Sinon les parametre du sys ne sont pas bon
      if (mModeAff)
      {
         ReinitAll();
         ResetMakeHomDual(false);
         for (int aKF=0 ; aKF<int(aVInit.size()) ; aKF++)
         {
             AddFace2EstimHom(aVInit[aKF],1);
         }
      }
   }

   for (int aKF=0 ; aKF<int(aVInit.size()) ; aKF++)
   {
       FaceEstimateEcH(aVInit[aKF],false);
       aVInit[aKF]->attr().mEcHom = 0.0; // On triche pour qu'elle ressortent du tas en premier
       mHeapF.push(aVInit[aKF]);
       SetExploredFace(aVInit[aKF]);
   }
}



void cOriPlanePatch::InsertVoisDual(tSomDualOPP * aF)
{
    // mSelectedFaceH.push_back(aF);
    SetSelectedFace(aF);
    AddFace2EstimHom(aF,1);
    
    for (tItDualAOPP itA=aF->begin(mSubGrDualFull) ; itA.go_on() ; itA++)
    {
        tSomDualOPP * aF2 = &((*itA).s2());
        if (! aF2->flag_kth(mFlagVisitFaceH))
        {
           FaceEstimateEcH(aF2,false);
           SetExploredFace(aF2);
           mHeapF.push(aF2);
        }
    }
}

void cOriPlanePatch::DualRecalculHom()
{
    SolveHom();
    std::vector<tSomDualOPP *> aVF = mHeapF.Els();
    mHeapF.clear();
    for (int aKF=0 ; aKF<int(aVF.size()) ; aKF++)
    {
         FaceEstimateEcH(aVF[aKF],true);
         mHeapF.push(aVF[aKF]);
    }
}





bool cOriPlanePatch::MakeHomogrInitDual(std::vector<tSomDualOPP *> aVInit,bool aModeAff,bool aShow)
{
  // Remise a zero des compteur globaux
   aModeAff = aModeAff && (aVInit.size() !=1); // Si Taille 1, triangle et prise en charge directe
   ResetMakeHomDual(aModeAff);


  // initialisation de la file
   InitHeapDual(aVInit);

   ELISE_PTR_U_INT aCpt=0;
    
   while (mVSelectedSom.size() < 20)
   {
         tSomDualOPP * aF=0;
         bool Ok = mHeapF.pop(aF);
         if (! Ok) return false;

         aCpt = aCpt ^ (ELISE_PTR_U_INT)(aF);

         ELISE_ASSERT(Ok,"Incoher in pop; cOriPlanePatch::MakeHomogrInitDual");

         InsertVoisDual(aF);
         if (aShow)
         {
            ShowPoint(aF->attr().mC,3.0,P8COL::yellow);
            // mW->clik_in();
         }
         if (mSelectedFaceH.size()==10) DualRecalculHom();
   }
   if (aShow)
      std::cout << "CksS " << aCpt << "\n";
   // Re
   return true;
}

double  cOriPlanePatch::MoyenneEcStd(const std::vector<tSomOPP *> & aV) const
{
    double aScore = 0;
    for (int aKS=0 ; aKS<int(aV.size()) ; aKS++)
    {
         aScore += EcHom(aV[aKS]->attr());
    }
    return sqrt(aScore/aV.size()) * mFoc;
}

bool BetterScoreNumOrEg(double aSc1,int aNum1,double aSc2,int aNum2)
{
   return  (aSc1 < aSc2) || ((aSc1==aSc2) && (aNum1 >= aNum2));
}

bool cOriPlanePatch::MakeHomogrInitDual(tSomDualOPP * aF0,bool aShow)
{
    std::vector<tSomDualOPP *> aVF0;
    aVF0.push_back(aF0);
    bool Ok = MakeHomogrInitDual(aVF0,true,aShow);
    if (!Ok) return false;
    SolveHom();
    aF0->attr().mHom = mCurHom;
    aF0->attr().mDoneHom = true;

    EstimStatHom(mCurHom,mVSelectedSom,aF0->attr().mMaxDist,aF0->attr().mMoyDist);

    double aScore = MoyenneEcStd(mVSelectedSom);
    cAttrSomDualOPP & anAF0 =  aF0->attr();
    anAF0.mOwnSc0 = aScore;

    for (int aKF=0 ; aKF<int(mSelectedFaceH.size()) ; aKF++)
    {
        cAttrSomDualOPP &  anAF  = mSelectedFaceH[aKF]->attr();
        if (BetterScoreNumOrEg(aScore,anAF0.mNum,anAF.mBestIncludeSc0,anAF.mNumBestIS0))
        {
           anAF.mBestIncludeSc0 = aScore;
           anAF.mNumBestIS0     = anAF0.mNum;
        }
    }

    if (aShow)
    {
       std::cout << "Score=" << aScore << " Nb=" << mVSelectedSom.size() << "\n";
    }
    ReinitAll();
    return true;
}


void cOriPlanePatch::CalcAllHomAndSel()
{
    ElTimer aChrono;
    for (int aK=0 ; aK<int(mVFace.size()) ; aK++)
    {
        MakeHomogrInitDual(mVFace[aK],false);
    }
    double aMinSc0 = 1e20;
    int aNbPres = 0;

    for (int aK=0 ; aK<int(mVFace.size()) ; aK++)
    {
        tSomDualOPP * aF = mVFace[aK];
        cAttrSomDualOPP &  anAF  = aF->attr();
        if (anAF.mDoneHom)
        {
           ElSetMin(aMinSc0,anAF.mOwnSc0);
           if (BetterScoreNumOrEg(anAF.mOwnSc0,anAF.mNum,anAF.mBestIncludeSc0,anAF.mNumBestIS0))
           {
               bool IsMinLoc = true;

               for (tItDualAOPP itA=aF->begin(mSubGrDualFull) ; itA.go_on() ; itA++)
               {
                   if ((*itA).s2().attr().mOwnSc0 < anAF.mOwnSc0)
                      IsMinLoc = false;
               }

               int aCoul = IsMinLoc ? P8COL::red : P8COL::cyan;

               ShowPoint(anAF.mC,2.0,aCoul);
               ShowPoint(anAF.mC,4.0,aCoul);
               ShowPoint(anAF.mC,6.0,aCoul);
               if (IsMinLoc) 
               {
                  anAF.mPreselH= true;
                  aNbPres++;
               // std::cout << "SCORE " << anAF.mOwnSc0 << "\n";
               }
           }
        }
    }

    if (mW)
    {
       std::cout << "TIME-Hom " << aChrono.uval()  << " NbS=" << aNbPres << " SCMin " << aMinSc0  << "\n";
    }
    ElTimer aChrono2;
    std::vector<tSomDualOPP*> aVPreselH;
    for (int aK=0 ; aK<int(mVFace.size()) ; aK++)
    {
        tSomDualOPP * aF = mVFace[aK];
        cAttrSomDualOPP &  anAF  = aF->attr();
        if (anAF.mPreselH)
        {
            TestEvalHomographie(*aF,false,0,10);
            if (0)
            {
                cElHomographie aHom = aF->attr().mHom;
                for (int aK=0 ; aK< 2 ; aK++)
                {
                   aHom = ReestimHom(aHom,1.0*aF->attr().mMaxDist,aF->attr().mMoyDist);
                }
                TestEvalHomographie(aHom,*aF,false,0,10);
            }
            anAF.mRotPReSel = anAF.mRot;
            anAF.mMedRP = MedianNuage(mPack30,anAF.mRot);
            aVPreselH.push_back(aF);
            // TestEvalHomographie(mVFace[aK]->attr().mHom,false,0,20);
        }
    }
    if (mW)
    {
       std::cout << "TIME-R0 " << aChrono2.uval()   << "\n\n";
    }


    cCmpSc3DAttrFaceOPP aCmpS3d;
    std::sort(aVPreselH.begin(),aVPreselH.end(),aCmpS3d);


    // On selectione les 4 premieres qui ne convergent pas vers le meme point
    
    std::vector<tSomDualOPP*> aVSelFinale;
    for (int aKP=0 ; (aKP<int(aVPreselH.size())) && (aVSelFinale.size() < 6)  ; aKP++)
    {
         bool GotNear = false;
         for (int aKF=0 ; (aKF<int(aVSelFinale.size())) && (!GotNear) ; aKF++)
         {
             double aDist = aVPreselH[aKP]->attr().DistRotPres(aVSelFinale[aKF]->attr());
             if (aDist < 5e-3)
             {
                GotNear = true;
                // double aDist = DistRot
             }
         }
         if (! GotNear) 
         {
            tSomDualOPP * aF =  aVPreselH[aKP];
            cAttrSomDualOPP &  anAF  = aF->attr();
            anAF.mScore3D = 1e5;
            TestEvalHomographie(*aF,false,2,10);
            aVSelFinale.push_back(aF);

            if (mW)
                std::cout << "COST FINAL " << aF->attr().mScore3D << "\n";
         
            int aCoul = P8COL::yellow;
            ShowPoint(anAF.mC,2.0,aCoul);
            ShowPoint(anAF.mC,4.0,aCoul);
            ShowPoint(anAF.mC,6.0,aCoul);
            if ((mRes==0) || (mRes->attr().mScore3D > aF->attr().mScore3D))
               mRes = aF;
         }
    }

    if (mW && mRes)
       std::cout << " FiiiiNNN  " <<  ProjCostMEP(mPack,mRes->attr().mRot,0.1)*mFoc << "\n";

}




void cOriPlanePatch::TestHomogrDual()
{
   tSomDualOPP * aFace = GetFace(P8COL::white);
   MakeHomogrInitDual(aFace,true);

   std::cout << "FACE " << aFace->attr().mOwnSc0 << "\n";

   TestEvalHomographie(*aFace,true,2,10);
/*
   cElHomographie aHom = aFace->attr().mHom;
   for (int aK=0 ; aK< 2 ; aK++)
   {
      aHom = ReestimHom(aHom,1.0*aFace->attr().mMaxDist,aFace->attr().mMoyDist);
   }
   TestEvalHomographie(aHom,*aFace,true,0,10);
*/
   // TestEvalHomographie(*aFace,true,2,20);


   std::cout << "\n";
}



     //=============================================
     //
     //
     //
     //=============================================

extern bool ShowStatMatCond;

cInterfBundle2Image *  cOriPlanePatch::IBIOfNum(int aK)
{
    if (aK==0) return mIBI_Lin;
    if (aK==1) return mIBI_Ang;
    if (aK==2) return mIBI_Bund;

    ELISE_ASSERT(false,"cOriPlanePatch::IBIOfNum");
    return 0;
}

cOriPlanePatch::cOriPlanePatch
( 
      double                  aFoc,
      const ElPackHomologue & aPack,
      const ElPackHomologue & aPack150,
      const ElPackHomologue & aPack30,
      Video_Win * aW,
      Pt2dr       aP0W,
      double      aScaleW
)  :
   mFoc         (aFoc),
   mPack        (aPack),
   mPack150     (aPack150),
   mPack30      (aPack30),
   mW           (aW),
   mP0W         (aP0W),
   mScaleW      (aScaleW),
   mFlagVisitSomH  (mGrOPP.alloc_flag_som()),
   mFlagSelectedSomH  (mGrOPP.alloc_flag_som()),
   mSubGrFlagVH (mSubGrFull,mFlagVisitSomH),
   mSysHom      (8),
   mModeAff     (false),
   mIBI_Lin     (cInterfBundle2Image::LinearDet(mPack,mFoc)),
   mIBI_Ang     (cInterfBundle2Image::LineariseAngle(mPack,mFoc,true)),
   mIBI_Bund    (cInterfBundle2Image::Bundle(mPack,mFoc,true)),
   mCurHom      (cElHomographie::Id()),
   mHeap        (mCmpPPP),
   mHeapF       (mCmpF),
   mFlagVisitFaceH  (mGrDual.alloc_flag_som()),
   mFlagSelectedFaceH  (mGrDual.alloc_flag_som()),
   mRes                (0)
{
    ShowStatMatCond = false;
    // if (mW) mW->clear();
    int aCpt=0;
    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
         tSomOPP & aSom = mGrOPP.new_som(cAttrSomOPP(itP->P1(),itP->P2(),itP->Pds(),aCpt));
         mVSom.push_back(&aSom);
         if (mW) ShowPoint(aSom,3.0,P8COL::green);
         aCpt++;
    }
    mNbSom = (int)mVSom.size();

    if (mW) 
       std::cout << "ENETR DELAU , Nb " << aPack.size() << " \n";

    ElTimer aChrono;
    Delaunay_Mediatrice
    (
         &(mVSom[0]),
         &(mVSom[0])+mVSom.size(),
          POfSomPtr,
       *this,
       1e10,
       (tSomOPP **) 0
    );


    ElTimer aChronoD;
    // Graphe dual
    {
         int aCpt = 0;
         ElPartition<tArcOPP *> aPart0;
         bool OkFT = all_face_trigo(mGrOPP,mSubGrFull,aPart0);
         ELISE_ASSERT(OkFT,"Pb Graphe Dual in  cOriPlanePatch::cOriPlanePatch");
         ElPartition<tArcOPP *> aPartA;
         make_real_face(mGrOPP,aPartA,aPart0);

         for (int aKF=0 ; aKF<int(aPartA.nb()) ; aKF++)
         {
             ElSubFilo<tArcOPP *> aFaceA = aPartA[aKF];
             if (aFaceA.nb() != 3)
             {
                if (mW)
                   std::cout << "NB  " << aFaceA.nb() << "\n";
             }
             else
             {
                  tSomDualOPP & aFaceDual = mGrDual.new_som(cAttrSomDualOPP(aFaceA,aCpt));
                  mVFace.push_back(&aFaceDual);
                  for (int aKA=0 ; aKA < aFaceA.nb() ; aKA++)
                  {
                        aFaceA[aKA]->attr().mFInt = & aFaceDual;
                  }
                  aCpt++;
             }
         }

         for (tItSOPP itS=mGrOPP.begin(mSubGrFull) ; itS.go_on() ; itS++)
         {
             for (tItAOPP itA=(*itS).begin(mSubGrFull) ; itA.go_on() ; itA++)
             {
                  tSomDualOPP  * aF1 = (*itA).attr().mFInt;
                  tSomDualOPP  * aF2 = (*itA).arc_rec().attr().mFInt;
                  if ((aF1!=0) && (aF2!=0) && (aF1<aF2))
                  {
                       cAttrArcDualOPP anAttr;
                       mGrDual.add_arc(*aF1,*aF2,anAttr);
                  }
             }
         }

         if (0)
         {
            for (tItDualSOPP itS=mGrDual.begin(mSubGrDualFull) ; itS.go_on() ; itS++)
            {
                ShowPoint((*itS).attr().mC,3.0,P8COL::cyan);
            }

            for (tItDualSOPP itS=mGrDual.begin(mSubGrDualFull) ; itS.go_on() ; itS++)
            {
                for (tItDualAOPP itA=(*itS).begin(mSubGrDualFull) ; itA.go_on() ; itA++)
                {
                     tSomDualOPP * aS1 = & ((*itA).s1());
                     tSomDualOPP * aS2 = & ((*itA).s2());
                     if (aS1 < aS2)
                        ShowSeg(aS1->attr().mC,aS2->attr().mC,P8COL::yellow);
                }
            }
             
         }

     }
     if (mW)
        std::cout << "Time Delaunay " << aChrono.uval() << " " << aChronoD.uval() << "\n";

     CalcAllHomAndSel();

    if (aW)
    {
        while (0)
        {
              TestHomogrDual();
        }
    }
}


ElRotation3D * cOriPlanePatch::Sol() const 
{
   if (! mRes) return 0;
   return new ElRotation3D(mRes->attr().mRot);
}


ElRotation3D * TestOriPlanePatch
     (
         double                  aFoc,
         const ElPackHomologue & aPack,
         const ElPackHomologue & aPack150,
         const ElPackHomologue & aPack30,
         Video_Win * aW,
         Pt2dr       aP0W,
         double      aScaleW
            
     )
{
    cOriPlanePatch anOPP(aFoc,aPack,aPack150,aPack30,aW,aP0W,aScaleW);
    return anOPP.Sol();

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
