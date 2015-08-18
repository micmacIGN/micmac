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

// bool DebugBundleGen=false;

class cBGC3_Modif2D ; //   : public cBasicGeomCap3D
class cPolynomial_BGC3M2D ;//  : public cBGC3_Modif2D
class cPolynBGC3M2D_Formelle; // : public cGenPDVFormelle
class cOneEq_PBGC3M2DF;


/*

   N ((i,j)+D(i,j)) = R N(i,j) = (I + W e ^ N(i,j))
  
   (Id + J * D(i,j) )  = I + W e ^ N(i,j))

    D(i,j) = J-1 W ^ N(i,j)

*/

class cBGC3_Modif2D  : public cBasicGeomCap3D
{
      public : 
           cBGC3_Modif2D(cBasicGeomCap3D * aCam0);




           virtual ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const ;
           virtual Pt2dr    Ter2Capteur   (const Pt3dr & aP) const ;
           Pt2dr            Ter2CapteurSsCorrec   (const Pt3dr & aP) const ;
           virtual Pt2di    SzBasicCapt3D() const ;
           virtual double ResolSolOfPt(const Pt3dr &) const ;
           virtual bool  CaptHasData(const Pt2dr &) const ;
           virtual bool     PIsVisibleInImage   (const Pt3dr & aP) const ;

  // Optical center 
           virtual bool     HasOpticalCenterOfPixel() const; // 1 - They are not alway defined
  // When they are, they may vary, as with push-broom, Def fatal erreur (=> Ortho cam)
           virtual Pt3dr    OpticalCenterOfPixel(const Pt2dr & aP) const ;

           inline Pt2dr CamInit2CurIm(const Pt2dr & aP) const{return aP+DeltaCamInit2CurIm(aP);}
           inline Pt2dr CurIm2CamInit(const Pt2dr & aP) const{return aP+DeltaCurIm2CamInit(aP);}

           cBasicGeomCap3D * CamSsCor();

      protected  : 
            cBasicGeomCap3D * mCam0;
            Pt2di  mSz;

      private  : 

            // Ter2Cam (x) = x + DifCorTer2Cal(x)

            virtual Pt2dr DeltaCamInit2CurIm(const Pt2dr & aP) const = 0;
            Pt2dr   DeltaCurIm2CamInit(const Pt2dr & aP) const ;
            

};


class cPolynomial_BGC3M2D  : public cBGC3_Modif2D
{
      public : 
           cPolynomial_BGC3M2D(cBasicGeomCap3D * aCam0,int aDegree,double aRandomPert);
           Pt2dr DeltaCamInit2CurIm(const Pt2dr & aP) const ;
           inline Pt2dr ToPNorm(const Pt2dr aP) const {return (aP-mCenter)/mAmpl;}
           inline Pt2dr FromPNorm(const Pt2dr aP) const {return aP*mAmpl + mCenter;}

           std::vector<double> & Cx();
           std::vector<double> & Cy();
           inline int DegX(int aK) const {return mDegX.at(aK);}
           inline int DegY(int aK) const {return mDegY.at(aK);}
           inline const int & DegreMax()   const {return mDegreMax;}
           inline const double  &       Ampl() const {return mAmpl;}
           inline const Pt2dr &      Center() const {return mCenter;}
           void Show() const;
      private : 
           void Show(const std::string & aMes,const std::vector<double> & aCoef) const;
           void ShowMonome(const std::string & , int aDeg) const;
           void SetPow(const Pt2dr & aPN) const;
 
           int                 mDegreMax;
           Pt2dr               mCenter;
           double              mAmpl;
           std::vector<double> mCx;
           std::vector<double> mCy;

           static std::vector<int> mDegX;
           static std::vector<int> mDegY;


           static std::vector<double> mPowX;
           static std::vector<double> mPowY;
           mutable Pt2dr  mCurPPow;
};

class cOneEq_PBGC3M2DF : public cElemEqFormelle,
                         public cObjFormel2Destroy

{
    public :
       cOneEq_PBGC3M2DF(cPolynBGC3M2D_Formelle &,std::vector<double > &);

       Fonc_Num  EqFormProjCor(Pt2d<Fonc_Num> aP);
       
   private :
       std::vector<Fonc_Num>     mVFCoef;
       cPolynBGC3M2D_Formelle *  mPF;
       cPolynomial_BGC3M2D*      mCamCur;
};


class cCellPolBGC3M2DForm
{
      public :
          cCellPolBGC3M2DForm(Pt2dr mPt,cPolynBGC3M2D_Formelle * aPF);
          cCellPolBGC3M2DForm();
          void InitRep(cPolynBGC3M2D_Formelle * aPF);
          void SetGrad(const Pt2dr & aGX,const Pt2dr & aGy);
      
          Pt2dr  mPt;
          Pt3dr  mNorm;
          Pt2dr  mDerPnlRot[3];
          bool   mActive;
          Pt2dr  mValDep[3];
          bool   mHasDep;
};

class cPolynBGC3M2D_Formelle : public cGenPDVFormelle
{

    public  :
         friend class cOneEq_PBGC3M2DF;
         friend class cCellPolBGC3M2DForm;

         cPolynBGC3M2D_Formelle(cSetEqFormelles & aSet,cPolynomial_BGC3M2D aCam0,bool GenCodeAppui,bool GenCodeAttach,bool   GenCodeRot);
         void GenerateCode(Pt2d<Fonc_Num>,const std::string &,cIncListInterv &);
         cIncListInterv & IntervAppuisPtsInc() ;
         void PostInit();
         const cBasicGeomCap3D * GPF_CurBGCap3D() const ;
         cBasicGeomCap3D * GPF_NC_CurBGCap3D() ;
         Pt2dr AddEqAppuisInc(const Pt2dr & aPIm,double aPds, cParamPtProj &,bool IsEqDroite);
         
         const cPolynomial_BGC3M2D *  TypedCamCur() const { return & mCamCur; }
         cPolynomial_BGC3M2D *  TypedCamCur() { return & mCamCur; }
         void AddEqAttachGlob(double aPds,bool Cur,int aNbPts,CamStenope * aKnownSol);
         cBasicGeomCap3D *   CamSsCorr() const ;

         // cCellPolBGC3M2DForm & Cell(const Pt2di & aP) {return mVCells.at(aP.y).at(aP.x);}
         cCellPolBGC3M2DForm & Cell(const Pt2di & aP) {return mVCells[aP.y][aP.x];}
         const cCellPolBGC3M2DForm & Cell(const Pt2di & aP) const {return mVCells[aP.y][aP.x];}

         bool CellHasValue(const Pt2di &) const;
         bool CellHasGradValue(const Pt2di &) const;

         void TestRot(const Pt2di & aP0,const Pt2di &aP1,double & aSomD,double & aSomR,ElMatrix<double> *);
         Pt2di SzCell() {return Pt2di(mNbCellX,mNbCellY);}
         Pt2dr  P2dNL(const Pt2dr & aPt) const;

         void AddEqRot(const Pt2di & aP0,const Pt2di &aP1,double aPds);
         double ModifInTervGrad(const double & aVal,const double & aBorne) const;

    private :
         Pt2dr DepSimul(const Pt2dr & aP,const ElMatrix<double> & aMat);
         Pt2dr DepOfKnownSol(const Pt2dr & aP,CamStenope *);
         cPolynBGC3M2D_Formelle(const cPolynBGC3M2D_Formelle &); // N.I.


   // ==> To unvirtualize cGenPDVFormelle 
         Pt2d<Fonc_Num>  EqFormProj();
         Pt2d<Fonc_Num>  EqFixedVal();
         Pt2d<Fonc_Num>  EqAttachRot();


         Pt2d<Fonc_Num>  FormalCorrec(Pt2d<Fonc_Num> aPF,cVarEtat_PhgrF aFAmpl,cP2d_Etat_PhgrF aFCenter);


         void AddEqAttach(Pt2dr aPIm,double aPds,bool Cur,CamStenope * aKnownSol);

         cBasicGeomCap3D *   mCamSsCorr;
         cPolynomial_BGC3M2D mCamInit;
         cPolynomial_BGC3M2D mCamCur;



         cEqfP3dIncTmp * mEqP3I;

         cVarEtat_PhgrF    mFAmplAppui;
         cVarEtat_PhgrF    mFAmplFixVal;
         cVarEtat_PhgrF    mFAmplAttRot;
         cP2d_Etat_PhgrF   mFCentrAppui;
         cP2d_Etat_PhgrF   mFCentrFixVal;
         cP2d_Etat_PhgrF   mFCentrAttRot;

         cP3d_Etat_PhgrF   mFP3DInit;
         cP2d_Etat_PhgrF   mFProjInit;

         cP2d_Etat_PhgrF   mFGradX;
         cP2d_Etat_PhgrF   mFGradY;
         cP2d_Etat_PhgrF   mFGradZ;
         cP2d_Etat_PhgrF   mObsPix;


         cP2d_Etat_PhgrF   mPtFixVal;
         cP2d_Etat_PhgrF   mFixedVal;

         cP2d_Etat_PhgrF    mRotPt;
         cP2d_Etat_PhgrF    mDepR1;
         cP2d_Etat_PhgrF    mDepR2;
         cP2d_Etat_PhgrF    mDepR3;


         cOneEq_PBGC3M2DF    mCompX;
         cOneEq_PBGC3M2DF    mCompY;
         std::string         mNameType;
         std::string         mNameAttach;
         std::string         mNameRot;
         cIncListInterv      mLIntervResiduApp;
         cIncListInterv      mLIntervAttach;
         cIncListInterv      mLIntervRot;
         cElCompiledFonc *   mFoncEqResidu;
         cElCompiledFonc *   mFoncEqAttach;
         cElCompiledFonc *   mFoncEqRot;
         int                 mNbCellX;
         int                 mNbCellY;
         Pt2di               mIndCenter;

         std::vector<std::vector<cCellPolBGC3M2DForm> > mVCells;
         ElMatrix<double>    mMatW2Loc;

         static double                           mEpsAng;
         static std::vector<ElMatrix<double> >   mEpsRot;
         static double                           mEpsGrad;
         cSubstitueBlocIncTmp * mBufSubRot;
};

double  cPolynBGC3M2D_Formelle::mEpsAng;
double  cPolynBGC3M2D_Formelle::mEpsGrad = 5.0;
std::vector<ElMatrix<double> > cPolynBGC3M2D_Formelle::mEpsRot;

// cPolynBGC3M2D_Formelle

/***************************************************************/
/*                                                             */
/*               cOneEq_PBGC3M2DF                              */
/*            cPolynBGC3M2D_Formelle                           */
/*                                                             */
/***************************************************************/

               // ==============================================
               //             cOneEq_PBGC3M2DF
               // ==============================================

cOneEq_PBGC3M2DF::cOneEq_PBGC3M2DF(cPolynBGC3M2D_Formelle & aPF,std::vector<double > & aCoef) :
   cElemEqFormelle (aPF.Set(),false),
   mPF     (&aPF),
   mCamCur (&(aPF.mCamCur))
{
    for (int aK=0 ; aK<int(aCoef.size()) ; aK++)
    {
        mVFCoef.push_back(aPF.Set().Alloc().NewF(&(aCoef[aK])));
    }

    CloseEEF();
    aPF.Set().AddObj2Kill(this);

}


Fonc_Num  cOneEq_PBGC3M2DF::EqFormProjCor(Pt2d<Fonc_Num> aP)
{
   //  ELISE_ASSERT(false,"cOneEq_PBGC3M2DF::FormProjCor 2 complete");
   Fonc_Num aRes = 0;
   for (int aK=0 ; aK<int(mVFCoef.size()) ; aK++)
   {
       aRes = aRes + mVFCoef[aK] * PowI(aP.x,mCamCur->DegX(aK)) *  PowI(aP.y,mCamCur->DegY(aK));
   }

   return aRes;
}


   //==========================================
   //      cCellPolBGC3M2DForm
   //==========================================


cCellPolBGC3M2DForm::cCellPolBGC3M2DForm(Pt2dr aPt,cPolynBGC3M2D_Formelle * aPF) : 
    mPt       (aPt),
    mActive   (aPF->CamSsCorr()->CaptHasDataGeom(mPt)),
    mHasDep   (false)
{
   if (mActive)
   {
       ElSeg3D aSeg = aPF->CamSsCorr()->Capteur2RayTer(mPt);
       mNorm = aSeg.TgNormee();
   } 
}
      
cCellPolBGC3M2DForm::cCellPolBGC3M2DForm() :
    mActive   (false),
    mHasDep   (false)
{
}


void cCellPolBGC3M2DForm::InitRep(cPolynBGC3M2D_Formelle * aPF)
{
     Pt3dr aNormLoc = aPF->mMatW2Loc * mNorm;

     for (int aK=0 ; aK<3 ; aK++)
     {
         Pt2dr aPNPert1 = ProjStenope( cPolynBGC3M2D_Formelle::mEpsRot[aK]* aNormLoc) ;
         Pt2dr aPNPert2 = ProjStenope( cPolynBGC3M2D_Formelle::mEpsRot[aK].transpose() * aNormLoc) ;
         mDerPnlRot[aK]  = (aPNPert1-aPNPert2) / cPolynBGC3M2D_Formelle::mEpsAng;
     }
}

void cCellPolBGC3M2DForm::SetGrad(const Pt2dr & aGx,const Pt2dr & aGy)
{
    mHasDep = true;
    ElMatrix<double> aM = MatFromCol(aGx,aGy);

    aM = gaussj(aM);
   
    for (int aK=0 ; aK<3 ; aK++)
    {
        //  mValDep[aK] = aM.transpose() * mDerPnlRot[aK];
        mValDep[aK] = aM * mDerPnlRot[aK];
    }

    // std::cout << mValDep[0] << mValDep[1] <<  mValDep[2] << "\n";
}

               // ============ cPolynBGC3M2D_Formelle =================





cPolynBGC3M2D_Formelle::cPolynBGC3M2D_Formelle
(
        cSetEqFormelles &   aSet,
        cPolynomial_BGC3M2D aCam0,
        bool                GenCodeAppui,
        bool                GenCodeAttach,
        bool                GenCodeRot
) :
   cGenPDVFormelle (aSet),
   mCamSsCorr      (aCam0.CamSsCor()),
   mCamInit        (aCam0),
   mCamCur         (aCam0),
   mEqP3I          ((GenCodeAppui | GenCodeRot) ?  mSet.Pt3dIncTmp() : 0),
   mFAmplAppui     ("AmplApp"),
   mFAmplFixVal    ("AmplFixV"),
   mFAmplAttRot    ("AmplAttR"),
   mFCentrAppui    ("CentrApp"),
   mFCentrFixVal   ("CentrFixV"),
   mFCentrAttRot   ("CentrAttR"),
   mFP3DInit       ("PTerInit"),
   mFProjInit      ("ProjInit"),
   mFGradX         ("GradX"),
   mFGradY         ("GradY"),
   mFGradZ         ("GradZ"),
   mObsPix         ("PIm"),
   mPtFixVal       ("PFixV"),
   mFixedVal       ("FixedV"),
   mRotPt          ("RotPt"),
   mDepR1          ("DepR1"),
   mDepR2          ("DepR2"),
   mDepR3          ("DepR3"),
   mCompX          (*this,mCamCur.Cx()),
   mCompY          (*this,mCamCur.Cy()),
   mNameType       ("cGen2DBundleEgProj_Deg"+ToString(mCamCur.DegreMax())),
   mNameAttach     ("cGen2DBundleAttach_Deg"+ToString(mCamCur.DegreMax())),
   mNameRot        ("cGen2DBundleAtRot_Deg"+ToString(mCamCur.DegreMax())),
   mFoncEqResidu   (0),
   mFoncEqAttach   (0),
   mIndCenter      (-1,-1),
   mMatW2Loc       (3,3)
{
    AllowUnsortedVarIn_SetMappingCur = true;
    mCompX.IncInterv().SetName("CX");
    mCompY.IncInterv().SetName("CY");
    if (mEqP3I)
    {
       mLIntervResiduApp.AddInterv(mEqP3I->IncInterv());
       mLIntervRot.AddInterv(mEqP3I->IncInterv());
    }
    mLIntervResiduApp.AddInterv(mCompX.IncInterv());
    mLIntervResiduApp.AddInterv(mCompY.IncInterv());
    mLIntervAttach.AddInterv(mCompX.IncInterv());
    mLIntervAttach.AddInterv(mCompY.IncInterv());

    mLIntervRot.AddInterv(mCompX.IncInterv());
    mLIntervRot.AddInterv(mCompY.IncInterv());

    if (GenCodeAppui)
    {
        GenerateCode(EqFormProj()-mObsPix.PtF(),mNameType,mLIntervResiduApp);
        return;
    } 
    if (GenCodeAttach)
    {
        GenerateCode(EqFixedVal(),mNameAttach,mLIntervAttach);
        return;
    } 
    if (GenCodeRot)
    {
        GenerateCode(EqAttachRot(),mNameRot,mLIntervRot);
        return;
    } 




    // ============================================================
    //   Calcul des cellules 
    // ============================================================


    int aNbPtsStd = 20;
    int aNbPtsMin = 1+ 2 * mCamInit.DegreMax();
    double aSzMax = 1000;

    Pt2dr aSzIm = Pt2dr(mCamCur.SzBasicCapt3D());
    double aSurfTot = aSzIm.x * aSzIm.y ;
    double aSurfCell = aSurfTot / ElSquare(aNbPtsStd);
    double aSzCell = ElMin(sqrt(aSurfCell),aSzMax);
    mNbCellX = ElMax(aNbPtsMin,round_up(aSzIm.x/aSzCell));
    mNbCellY = ElMax(aNbPtsMin,round_up(aSzIm.y/aSzCell));

    Pt2dr aCenter = aSzIm /2.0;
    double aDistMinCenter = 1e20;


    mVCells =  std::vector<std::vector<cCellPolBGC3M2DForm> > (mNbCellY+1);
    Pt2di aPInd(-1,-1);
    for (aPInd.y=0 ; aPInd.y<= mNbCellY ; aPInd.y++)
    {
        mVCells[aPInd.y] = std::vector<cCellPolBGC3M2DForm>  (mNbCellX+1);
        for (aPInd.x=0 ; aPInd.x<= mNbCellX ; aPInd.x++)
        {
            double aPdsX = aPInd.x/double(mNbCellX);
            double aPdsY = aPInd.y/double(mNbCellY);
            Pt2dr aP = aSzIm.mcbyc(Pt2dr(aPdsX,aPdsY));


            aP.x = ModifInTervGrad(aP.x,aSzIm.x); //  ElMin(ElMax(2*mEpsGrad,aP.x),aSzIm.x-2*mEpsGrad);
            aP.y = ModifInTervGrad(aP.y,aSzIm.y); //  ElMin(ElMax(2*mEpsGrad,aP.x),aSzIm.x-2*mEpsGrad);

            cCellPolBGC3M2DForm & aCurCell =  Cell(aPInd);
            aCurCell  = cCellPolBGC3M2DForm(aP,this);
            if (aCurCell.mActive)
            {
                double aDist = euclid(aCurCell.mPt,aCenter);
                if (aDist<aDistMinCenter)
                {
                    aDistMinCenter = aDist;
                    mIndCenter = aPInd;
                }
            }
        }
    }

    ELISE_ASSERT(mIndCenter.x>0,"Cannot determine center in cPolynBGC3M2D_Formelle");

    Pt3dr aZ = Cell(mIndCenter).mNorm;
    Pt3dr aX = Cell(mIndCenter+Pt2di(1,0)).mNorm;
    Pt3dr aY = vunit(aZ ^ aX);
    aX = vunit(aY^aZ);

    mMatW2Loc = MatFromCol(aX,aY,aZ).transpose();

    if (mEpsRot.empty())
    {
       mEpsAng = 1e-3;
       for (int aK=0 ; aK<3 ; aK++)
       {
           mEpsRot.push_back(ElMatrix<double>::Rotation3D(mEpsAng,aK));
       }
    }

    for (aPInd.y=0 ; aPInd.y<= mNbCellY ; aPInd.y++)
    {
        for (aPInd.x=0 ; aPInd.x<= mNbCellX ; aPInd.x++)
        {
            cCellPolBGC3M2DForm & aCurCell =  Cell(aPInd);
            if (aCurCell.mActive)
            {
               aCurCell.InitRep(this);
            }
        }
    }

    for (aPInd.y=0 ; aPInd.y<= mNbCellY ; aPInd.y++)
    {
        for (aPInd.x=0 ; aPInd.x<= mNbCellX ; aPInd.x++)
        {
            cCellPolBGC3M2DForm & aCurCell =  Cell(aPInd);
            if (aCurCell.mActive)
            {
                 std::vector<Pt2dr> aVGrad;

                 for (int aK=0 ; aK< 2 ; aK++)
                 {
                      Pt2dr aP1 = aCurCell.mPt + Pt2dr(TAB_4_NEIGH[aK]) * mEpsGrad;
                      Pt2dr aP2 = aCurCell.mPt - Pt2dr(TAB_4_NEIGH[aK]) * mEpsGrad;
                      if (mCamSsCorr->CaptHasDataGeom(aP1) && mCamSsCorr->CaptHasDataGeom(aP2))
                      {
                          Pt2dr aGrad = (P2dNL(aP1) - P2dNL(aP2) ) / (2*mEpsGrad);
                          aVGrad.push_back(aGrad);
                      }
                 }

                if (aVGrad.size()==2)
                {
                    aCurCell.SetGrad(aVGrad[0],aVGrad[1]);
                }
            }
        }
    }

    // ElMatrix<double> aM2Loc = MatFromCol(aX,aY,aZ);

    std::cout << "NB " << mNbCellX << " " << mNbCellY << " " << mIndCenter << "\n";

}


double cPolynBGC3M2D_Formelle::ModifInTervGrad(const double & aV,const double & aBorne) const
{
    return ElMin(ElMax(2*mEpsGrad,aV),aBorne-2*mEpsGrad);
}

Pt2dr  cPolynBGC3M2D_Formelle::P2dNL(const Pt2dr & aPt) const
{
    ElSeg3D aSeg = mCamSsCorr->Capteur2RayTer(aPt);
    Pt3dr aNorm =  mMatW2Loc*aSeg.TgNormee();
    return ProjStenope(aNorm);
}

Pt2dr cPolynBGC3M2D_Formelle::DepOfKnownSol(const Pt2dr & aP0,CamStenope * aCSOut)
{
    CamStenope * aCS = (CamStenope *) mCamSsCorr;
    Pt3dr aP1 =  aCS->ImEtProf2Terrain(aP0,1.0);
    Pt2dr aP2 = aCSOut->R3toF2(aP1);

    return aP2 - aP0;
}




Pt2dr cPolynBGC3M2D_Formelle::DepSimul(const Pt2dr & aP0,const ElMatrix<double> & aMat)
{
    CamStenope * aCS = (CamStenope *) mCamSsCorr;
    // ElSeg3D aSeg = mCamSsCorr->Capteur2RayTer(aP0);
    Pt3dr aP1 =  aCS->F2toDirRayonL3(aP0);
    aP1 = aMat * aP1;
    Pt2dr aP2 = aCS->L3toF2(aP1);

    return aP2 - aP0;
}

void cPolynBGC3M2D_Formelle::AddEqRot(const Pt2di & aP0,const Pt2di &aP1,double aPds)
{
    PostInit();
    std::set<int> aSX;
    std::set<int> aSY;
    Pt2di aPInd;
    int aNbOk=0;
    for (aPInd.x=aP0.x ; aPInd.x<=aP1.x ; aPInd.x++)
    {
        for (aPInd.y=aP0.y ; aPInd.y<=aP1.y ; aPInd.y++)
        {
            if (CellHasGradValue(aPInd))
            {
               aSX.insert(aPInd.x);
               aSY.insert(aPInd.y);
               aNbOk++;
            }
       }
    }

    if (aSX.size() < 2) return;
    if (aSY.size() < 2) return;

    for (aPInd.x=aP0.x ; aPInd.x<=aP1.x ; aPInd.x++)
    {
        for (aPInd.y=aP0.y ; aPInd.y<=aP1.y ; aPInd.y++)
        {
            if (CellHasGradValue(aPInd))
            {
                cCellPolBGC3M2DForm & aCurCell =  Cell(aPInd);
                mRotPt.SetEtat(aCurCell.mPt);
                mDepR1.SetEtat(aCurCell.mValDep[0]);
                mDepR2.SetEtat(aCurCell.mValDep[1]);
                mDepR3.SetEtat(aCurCell.mValDep[2]);

                mEqP3I->InitVal(Pt3dr(0,0,0));

                mSet.VAddEqFonctToSys(mFoncEqRot,aPds/aNbOk,false) ;
            }
       }
    }

    mBufSubRot->DoSubst();

}
/*
   mRotPt.SetEtat(aPIm);
   Pt2dr aValFix = Cur ? mCamCur.DeltaCamInit2CurIm(aPIm) : Pt2dr(0,0);
   mFixedVal.SetEtat(aValFix);

   mSet.VAddEqFonctToSys(mFoncEqAttach,aPds,false) ;
*/




void cPolynBGC3M2D_Formelle::TestRot(const Pt2di & aP0,const Pt2di &aP1,double & aSomD,double & aSomRot,ElMatrix<double> *aRotPert)
{
    aSomD=0;
    aSomRot=0;
    int aNbOk=0;
    L2SysSurResol aSys(3);

    Pt2di aPInd;
    for (aPInd.x=aP0.x ; aPInd.x<=aP1.x ; aPInd.x++)
    {
        for (aPInd.y=aP0.y ; aPInd.y<=aP1.y ; aPInd.y++)
        {
            if (CellHasGradValue(aPInd))
            {
                cCellPolBGC3M2DForm & aCell = Cell(aPInd);
                Pt2dr aPt = aCell.mPt;
                Pt2dr aDep = mCamCur.DeltaCamInit2CurIm(aPt);
                if (aRotPert) aDep = DepSimul(aPt,*aRotPert);

                aSomD += euclid(aDep);
                aNbOk++;
                //  SUM  P(K) *   mValDep[aK] =  aDep
                double aCoefX[3],aCoefY[3];

                for (int aK=0 ; aK<3 ; aK++)
                {
                     aCoefX[aK] = aCell.mValDep[aK].x;
                     aCoefY[aK] = aCell.mValDep[aK].y;
                }
                aSys.AddEquation(1.0,aCoefX,aDep.x);
                aSys.AddEquation(1.0,aCoefY,aDep.y);
            }
        }
    }

    bool Ok;
    Im1D_REAL8  aSol = aSys.Solve(&Ok);
    double * aDS = aSol.data();
    // std::cout << "SSSS " << aDS[0] << " "  << aDS[1] << " " << aDS[2] << "\n";

    for (aPInd.x=aP0.x ; aPInd.x<=aP1.x ; aPInd.x++)
    {
        for (aPInd.y=aP0.y ; aPInd.y<=aP1.y ; aPInd.y++)
        {
            if (CellHasGradValue(aPInd))
            {
                cCellPolBGC3M2DForm & aCell = Cell(aPInd);
                Pt2dr aPt = aCell.mPt;
                Pt2dr aDep = mCamCur.DeltaCamInit2CurIm(aPt);
                if (aRotPert)
                {
                   aDep = DepSimul(aPt,*aRotPert);
                }

                Pt2dr aDepR(0,0);
                for (int aK=0 ; aK<3 ; aK++)
                {
                    aDepR = aDepR + aCell.mValDep[aK] * aDS[aK];
                }

                aSomRot += euclid(aDep-aDepR);
                aNbOk++;
            }
        }
    }

    aSomD /= aNbOk;
    aSomRot /= aNbOk;
}



bool cPolynBGC3M2D_Formelle::CellHasValue(const Pt2di & aP) const
{
    return (aP.x>=0) && (aP.x<=mNbCellX) && (aP.y>=0) && (aP.y<=mNbCellY) && Cell(aP).mActive;
}

bool cPolynBGC3M2D_Formelle::CellHasGradValue(const Pt2di & aP) const
{
    return CellHasValue(aP) &&  Cell(aP).mHasDep;
}


cBasicGeomCap3D *   cPolynBGC3M2D_Formelle::CamSsCorr() const 
{
   return mCamSsCorr;
}

void cPolynBGC3M2D_Formelle::PostInit()
{
    if (mFoncEqResidu!=0) return ;

    mEqP3I  = mSet.Pt3dIncTmp();
    mLIntervResiduApp.AddInterv(mEqP3I->IncInterv());
    mLIntervRot.AddInterv(mEqP3I->IncInterv());

    mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
    mFoncEqAttach = cElCompiledFonc::AllocFromName(mNameAttach);
    mFoncEqRot = cElCompiledFonc::AllocFromName(mNameRot);


    if ((mFoncEqResidu==0) || (mFoncEqAttach==0) || (mFoncEqRot==0))
    {
       std::cout << "NAME = " << mNameType << " , " << mNameAttach << "\n";
       ELISE_ASSERT(false,"Can Get Code Comp for cCameraFormelle::cEqAppui");
    }


    mFoncEqResidu->SetMappingCur(mLIntervResiduApp,&mSet);
    mFoncEqAttach->SetMappingCur(mLIntervAttach,&mSet);
    mFoncEqRot->SetMappingCur(mLIntervRot,&mSet);


    mSet.AddFonct(mFoncEqResidu);
    mSet.AddFonct(mFoncEqAttach);
    mSet.AddFonct(mFoncEqRot);

    mFAmplAppui.InitAdr(*mFoncEqResidu);
    mFCentrAppui.InitAdr(*mFoncEqResidu);
    mFAmplFixVal.InitAdr(*mFoncEqAttach);
    mFCentrFixVal.InitAdr(*mFoncEqAttach);

    mFAmplAttRot.InitAdr(*mFoncEqRot);
    mFCentrAttRot.InitAdr(*mFoncEqRot);


    mFP3DInit.InitAdr(*mFoncEqResidu);
    mFProjInit.InitAdr(*mFoncEqResidu);
    mFGradX.InitAdr(*mFoncEqResidu);
    mFGradY.InitAdr(*mFoncEqResidu);
    mFGradZ.InitAdr(*mFoncEqResidu);
    mObsPix.InitAdr(*mFoncEqResidu);

    mPtFixVal.InitAdr(*mFoncEqAttach);
    mFixedVal.InitAdr(*mFoncEqAttach);

    mRotPt.InitAdr(*mFoncEqRot);
    mDepR1.InitAdr(*mFoncEqRot);
    mDepR2.InitAdr(*mFoncEqRot);
    mDepR3.InitAdr(*mFoncEqRot);

    mFAmplAppui.SetEtat(mCamCur.Ampl());
    mFCentrAppui.SetEtat(mCamCur.Center());
    mFAmplFixVal.SetEtat(mCamCur.Ampl());
    mFCentrFixVal.SetEtat(mCamCur.Center());
    mFAmplAttRot.SetEtat(mCamCur.Ampl());
    mFCentrAttRot.SetEtat(mCamCur.Center());

   
    mBufSubRot = new cSubstitueBlocIncTmp(*mEqP3I);
    mBufSubRot->AddInc(mLIntervRot);
    mBufSubRot->Close();

    mSet.AddObj2Kill(this);
}

cBasicGeomCap3D * cPolynBGC3M2D_Formelle::GPF_NC_CurBGCap3D() { return & mCamCur; }
const cBasicGeomCap3D *  cPolynBGC3M2D_Formelle::GPF_CurBGCap3D() const { return & mCamCur; }

Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::FormalCorrec(Pt2d<Fonc_Num> aPF,cVarEtat_PhgrF aFAmpl,cP2d_Etat_PhgrF aFCenter)
{
   Pt2d<Fonc_Num> aPPN =  (aPF-aFCenter.PtF()).div(aFAmpl.FN());
   return    Pt2d<Fonc_Num>
             (
                 mCompX.EqFormProjCor(aPPN),
                 mCompY.EqFormProjCor(aPPN)
             );
}

Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::EqFormProj()
{
   Pt3d<Fonc_Num>  aPTerUnknown  = mEqP3I->PF();
   Pt3d<Fonc_Num>  aDeltaPTU = aPTerUnknown-mFP3DInit.PtF();
 

   // Fonc_Num aDptUX = aDeltaPTU.x;
   Pt2d<Fonc_Num>  aProj  =   mFProjInit.PtF() 
                            + mFGradX.PtF().mul(aDeltaPTU.x) 
                            + mFGradY.PtF().mul(aDeltaPTU.y) 
                            + mFGradZ.PtF().mul(aDeltaPTU.z);

   

   return    aProj + FormalCorrec(aProj,mFAmplAppui,mFCentrAppui);
}


Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::EqFixedVal()
{
   return  FormalCorrec(mPtFixVal.PtF(),mFAmplFixVal,mFCentrFixVal) - mFixedVal.PtF();
}


Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::EqAttachRot()
{
    Pt3d<Fonc_Num>  aTeta  = mEqP3I->PF();
    Pt2d<Fonc_Num> aDep = FormalCorrec(mRotPt.PtF(),mFAmplAttRot,mFCentrAttRot);


   return aDep - mDepR1.PtF().mul(aTeta.x) -mDepR2.PtF().mul(aTeta.y) - mDepR3.PtF().mul(aTeta.z);
}

/*
*/

void  cPolynBGC3M2D_Formelle::GenerateCode(Pt2d<Fonc_Num> aFormP,const std::string & aName,cIncListInterv & anInterv)
{
/*
    Pt2d<Fonc_Num>  aFProj =  FormProj() - mObsPix.PtF();
    std::vector<Fonc_Num> aV;
    aV.push_back(aFProj.x);
    aV.push_back(aFProj.y);
*/

    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
        aName,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aFormP.ToTab(),  // expressions formelles 
        anInterv  // intervalle de reference
    );

}


void cPolynBGC3M2D_Formelle::AddEqAttach(Pt2dr aPIm,double aPds,bool Cur,CamStenope * aKnownSol)
{
   PostInit();
   mPtFixVal.SetEtat(aPIm);
   Pt2dr aValFix = Cur ? mCamCur.DeltaCamInit2CurIm(aPIm) : Pt2dr(0,0);
   if (aKnownSol)
   {
       aValFix = DepOfKnownSol(aPIm,aKnownSol);
   }
   mFixedVal.SetEtat(aValFix);

   mSet.VAddEqFonctToSys(mFoncEqAttach,aPds,false) ;
}

void cPolynBGC3M2D_Formelle::AddEqAttachGlob(double aPds,bool Cur,int aNbPts,CamStenope * aKnownSol)
{
    Pt2dr aSzIm = Pt2dr(mCamCur.SzBasicCapt3D());
    std::vector<Pt2dr> aVP;
    for (int aKx=0 ; aKx< aNbPts ; aKx++)
    {
        for (int aKy=0 ; aKy< aNbPts ; aKy++)
        {
             double aPdsX = (aKx+0.5)/aNbPts;
             double aPdsY = (aKy+0.5)/aNbPts;
             Pt2dr aP = aSzIm.mcbyc(Pt2dr(aPdsX,aPdsY));
             if (mCamSsCorr->CaptHasDataGeom(aP))
             {
                aVP.push_back(aP);
             }
        }
    }
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
         AddEqAttach(aVP[aKP],aPds/aVP.size(),Cur,aKnownSol);
    }
}



Pt2dr cPolynBGC3M2D_Formelle::AddEqAppuisInc(const Pt2dr & aPixObsIm,double aPds, cParamPtProj & aPPP,bool IsEqDroite)
{
   ELISE_ASSERT(!IsEqDroite,"cPolynBGC3M2D_Formelle::AddEqAppuisInc do not handle lines equation");
/*
*/
    Pt2dr aGx,aGy,aGz;

    Pt3dr aPTer = aPPP.mTer;
    Pt2dr aProjIm = mCamSsCorr->Ter2Capteur(aPTer);
    mFProjInit.SetEtat(aProjIm);
    mCamSsCorr->Diff(aGx,aGy,aGz,aProjIm,aPTer);

    mFP3DInit.SetEtat(aPTer);
    mFProjInit.SetEtat(aProjIm);
    mFGradX.SetEtat(aGx);
    mFGradY.SetEtat(aGy);
    mFGradZ.SetEtat(aGz);
    mObsPix.SetEtat(aPixObsIm);


    mEqP3I->InitVal(aPTer);

    std::vector<double> aVRes;

    if (aPds>0)
    {
       aVRes = mSet.VAddEqFonctToSys(mFoncEqResidu,aPds,false) ;
    }
    else
    {
       aVRes = mSet.VResiduSigne(mFoncEqResidu);
       
       if (1)
       {
           // Pt2dr aPRoj = 
           // Pt2dr 
           // std::cout << "VRES " << aVRes << "\n";
       }
    }


    ELISE_ASSERT(aVRes.size()==2,"cPolynBGC3M2D_Formelle::AddEqAppuisInc still un implemanted");
    return Pt2dr(aVRes[0],aVRes[1]);
}


cIncListInterv & cPolynBGC3M2D_Formelle::IntervAppuisPtsInc() 
{
     PostInit();
     return mLIntervResiduApp;
}

/***************************************************************/
/*                                                             */
/*                    cBGC3_Modif2D                            */
/*                                                             */
/***************************************************************/

cBGC3_Modif2D::cBGC3_Modif2D(cBasicGeomCap3D * aCam0) :
    mCam0 (aCam0),
    mSz (mCam0->SzBasicCapt3D())
{
}

ElSeg3D  cBGC3_Modif2D::Capteur2RayTer(const Pt2dr & aP) const
{
    return mCam0->Capteur2RayTer(CurIm2CamInit(aP));
}

Pt2dr  cBGC3_Modif2D::Ter2Capteur(const Pt3dr & aP) const
{
    return CamInit2CurIm(mCam0->Ter2Capteur(aP));
}

Pt2dr  cBGC3_Modif2D::Ter2CapteurSsCorrec(const Pt3dr & aP) const
{
    return mCam0->Ter2Capteur(aP);
}



Pt2di  cBGC3_Modif2D::SzBasicCapt3D() const
{
    return mSz;
}

double cBGC3_Modif2D::ResolSolOfPt(const Pt3dr & aP) const 
{
   return mCam0->ResolSolOfPt(aP);
}


bool  cBGC3_Modif2D::CaptHasData(const Pt2dr &aP) const
{
    return  mCam0->CaptHasData(CurIm2CamInit(aP));
}

bool      cBGC3_Modif2D::PIsVisibleInImage(const Pt3dr & aP) const
{
    return mCam0->PIsVisibleInImage(aP);
}

bool      cBGC3_Modif2D::HasOpticalCenterOfPixel() const
{
   return mCam0->HasOpticalCenterOfPixel();
}


Pt3dr    cBGC3_Modif2D::OpticalCenterOfPixel(const Pt2dr & aP) const 
{
   return mCam0->OpticalCenterOfPixel(CurIm2CamInit(aP));
}


          // A AFFINER PLUS TARD !!!!!  Version de base du point fixe a 1 iter ...
Pt2dr   cBGC3_Modif2D::DeltaCurIm2CamInit(const Pt2dr & aP) const
{
    Pt2dr aSol = -DeltaCamInit2CurIm(aP);


    Pt2dr aTest = CamInit2CurIm(aP+aSol);
    aSol = aSol + (aP-aTest);


    return aSol;
}

cBasicGeomCap3D *   cBGC3_Modif2D::CamSsCor()
{
    return mCam0;
}

// ==========================================================

/***************************************************************/
/*                                                             */
/*                    cPolynomial_BGC3M2D                      */
/*                                                             */
/***************************************************************/



std::vector<int> cPolynomial_BGC3M2D::mDegX;
std::vector<int> cPolynomial_BGC3M2D::mDegY;

std::vector<double> cPolynomial_BGC3M2D::mPowX;
std::vector<double> cPolynomial_BGC3M2D::mPowY;

std::vector<double> & cPolynomial_BGC3M2D::Cx() {return mCx;}
std::vector<double> & cPolynomial_BGC3M2D::Cy() {return mCy;}

void cPolynomial_BGC3M2D::SetPow(const Pt2dr & aPN) const
{
     if (aPN==mCurPPow) return;
     mCurPPow = aPN;
     for (int aD=1 ; aD<= mDegreMax ; aD++)
     {
           mPowX[aD] = mPowX[aD-1] * aPN.x;
           mPowY[aD] = mPowY[aD-1] * aPN.y;
     }
}


Pt2dr cPolynomial_BGC3M2D::DeltaCamInit2CurIm(const Pt2dr & aP0) const 
{
      Pt2dr aPN = ToPNorm(aP0); 
      SetPow(aPN);
      double aSx=0 ;
      double aSy=0 ;

      for (int aK=0 ; aK<int(mCx.size()) ; aK++)
      {
          double aPXY = mPowX[mDegX[aK]] * mPowY[mDegY[aK]] ;

          aSx += mCx[aK] * aPXY;
          aSy += mCy[aK] * aPXY;
      }


      return Pt2dr(aSx,aSy);
}



cPolynomial_BGC3M2D::cPolynomial_BGC3M2D(cBasicGeomCap3D * aCam0,int aDegreeMax,double aRandPerturb) :
    cBGC3_Modif2D (aCam0),
    mDegreMax     (aDegreeMax),
    mCenter       (Pt2dr(mSz)/2.0),
    mAmpl         (euclid(mCenter)),
    mCurPPow      (0.0,0.0)
{

     int aCpt=0;
     for (int  aDegreeTot=0 ; aDegreeTot<=aDegreeMax ; aDegreeTot++)
     {
          for (int aDegX=0 ; aDegX<= aDegreeTot ; aDegX++)
          {
              int aDegY=aDegreeTot - aDegX;

              double aVx=0,aVy=0;
              if (aRandPerturb)
              {
                 aVx = aRandPerturb * NRrandC();
                 aVy = aRandPerturb * NRrandC();
              }

              mCx.push_back(aVx);
              mCy.push_back(aVy);
              if (aCpt>=int(mDegX.size()))
              {
                  mDegX.push_back(aDegX);
                  mDegY.push_back(aDegY);
              }
              aCpt++;
          }
          if (int(mPowX.size()) <= aDegreeTot)
          {
              mPowX.push_back(1.0);
              mPowY.push_back(1.0);
          }
     }
     if (0)
     {
         Show();
     }
}

void cPolynomial_BGC3M2D::Show() const
{
    std::cout << "#### DMax= " << mDegreMax 
              << "  SizPow=" << mPowX.size()
              << "\n";
    Show("CoefX",mCx);
    Show("CoefY",mCy);
}

void cPolynomial_BGC3M2D::ShowMonome(const std::string & aVar , int aDeg) const
{
    if (aDeg==0) return;
    std::cout << "*" << aVar;
    if (aDeg==1) return;
    std::cout << "^" << aDeg ;
}
  

void cPolynomial_BGC3M2D::Show(const std::string & aMes,const std::vector<double> & aCoef) const
{
     std::cout << " -*-*-*- " << aMes << " -*-*-*-\n";
     for (int aK=0 ; aK<int(aCoef.size()) ; aK++)
     {
          std::cout << "    ";
          std::cout << ((aK==0) ? "  " : "+ ");
          std::cout << aCoef[aK] ;
          ShowMonome("X",mDegX[aK]);
          ShowMonome("Y",mDegY[aK]);
          // if (mDegX[aK]) std::cout << "X^" << mDegX[aK];
          // if (mDegY[aK]) std::cout << "Y^" << mDegY[aK];
          std::cout << "      ADR=" << &aCoef[aK] << "\n";
          std::cout << "\n";
     }
}


void TestBGC3M2D()
{
   std::string aName =  "/media/data2/Jeux-Test/Dino/Ori-Martini/Orientation-_MG_0140.CR2.xml";
   CamStenope * aCS = BasicCamOrientGenFromFile(aName);

   
   cPolynomial_BGC3M2D  aP1(aCS,1,0.0);
   cPolynomial_BGC3M2D  aP2(aCS,0,0.0);
   cPolynomial_BGC3M2D  aP2Bis(aCS,3,0.0);
   cPolynomial_BGC3M2D  aP3(aCS,2,0.0);
   cPolynomial_BGC3M2D  aP4(aCS,1,0.0);
}

void GenCodeEqProjGen(int aDeg,bool GenCode,bool GenCodeAttach,bool GenCodeRot)
{
    cSetEqFormelles  * aSet = new cSetEqFormelles(cNameSpaceEqF::eSysPlein);
    std::vector<double> aPAF;
    CamStenopeIdeale aCSI(false,1.0,Pt2dr(0,0),aPAF);
    aCSI.SetSz(Pt2di(100,100));

    cPolynomial_BGC3M2D aPolCSI(&aCSI,aDeg,0.0);

    new cPolynBGC3M2D_Formelle(*aSet,aPolCSI,GenCode,GenCodeAttach,GenCodeRot);
}

/***********************************************************************/
/*                                                                     */
/*                                                                     */
/*                                                                     */
/***********************************************************************/


class cCamTest_PBGC3M2DF;
class cTest_PBGC3M2DF;

class cCamTest_PBGC3M2DF
{
    public :
         cCamTest_PBGC3M2DF(cImaMM &,cTest_PBGC3M2DF&,int aK);
       
    // private :

       cTest_PBGC3M2DF *       mAppli; 
       cImaMM *                mIma;
       CamStenope *            mCS0;
       ElMatrix<double>        mMatPert;
       CamStenope *            mCSCur;
       cPolynomial_BGC3M2D     mPolCam;
       cPolynBGC3M2D_Formelle  mFPC;
       double                  mNbMesPts; 
       double                  mSomPdsMes; 
       int                     mK;

       void Show();
};

void cCamTest_PBGC3M2DF::Show()
{
   mFPC.TypedCamCur()->Show();
}

class cSetCTest_PBGC3M2DF
{
    public :
         std::vector<cCamTest_PBGC3M2DF *>   mVCams;
         ElPackHomologue        mPack12;     
         ElPackHomologue        mPack21;     
         std::vector<Pt2dr>     mVP[3];
         cSubstitueBlocIncTmp * mBufSub;

         const cBasicGeomCap3D * KCamCur(int aKC) const {return mVCams[aKC]->mFPC.GPF_CurBGCap3D();}

         const cBasicGeomCap3D * Cam0(int aKC) const {return mVCams[aKC]->mCS0;}
         
};


class cTest_PBGC3M2DF : public cAppliWithSetImage
{
    public :
       cTest_PBGC3M2DF(int argc,char ** argv);
    // private :
       cSetEqFormelles *                       mSet;
       cEqfP3dIncTmp *                         mEqP3I;
       std::string                             mPat;
       std::string                             mOri;
       std::vector<cCamTest_PBGC3M2DF *>       mVCT;
       std::vector<cSetCTest_PBGC3M2DF *>      mVCpleT;
       std::vector<cSetCTest_PBGC3M2DF *>      mVTriT;
       std::map<Pt2di,cSetCTest_PBGC3M2DF *>  mMapCpleT;
       int                                     mDeg;
       int                                     mNbSom;
       double                                  mPerturbAng;
       double                                  mPerturbPol;
       bool                                    mPerfectData;
       CamStenope *                            CamPerturb(CamStenope *, ElMatrix<double> &);

       bool HasArc(int aK1, int aK2)
       {
            if (aK1>aK2) ElSwap(aK1,aK2);
            return DicBoolFind(mMapCpleT,Pt2di(aK1,aK2));
       }
       double RandAngle() {return mPerturbAng * NRrandC();}

       void OneIterBundle();
       double AddBundle(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS,double anErr);
       void SetPerfectData(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS);
};

cCamTest_PBGC3M2DF::cCamTest_PBGC3M2DF(cImaMM & anIma,cTest_PBGC3M2DF& anAppli,int aK) :
   mAppli   (& anAppli),
   mIma     (& anIma),
   mCS0     (mIma->mCam),
   mMatPert (3,3),
   mCSCur   (mAppli->CamPerturb(mCS0,mMatPert)),
   mPolCam  (mCSCur,anAppli.mDeg,anAppli.mPerturbPol),
   mFPC     (*(mAppli->mSet),mPolCam,false,false,false),
   mNbMesPts (0.0),
   mSomPdsMes (0.0),
   mK       (aK)
{
     std::cout << " cCamTest_PBGC3M2DF: " << mIma->mNameIm << "\n";
}


CamStenope * cTest_PBGC3M2DF::CamPerturb(CamStenope * aCS0,ElMatrix<double> & aMPert)
{
   CamStenope * aCS = aCS0->Dupl();
   ElRotation3D  aR =aCS->Orient().inv();

   // int i = 3 +  5* 4;

   aMPert  = ElMatrix<double>::Rotation(RandAngle(),RandAngle(),RandAngle());
   aR = ElRotation3D(aR.tr(),aR.Mat()*aMPert,true);
   aCS->SetOrientation(aR.inv());

   return aCS;
}



void cTest_PBGC3M2DF::SetPerfectData(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS)
{
    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
       cSetCTest_PBGC3M2DF * aSet = aVS[aKS];
       int aNbCam = aSet->mVCams.size();
       int aNbP = aSet->mVP[0].size();
       std::vector<Pt2dr>  aNewVP[3];
       for (int aKP=0 ; aKP<aNbP ; aKP++)
       {
          std::vector<Pt3dr> aVP0;
          std::vector<Pt3dr> aVP1;
          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              // ElSeg3D aSeg = aSet->mVCams[aKC]->mFPC.GPF_CurBGCap3D()->Capteur2RayTer(aSet->mVP[aKC][aKP]);
              ElSeg3D aSeg = aSet->Cam0(aKC)->Capteur2RayTer(aSet->mVP[aKC][aKP]);
              aVP0.push_back(aSeg.P0());
              aVP1.push_back(aSeg.P1());
          }
          bool Ok;
          Pt3dr  aPImTer = InterSeg(aVP0,aVP1,Ok);
          bool AllOk = true;

          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              if (!aSet->Cam0(aKC)->PIsVisibleInImage(aPImTer))
              {
                  AllOk = false;
              }
          }

          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              Pt2dr aProj  = aSet->Cam0(aKC)->Ter2Capteur(aPImTer);
              if (!aSet->Cam0(aKC)->CaptHasData(aProj))
              {
                  AllOk = false;
              }
          }

          if (AllOk)
          {
              for (int aKC=0 ; aKC<aNbCam ; aKC++)
              {
                  aNewVP[aKC].push_back(aSet->Cam0(aKC)->Ter2Capteur(aPImTer));
              }
          }
       }
       for (int aK=0 ; aK<3 ; aK++)
       {
          aSet->mVP[aK] = aNewVP[aK];
       }
    }
}


double cTest_PBGC3M2DF::AddBundle(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS,double aErrStd)
{
    std::vector<double> aVEr;
    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
       cSetCTest_PBGC3M2DF * aSet = aVS[aKS];
       int aNbCam = aSet->mVCams.size();
       int aNbP = aSet->mVP[0].size();
       for (int aKP=0 ; aKP<aNbP ; aKP++)
       {
          std::vector<Pt3dr> aVP0;
          std::vector<Pt3dr> aVP1;
          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              // ElSeg3D aSeg = aSet->mVCams[aKC]->mFPC.GPF_CurBGCap3D()->Capteur2RayTer(aSet->mVP[aKC][aKP]);

              ElSeg3D aSeg = aSet->KCamCur(aKC)->Capteur2RayTer(aSet->mVP[aKC][aKP]);
              aVP0.push_back(aSeg.P0());
              aVP1.push_back(aSeg.P1());
          }
          bool Ok;
          Pt3dr  aPImTer = InterSeg(aVP0,aVP1,Ok);
          double anEr = 0;
          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              Pt2dr aPIm = aSet->mVP[aKC][aKP];
              cParamPtProj aPPP(1.0,1.0,false);
              aPPP.mTer = aPImTer;

              Pt2dr aPAp = aSet->mVCams[aKC]->mFPC.AddEqAppuisInc(aPIm,0,aPPP,false);
              anEr += euclid(aPAp);
          }
          anEr /= aNbCam;
          aVEr.push_back(anEr);
          
          if ((aErrStd >0) && (anEr< (10 * aErrStd)))
          {
              double aPds = 1 /(1 + ElSquare(anEr/aErrStd));
              for (int aKC=0 ; aKC<aNbCam ; aKC++)
              {
                  Pt2dr aPIm = aSet->mVP[aKC][aKP];
                  cParamPtProj aPPP(1.0,1.0,false);
                  aPPP.mTer = aPImTer;
                  aSet->mVCams[aKC]->mFPC.AddEqAppuisInc(aPIm,aPds,aPPP,false);
              }
              aSet->mBufSub->DoSubst();
          }
       }
    }
    // Max en cas de donnees parfaite ...
    return ElMax(0.0001,KthValProp(aVEr,0.75));
}

void cTest_PBGC3M2DF::OneIterBundle()
{
   // mVCT[0]->Show();
   mSet->SetPhaseEquation();

   for (int aKC=0 ; aKC<int(mVCT.size()) ; aKC++)
   {
       cCamTest_PBGC3M2DF * aCT =  mVCT[aKC];
       cPolynBGC3M2D_Formelle & aCF = aCT->mFPC;

       double aSomD,aSomRot;
       aCF.TestRot(Pt2di(0,0),aCF.SzCell(),aSomD,aSomRot,0);
       // Avec forcage, les resultat sont "bons"
       // aCF.TestRot(Pt2di(0,0),aCF.SzCell(),aSomD,aSomRot,&(mVCT[aKC]->mMatPert));
       ElTimer aT;
       aCF.AddEqAttachGlob(aCT->mSomPdsMes *1e-3,true,20,0);
       // aCF.AddEqAttachGlob(aCT->mSomPdsMes * 1e-5,false,20,0);

/*
*/
       //  aCF.AddEqAttachGlob(aCT->mSomPdsMes*10,false,20,aCT->mCS0);
       aCF.AddEqRot(Pt2di(0,0),aCF.SzCell(),aCT->mSomPdsMes* 1e-1);

       std::cout << "SOMD " << mVCT[aKC]->mIma->mNameIm << " " <<  aSomD << " " << aSomRot  << " T " << aT.uval() << " Pds " << aCT->mSomPdsMes << "\n";
   }

   double aErCple = AddBundle(mVCpleT,-1);
   std::cout << "ERCPLE " << aErCple << "\n";
   AddBundle(mVCpleT,aErCple);


   if (mVTriT.size())
   {
       double aErrTri = AddBundle(mVTriT,-1);
       AddBundle(mVTriT,aErrTri);
       std::cout << "ER TRI " << aErrTri  << "\n";
   }

// DebugBundleGen = true;
   std::cout << "\n";
   mSet->SolveResetUpdate();
}



extern bool ShowStatMatCond;

cTest_PBGC3M2DF::cTest_PBGC3M2DF(int argc,char ** argv)  :
    cAppliWithSetImage   (argc-1,argv+1,0),
    mSet                 (new cSetEqFormelles(cNameSpaceEqF::eSysPlein)),
    mDeg                 (2),
    mPerturbAng          (0.01),
    mPerturbPol          (0.0),
    mPerfectData         (false)
{
   ShowStatMatCond = false;
   //  cSubstitueBlocIncTmp::AddInc recouvrement / TMP
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mPat,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistDirOri),
        LArgMain()
                    << EAM(mDeg,"Degre", true,"Degre of polynomial correction (Def=3)")
                    << EAM(mPerturbAng,"PertAng", true,"Angle Perturbation")
                    << EAM(mPerturbPol,"PertPol", true,"Polynomial Perturbation")
                    << EAM(mPerfectData,"PerfectData", true,"Set data with potentially perfect projection")
   );
   mNbSom = mVSoms.size();

   for (int aK=0 ; aK<mNbSom ; aK++)
   {
       mVCT.push_back(new cCamTest_PBGC3M2DF(*mVSoms[aK]->attr().mIma,*this,aK));
   }

   std::cout << "DONE INIT \n"; 

   mEqP3I  =  mSet->Pt3dIncTmp();
   std::string aKey = "NKS-Assoc-CplIm2Hom@@dat";

   

   for (int aK1=0 ;  aK1 <mNbSom ; aK1++)
   {
       for (int aK2=aK1+1 ;  aK2 <mNbSom ; aK2++)
       {
            cCamTest_PBGC3M2DF * aC1 = mVCT[aK1];         
            cCamTest_PBGC3M2DF * aC2 = mVCT[aK2];         
            std::string aN12 =  mEASF.mICNM->Assoc1To2(aKey,aC1->mIma->mNameIm,aC2->mIma->mNameIm,true);
            std::string aN21 =  mEASF.mICNM->Assoc1To2(aKey,aC2->mIma->mNameIm,aC1->mIma->mNameIm,true);
            if (ELISE_fp::exist_file(aN12) && ELISE_fp::exist_file(aN21))
            {
                 cSetCTest_PBGC3M2DF * aCple = new cSetCTest_PBGC3M2DF;
                 aCple->mPack12 =  ElPackHomologue::FromFile(aN12);
                 aCple->mPack21 =  ElPackHomologue::FromFile(aN21);
                 aCple->mVCams.push_back(aC1);
                 aCple->mVCams.push_back(aC2);


                 Merge2Pack(aCple->mVP[0],aCple->mVP[1],1,aCple->mPack12,aCple->mPack21);

                 std::cout << aN12 << " " << aN21 
                          << " Sz0= " << aCple->mPack12.size() << " " << aCple->mPack21.size() 
                          << " SzM= " << aCple->mVP[0].size()  << "\n";
                 int aNbPts = aCple->mVP[0].size();
                 if (aNbPts > 10)
                 {
                     cSubstitueBlocIncTmp * aBS = new cSubstitueBlocIncTmp(*mEqP3I);
                     aCple->mBufSub =  aBS;
                     aBS->AddInc(aC1->mFPC.IntervAppuisPtsInc());
                     aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                     aBS->Close();

// aBuf->AddInc(*(mVLInterv[aK]));

                     mVCpleT.push_back(aCple);
                     mMapCpleT[Pt2di(aK1,aK2)] = aCple;

                     aC1->mNbMesPts+= aNbPts;
                     aC2->mNbMesPts+= aNbPts;
                 }
            }
       }
   }


   for (int aK1=0 ;  aK1 <mNbSom ; aK1++)
   {
       for (int aK2=aK1+1 ;  aK2 <mNbSom ; aK2++)
       {
           for (int aK3=aK2+1 ;  aK3 <mNbSom ; aK3++)
           {
               if (HasArc(aK1,aK2) && HasArc(aK1,aK3) && HasArc(aK2,aK3))
               {
                   cSetCTest_PBGC3M2DF * aTri = new cSetCTest_PBGC3M2DF;
                   cCamTest_PBGC3M2DF * aC1 = mVCT[aK1];         
                   cCamTest_PBGC3M2DF * aC2 = mVCT[aK2];         
                   cCamTest_PBGC3M2DF * aC3 = mVCT[aK3];         
                   aTri->mVCams.push_back(aC1);
                   aTri->mVCams.push_back(aC2);
                   aTri->mVCams.push_back(aC3);

                   /*aTri->mCam1 = mVCT[aK1];         
                   aTri->mCam2 = mVCT[aK2];         
                   aTri->mCam3 = mVCT[aK3];          */
                   cSetCTest_PBGC3M2DF * aCp12 = mMapCpleT[Pt2di(aK1,aK2)];
                   cSetCTest_PBGC3M2DF * aCp13 = mMapCpleT[Pt2di(aK1,aK3)];
                   cSetCTest_PBGC3M2DF * aCp23 = mMapCpleT[Pt2di(aK2,aK3)];



                   Merge3Pack
                   (
                        aTri->mVP[0],aTri->mVP[1],aTri->mVP[2],
                        3,
                        aCp12->mVP[0], aCp12->mVP[1],
                        aCp13->mVP[0], aCp13->mVP[1],
                        aCp23->mVP[0], aCp23->mVP[1]
                   );
                   int aNbPts = aTri->mVP[0].size();
                   if (aNbPts > 5)
                   {
                       cSubstitueBlocIncTmp * aBS = new cSubstitueBlocIncTmp(*mEqP3I);
                       aTri->mBufSub =  aBS;
                       aBS->AddInc(aC1->mFPC.IntervAppuisPtsInc());
                       aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                       aBS->AddInc(aC3->mFPC.IntervAppuisPtsInc());

                       // aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                       // aBS->AddInc(aC3->mFPC.IntervAppuisPtsInc());
                       aBS->Close();
  
                       aC1->mNbMesPts += aNbPts;
                       aC2->mNbMesPts += aNbPts;
                       aC3->mNbMesPts += aNbPts;


                       mVTriT.push_back(aTri);
                       std::cout  << aC1->mIma->mNameIm  << " "
                                  << aC2->mIma->mNameIm  << " "
                                  << aC3->mIma->mNameIm  << " "
                                  <<  aTri->mVP[0].size()  << " "
                                  <<  aTri->mVP[1].size()  << " "
                                  <<  aTri->mVP[2].size()  << "\n";
                   }
               }
           }
       }
   }

   for (int aK=0 ; aK<mNbSom ; aK++)
   {
       mVCT[aK]->mSomPdsMes = mVCT[aK]->mNbMesPts;
   }


   if (mPerfectData)
   {
      SetPerfectData(mVCpleT);
      SetPerfectData(mVTriT);
   }
/*
*/

   mSet->SetClosed();


   for (int aK=0 ; aK<100 ; aK++)
   {
      OneIterBundle();
   }
}

int CPP_TestBundleGen(int argc,char ** argv)  
{
    cTest_PBGC3M2DF anAppli(argc,argv);

    return EXIT_SUCCESS;
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
