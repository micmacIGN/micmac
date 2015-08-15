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

bool DebugBundleGen=false;

class cBGC3_Modif2D ; //   : public cBasicGeomCap3D
class cPolynomial_BGC3M2D ;//  : public cBGC3_Modif2D
class cPolynBGC3M2D_Formelle; // : public cGenPDVFormelle
class cOneEq_PBGC3M2DF;


/*

   N ((i,j)+D(i,j)) = R N(i,j) = (I + W e ^ N(i,j))
  
   (Id + J * D(i,j) )  = I + W e ^ N(i,j))

    D(i,j) = J-1 (W ^

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

       Fonc_Num  FormProjCor(Pt2d<Fonc_Num> aP);
       
   private :
       std::vector<Fonc_Num>     mVFCoef;
       cPolynBGC3M2D_Formelle *  mPF;
       cPolynomial_BGC3M2D*      mCamCur;
};



class cPolynBGC3M2D_Formelle : public cGenPDVFormelle
{

    public  :
         friend class cOneEq_PBGC3M2DF;

         cPolynBGC3M2D_Formelle(cSetEqFormelles & aSet,cPolynomial_BGC3M2D aCam0,bool GenCode,bool GenCodeAttach);
         void GenerateCode(Pt2d<Fonc_Num>,const std::string &,cIncListInterv &);
         cIncListInterv & IntervAppuisPtsInc() ;
         void PostInit();
         const cBasicGeomCap3D * GPF_CurBGCap3D() const ;
         cBasicGeomCap3D * GPF_NC_CurBGCap3D() ;
         Pt2dr AddEqAppuisInc(const Pt2dr & aPIm,double aPds, cParamPtProj &,bool IsEqDroite);
         
         const cPolynomial_BGC3M2D *  TypedCamCur() const { return & mCamCur; }
         cPolynomial_BGC3M2D *  TypedCamCur() { return & mCamCur; }
         void AddEqAttachGlob(double aPds,bool Cur,int aNbPts);
    private :
         cPolynBGC3M2D_Formelle(const cPolynBGC3M2D_Formelle &); // N.I.


   // ==> To unvirtualize cGenPDVFormelle 
         Pt2d<Fonc_Num>  FormProj();
         Pt2d<Fonc_Num>  FixedVal();
         Pt2d<Fonc_Num>  FormalCorrec(Pt2d<Fonc_Num> aPF,cVarEtat_PhgrF aFAmpl,cP2d_Etat_PhgrF aFCenter);


         void AddEqAttach(Pt2dr aPIm,double aPds,bool Cur);

         cBasicGeomCap3D *   mCamSsCorr;
         cPolynomial_BGC3M2D mCamInit;
         cPolynomial_BGC3M2D mCamCur;



         cEqfP3dIncTmp * mEqP3I;

         cVarEtat_PhgrF    mFAmplAppui;
         cVarEtat_PhgrF    mFAmplAttach;
         cP2d_Etat_PhgrF   mFCentrAppui;
         cP2d_Etat_PhgrF   mFCentrAttach;

         cP3d_Etat_PhgrF   mFP3DInit;
         cP2d_Etat_PhgrF   mFProjInit;

         cP2d_Etat_PhgrF   mFGradX;
         cP2d_Etat_PhgrF   mFGradY;
         cP2d_Etat_PhgrF   mFGradZ;
         cP2d_Etat_PhgrF   mObsPix;


         cP2d_Etat_PhgrF   mPtFixVal;
         cP2d_Etat_PhgrF   mFixedVal;


         cOneEq_PBGC3M2DF    mCompX;
         cOneEq_PBGC3M2DF    mCompY;
         std::string         mNameType;
         std::string         mNameAttach;
         cIncListInterv      mLIntervResiduApp;
         cIncListInterv      mLIntervAttach;
         cElCompiledFonc *   mFoncEqResidu;
         cElCompiledFonc *   mFoncEqAttach;
};


/***************************************************************/
/*                                                             */
/*               cOneEq_PBGC3M2DF                              */
/*            cPolynBGC3M2D_Formelle                           */
/*                                                             */
/***************************************************************/

               // ============ cOneEq_PBGC3M2DF =================

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


Fonc_Num  cOneEq_PBGC3M2DF::FormProjCor(Pt2d<Fonc_Num> aP)
{
   //  ELISE_ASSERT(false,"cOneEq_PBGC3M2DF::FormProjCor 2 complete");
   Fonc_Num aRes = 0;
   for (int aK=0 ; aK<int(mVFCoef.size()) ; aK++)
   {
       aRes = aRes + mVFCoef[aK] * PowI(aP.x,mCamCur->DegX(aK)) *  PowI(aP.y,mCamCur->DegY(aK));
   }

/*
{
    std::cout << "cOneEq_PBGC3M2DF::FormProjCor \n";

    aRes.show(std::cout);
    getchar();
}
*/


   return aRes;
}


               // ============ cPolynBGC3M2D_Formelle =================



cPolynBGC3M2D_Formelle::cPolynBGC3M2D_Formelle
(
        cSetEqFormelles &   aSet,
        cPolynomial_BGC3M2D aCam0,
        bool                GenCode,
        bool                GenCodeAttach
) :
   cGenPDVFormelle (aSet),
   mCamSsCorr      (aCam0.CamSsCor()),
   mCamInit        (aCam0),
   mCamCur         (aCam0),
   mEqP3I          (GenCode ?  mSet.Pt3dIncTmp() : 0),
   mFAmplAppui     ("Ampl"),
   mFAmplAttach    ("Ampl"),
   mFCentrAppui    ("Centr"),
   mFCentrAttach   ("Centr"),
   mFP3DInit       ("PTerInit"),
   mFProjInit      ("ProjInit"),
   mFGradX         ("GradX"),
   mFGradY         ("GradY"),
   mFGradZ         ("GradZ"),
   mObsPix         ("PIm"),
   mPtFixVal       ("PFixV"),
   mFixedVal       ("FixedV"),
   mCompX          (*this,mCamCur.Cx()),
   mCompY          (*this,mCamCur.Cy()),
   mNameType       ("cGen2DBundleEgProj_Deg"+ToString(mCamCur.DegreMax())),
   mNameAttach     ("cGen2DBundleAttach_Deg"+ToString(mCamCur.DegreMax())),
   mFoncEqResidu   (0),
   mFoncEqAttach   (0)
{
    AllowUnsortedVarIn_SetMappingCur = true;
    mCompX.IncInterv().SetName("CX");
    mCompY.IncInterv().SetName("CY");
    if (mEqP3I)
    {
       mLIntervResiduApp.AddInterv(mEqP3I->IncInterv());
    }
    mLIntervResiduApp.AddInterv(mCompX.IncInterv());
    mLIntervResiduApp.AddInterv(mCompY.IncInterv());
    mLIntervAttach.AddInterv(mCompX.IncInterv());
    mLIntervAttach.AddInterv(mCompY.IncInterv());

    if (GenCode)
    {
        GenerateCode(FormProj()-mObsPix.PtF(),mNameType,mLIntervResiduApp);
        return;
    } 
    if (GenCodeAttach)
    {
        GenerateCode(FixedVal(),mNameAttach,mLIntervAttach);
        return;
    } 
}

void cPolynBGC3M2D_Formelle::PostInit()
{
    if (mFoncEqResidu!=0) return ;

    mEqP3I  = mSet.Pt3dIncTmp();
    mLIntervResiduApp.AddInterv(mEqP3I->IncInterv());

    mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
    mFoncEqAttach = cElCompiledFonc::AllocFromName(mNameAttach);
    if ((mFoncEqResidu==0) || (mFoncEqAttach==0))
    {
       std::cout << "NAME = " << mNameType << " , " << mNameAttach << "\n";
       ELISE_ASSERT(false,"Can Get Code Comp for cCameraFormelle::cEqAppui");
    }


    mFoncEqResidu->SetMappingCur(mLIntervResiduApp,&mSet);
    mFoncEqAttach->SetMappingCur(mLIntervAttach,&mSet);
    mSet.AddFonct(mFoncEqResidu);
    mSet.AddFonct(mFoncEqAttach);

    mFAmplAppui.InitAdr(*mFoncEqResidu);
    mFCentrAppui.InitAdr(*mFoncEqResidu);
    mFAmplAttach.InitAdr(*mFoncEqAttach);
    mFCentrAttach.InitAdr(*mFoncEqAttach);


    mFP3DInit.InitAdr(*mFoncEqResidu);
    mFProjInit.InitAdr(*mFoncEqResidu);
    mFGradX.InitAdr(*mFoncEqResidu);
    mFGradY.InitAdr(*mFoncEqResidu);
    mFGradZ.InitAdr(*mFoncEqResidu);
    mObsPix.InitAdr(*mFoncEqResidu);

    mPtFixVal.InitAdr(*mFoncEqAttach);
    mFixedVal.InitAdr(*mFoncEqAttach);

    mFAmplAppui.SetEtat(mCamCur.Ampl());
    mFCentrAppui.SetEtat(mCamCur.Center());
    mFAmplAttach.SetEtat(mCamCur.Ampl());
    mFCentrAttach.SetEtat(mCamCur.Center());


    mSet.AddObj2Kill(this);
}

cBasicGeomCap3D * cPolynBGC3M2D_Formelle::GPF_NC_CurBGCap3D() { return & mCamCur; }
const cBasicGeomCap3D *  cPolynBGC3M2D_Formelle::GPF_CurBGCap3D() const { return & mCamCur; }

Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::FormalCorrec(Pt2d<Fonc_Num> aPF,cVarEtat_PhgrF aFAmpl,cP2d_Etat_PhgrF aFCenter)
{
   Pt2d<Fonc_Num> aPPN =  (aPF-aFCenter.PtF()).div(aFAmpl.FN());
   return    Pt2d<Fonc_Num>
             (
                 mCompX.FormProjCor(aPPN),
                 mCompY.FormProjCor(aPPN)
             );
}

Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::FormProj()
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


Pt2d<Fonc_Num>  cPolynBGC3M2D_Formelle::FixedVal()
{
   return  FormalCorrec(mPtFixVal.PtF(),mFAmplAttach,mFCentrAttach) - mFixedVal.PtF();
}



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


void cPolynBGC3M2D_Formelle::AddEqAttach(Pt2dr aPIm,double aPds,bool Cur)
{
   PostInit();
   mPtFixVal.SetEtat(aPIm);
   Pt2dr aValFix = Cur ? mCamCur.DeltaCamInit2CurIm(aPIm) : Pt2dr(0,0);
   mFixedVal.SetEtat(aValFix);

   mSet.VAddEqFonctToSys(mFoncEqAttach,aPds,false) ;
}

void cPolynBGC3M2D_Formelle::AddEqAttachGlob(double aPds,bool Cur,int aNbPts)
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
             if (mCamCur.CaptHasData(aP))
             {
                aVP.push_back(aP);
             }
        }
    }
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
         AddEqAttach(aVP[aKP],aPds/aVP.size(),Cur);
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

void GenCodeEqProjGen(int aDeg,bool GenCode,bool GenCodeAttach)
{
    cSetEqFormelles  * aSet = new cSetEqFormelles(cNameSpaceEqF::eSysPlein);
    std::vector<double> aPAF;
    CamStenopeIdeale aCSI(false,1.0,Pt2dr(0,0),aPAF);
    aCSI.SetSz(Pt2di(100,100));

    cPolynomial_BGC3M2D aPolCSI(&aCSI,aDeg,0.0);

    new cPolynBGC3M2D_Formelle(*aSet,aPolCSI,GenCode,GenCodeAttach);
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
       CamStenope *            mCSCur;
       cPolynomial_BGC3M2D     mPolCam;
       cPolynBGC3M2D_Formelle  mFPC;
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
         
};

/*
class cTriTest_PBGC3M2DF
{
    public :
         std::vector<cCamTest_PBGC3M2DF *>   mVCams;
         // cCamTest_PBGC3M2DF *   mCam1;
         // cCamTest_PBGC3M2DF *   mCam2;
         // cCamTest_PBGC3M2DF *   mCam3;
         std::vector<Pt2dr>     mVP1;
         std::vector<Pt2dr>     mVP2;
         std::vector<Pt2dr>     mVP3;
         cSubstitueBlocIncTmp * mBufSub;
};
*/

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
       CamStenope *                            CamPerturb(CamStenope *);

       bool HasArc(int aK1, int aK2)
       {
            if (aK1>aK2) ElSwap(aK1,aK2);
            return DicBoolFind(mMapCpleT,Pt2di(aK1,aK2));
       }
       double RandAngle() {return mPerturbAng * NRrandC();}

       void OneIterBundle();
       double AddBundle(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS,double anErr);
};

cCamTest_PBGC3M2DF::cCamTest_PBGC3M2DF(cImaMM & anIma,cTest_PBGC3M2DF& anAppli,int aK) :
   mAppli   (& anAppli),
   mIma     (& anIma),
   mCS0     (mIma->mCam),
   mCSCur   (mAppli->CamPerturb(mCS0)),
   mPolCam  (mCSCur,anAppli.mDeg,anAppli.mPerturbPol),
   mFPC     (*(mAppli->mSet),mPolCam,false,false),
   mK       (aK)
{
     std::cout << " cCamTest_PBGC3M2DF: " << mIma->mNameIm << "\n";
}


CamStenope * cTest_PBGC3M2DF::CamPerturb(CamStenope * aCS0)
{
   CamStenope * aCS = aCS0->Dupl();
   ElRotation3D  aR =aCS->Orient();

   // int i = 3 +  5* 4;

   ElMatrix<double> aMPert = ElMatrix<double>::Rotation(RandAngle(),RandAngle(),RandAngle());
   aR = ElRotation3D(aR.tr(),aR.Mat()*aMPert,true);
   aCS->SetOrientation(aR);

   return aCS;
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
    return KthValProp(aVEr,0.75);
}

void cTest_PBGC3M2DF::OneIterBundle()
{
   // mVCT[0]->Show();
   mSet->SetPhaseEquation();

   for (int aKC=0 ; aKC<int(mVCT.size()) ; aKC++)
   {
       mVCT[aKC]->mFPC.AddEqAttachGlob(1e-1,true,20);
       mVCT[aKC]->mFPC.AddEqAttachGlob(1e-3,false,20);
   }
/*
*/

   double aErCple = AddBundle(mVCpleT,-1);
   std::cout << "ERCPLE " << aErCple << "\n";
   AddBundle(mVCpleT,aErCple);


   if (mVTriT.size())
   {
       double aErrTri = AddBundle(mVTriT,-1);
       AddBundle(mVTriT,aErrTri);
       std::cout << "ER TRI " << aErrTri  << "\n";
   }

DebugBundleGen = true;
   std::cout << "\n";
   mSet->SolveResetUpdate();

}

cTest_PBGC3M2DF::cTest_PBGC3M2DF(int argc,char ** argv)  :
    cAppliWithSetImage   (argc-1,argv+1,0),
    mSet                 (new cSetEqFormelles(cNameSpaceEqF::eSysPlein)),
    mDeg                 (3),
    mPerturbAng          (0.01),
    mPerturbPol          (0.0)
{
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
   );
   mNbSom = mVSoms.size();

   for (int aK=0 ; aK<mNbSom ; aK++)
   {
       mVCT.push_back(new cCamTest_PBGC3M2DF(*mVSoms[aK]->attr().mIma,*this,aK));
   }

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
                 if (aCple->mVP[0].size() )
                 {
                     cSubstitueBlocIncTmp * aBS = new cSubstitueBlocIncTmp(*mEqP3I);
                     aCple->mBufSub =  aBS;
                     aBS->AddInc(aC1->mFPC.IntervAppuisPtsInc());
                     aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                     aBS->Close();

// aBuf->AddInc(*(mVLInterv[aK]));

                     mVCpleT.push_back(aCple);
                     mMapCpleT[Pt2di(aK1,aK2)] = aCple;
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
                   if (aTri->mVP[0].size() > 5)
                   {
                       cSubstitueBlocIncTmp * aBS = new cSubstitueBlocIncTmp(*mEqP3I);
                       aTri->mBufSub =  aBS;
                       aBS->AddInc(aC1->mFPC.IntervAppuisPtsInc());
                       aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                       aBS->AddInc(aC3->mFPC.IntervAppuisPtsInc());

                       // aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                       // aBS->AddInc(aC3->mFPC.IntervAppuisPtsInc());
                       aBS->Close();


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
