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
   Sur le long terme, modifier :

     - la possiblite d'avoir des fonctions multiple (Dim N)
     - associer a un fctr, un cSetEqFormelles
*/

#include "StdAfx.h"




/************************************************************/
/*                                                          */
/*                   cParamIFDistPolynXY                    */
/*                                                          */
/************************************************************/

std::string  cParamIFDistPolynXY::NameType() const
{
    return  std::string("DPol") + ToString(mDist.Degre());
}
cParamIFDistPolynXY::~cParamIFDistPolynXY()
{
}

Pt2d<Fonc_Num> cParamIFDistPolynXY::VirtualDist(Pt2d<Fonc_Num> aP,bool UsePC,int aKCam)
{
   return mDist(aP);
}
cMultiContEQF cParamIFDistPolynXY::StdContraintes()
{
   return mDist.StdContraintes();
}

void cParamIFDistPolynXY::SetFige(INT DegreMax)
{
    mDist.SetFige(DegreMax,true);
}

cParamIFDistPolynXY::cParamIFDistPolynXY(bool isDC2M,cCamStenopeDistPolyn * aCam,cSetEqFormelles & aSet) :
	cParamIntrinsequeFormel(isDC2M,aCam,aSet,false),
	mDist(aCam->DistPol(),aSet),
	mCurPIF  (0)
{
    NV_UpdateCurPIF();
}

ElDistortionPolynomiale cParamIFDistPolynXY::DistCur() const
{
   return mDist.DistCur();
}


void  cParamIFDistPolynXY::NV_UpdateCurPIF()
{
    mCurPIF =  new cCamStenopeDistPolyn (DistIsC2M(),CurFocale(),CurPP(), DistCur(),CurAFoc());

    //  mCurPIF->SetDistInverse();
    mCurPIF->HeritComplAndSz(*CamInit());
}
void  cParamIFDistPolynXY::UpdateCurPIF() { NV_UpdateCurPIF(); }


CamStenope * cParamIFDistPolynXY::CurPIF() { return mCurPIF; }
cCamStenopeDistPolyn * cParamIFDistPolynXY::CurPIFPolyn() { return mCurPIF; }

CamStenope * cParamIFDistPolynXY::DupCurPIF() 
{ 
    cCamStenopeDistPolyn * aS = mCurPIF;
    NV_UpdateCurPIF();
    ElSwap(aS,mCurPIF);
    return aS;
}


/************************************************************/
/*                                                          */
/*                   cDistFormelPolXY                       */
/*                                                          */
/************************************************************/

class cPolynFormelXY :  public cElemEqFormelle
{
	public :
        // Tous les Poly sont intialises soit sur IdX soit sur IdY
            cPolynFormelXY
            (
                   const Polynome2dReal &    aPol,
                   cSetEqFormelles &   aSet,
		   bool                isX
	    );

	    Fonc_Num operator () (Pt2d<Fonc_Num> aP);
	    void AddFRap(cMultiContEQF &);
	    void SetFige(INT DegreMax,bool Std);
	    const Polynome2dReal & PolCur () const;

	private :
	    Polynome2dReal         mPolInit;
	    Polynome2dReal         mPolCur;
	    std::vector<Fonc_Num>  vCOeffs;
	    std::vector<bool   >   mFige;
	    bool                   mIsX;
	    double                 mPolXYTol;
};

cDistFormelPolXY::cDistFormelPolXY
(
   const ElDistortionPolynomiale & aDist,
   cSetEqFormelles & aSet 
)  :
    mDegre      (aDist.DistX().DMax()),
    mDegreFige  (mDegre+1)
{
/*
   Polynome2dReal aPX (mDegre,1.0); 
   aPX.SetDegre1(0.0,1.0,0.0);
   Polynome2dReal aPY (mDegre,1.0); 
   aPY.SetDegre1(0.0,0.0,1.0);
*/

   mPolX = new cPolynFormelXY(aDist.DistX(),aSet,true);
   mPolY = new cPolynFormelXY(aDist.DistY(),aSet,false);
   mPolX->SetFige(mDegreFige,true);
   mPolY->SetFige(mDegreFige,true);
}

cDistFormelPolXY::~cDistFormelPolXY()
{
   delete mPolX;
   delete mPolY;
}

Pt2d<Fonc_Num> cDistFormelPolXY::operator () (Pt2d<Fonc_Num> aP)
{
    return Pt2d<Fonc_Num>((*mPolX)(aP),(*mPolY)(aP));
}

INT cDistFormelPolXY::Degre() const
{
    return mDegre;
}

cMultiContEQF  cDistFormelPolXY::StdContraintes()
{
   cMultiContEQF aRes;
   mPolX->AddFRap(aRes);
   mPolY->AddFRap(aRes);

   return aRes;
}

void cDistFormelPolXY::SetFige(INT DegreMax,bool Std)
{
     mPolX->SetFige(DegreMax,Std);
     mPolY->SetFige(DegreMax,Std);
}

ElDistortionPolynomiale cDistFormelPolXY::DistCur() const
{
    return ElDistortionPolynomiale
           (
               mPolX->PolCur(),
               mPolY->PolCur()
           );
}

/************************************************************/
/*                                                          */
/*                    cPolynFormelXY                        */
/*                                                          */
/************************************************************/


cPolynFormelXY::cPolynFormelXY
(
      const Polynome2dReal &    aPol,
      cSetEqFormelles &   aSet,
      bool                isX
)   :
    cElemEqFormelle(aSet,false),
    mPolInit (aPol),
    mPolCur (aPol),
    mIsX    (isX),
    mPolXYTol    (cContrainteEQF::theContrStricte)
{
     for (INT aK = 0  ; aK<mPolCur.NbMonome() ; aK++)
     {
        std::string aNameMon = "X"+ToString(mPolCur.DegreX(aK)) + "Y"+ToString(mPolCur.DegreX(aK));
        vCOeffs.push_back(aSet.Alloc().NewF("cPolynFormelXY",aNameMon,&mPolCur.Coeff(aK)));
	mFige.push_back(true);
     }
     CloseEEF();
}

void cPolynFormelXY::SetFige(INT DegreMax,bool Std)
{
    for (INT aK = 0  ; aK<mPolCur.NbMonome() ; aK++)
    {
	    const Monome2dReal & aMon = mPolCur.KthMonome(aK);
	    mFige[aK] = (aMon.DegreTot() >= DegreMax);
	    if (Std)
	    {
                if ((aMon.DegreX() ==0) && (aMon.DegreY()==0))
                   mFige[aK] = true;
                if (mIsX & (aMon.DegreX() ==0) && (aMon.DegreY()==1))
                   mFige[aK] = true;
	    }
    }
}

void cPolynFormelXY::AddFRap(cMultiContEQF & aVF)
{
    for (INT aK = 0  ; aK<mPolCur.NbMonome() ; aK++)
        if (mFige[aK])
           AddFoncRappInit(aVF,aK,aK+1,mPolXYTol);
}

Fonc_Num cPolynFormelXY::operator () (Pt2d<Fonc_Num> aP)
{
    Fonc_Num aRes = 0;
    for (INT aK = 0  ; aK<mPolCur.NbMonome() ; aK++)
    {
	    const Monome2dReal & aMon = mPolCur.KthMonome(aK);
	    aRes = aRes +  vCOeffs[aK]
		         * PowI(aP.x,aMon.DegreX())
		         * PowI(aP.y,aMon.DegreY());
    }
    return aRes;
}

const Polynome2dReal & cPolynFormelXY::PolCur () const
{
   return mPolCur;
}

/************************************************************/
/*                                                          */
/*                    cDistRadialeFormelle                  */
/*                                                          */
/************************************************************/

cDistRadialeFormelle::cDistRadialeFormelle
(
     bool asIntervBlock,
     bool isCFige,
     int aDegFig,
     const ElDistRadiale_PolynImpair & aDist,
     cSetEqFormelles &                    aSet,
     cCamStenopeModStdPhpgr * aDP,
     const std::string &      anIdAlloc
)  :
     cElemEqFormelle(aSet,false),
     mCentreFige  (isCFige),
     mDegreFige   (aDegFig),
     mDistInitStd (aDist),
     mCurDistStd  (aDist),
     mDistInit    (mDistInitStd.DRad()),
     mCurDist     (mCurDistStd.DRad()),
     mIndCentre   (aSet.Alloc().CurInc()),
     mFCentre     (aSet.Alloc().NewPt2(anIdAlloc,mCurDist.Centre())),
     mTolCDist    (cContrainteEQF::theContrStricte),
     mTolCoeffs   (cContrainteEQF::theContrStricte)
{
   if (aDP)
   {
        mCurDistStd.P1() = aDP->DModPhgrStd().P1();
        mCurDistStd.P2() = aDP->DModPhgrStd().P2();
        mCurDistStd.b1() = aDP->DModPhgrStd().b1();
        mCurDistStd.b2() = aDP->DModPhgrStd().b2();
   }
   for (INT aK=0 ; aK<INT(mCurDist.NbCoeff()) ; aK++)
   {
       mFCoeffs.push_back(aSet.Alloc().NewF(anIdAlloc,"R"+ToString(3+2*aK),&(mCurDist.Coeff(aK))));
   }

   CloseEEF(asIntervBlock);

/// std::cout << "VPPPPPPPpp   " << mCurDist.RMax() << "\n"; getchar();
}

bool cDistRadialeFormelle::IsFiged() const
{
  return (mDegreFige==0) && mCentreFige;
}

void cDistRadialeFormelle::SetTolCDist(double aTol)
{
   mTolCDist = aTol;
}

void cDistRadialeFormelle::SetTolCoeff(double aTol)
{
   mTolCoeffs = aTol;
}


INT cDistRadialeFormelle::IndCentre() const
{
  return mIndCentre;
}

void cDistRadialeFormelle::SetDegreFige(INT aDF, double aTol)
{
   SetTolCoeff(aTol);
   mDegreFige  = aDF;
}


void  cDistRadialeFormelle::AddContrCentre(cMultiContEQF & aContr)
{
    if (mCentreFige)
       AddFoncRappInit(aContr,0,2,mTolCDist);
}


double  cDistRadialeFormelle::RMax() const
{
   return mDistInit.RMax();
}

void  cDistRadialeFormelle::AddContrCoeff(cMultiContEQF & aContr)
{
    for (int aK= 2+mDegreFige ; aK<2+mCurDist.NbCoeff() ;aK++)
    {
        double aR =  RMax();
	// std::cout << mTolCoeffs << " " << mTolCoeffs/pow(aR,3+2*(aK-1)) << "\n";
	double aTolN = (mTolCoeffs > 0) ? ( mTolCoeffs/pow(aR,3+2*(aK-1))) : mTolCoeffs;
        AddFoncRappInit(aContr,aK,aK+1,aTolN);
    }
}

void cDistRadialeFormelle::ActuRMax(CamStenope * aCam,bool C2M)
{
     
     double aRMax =  DMaxCoins
                     (
                            aCam->NormC2M(aCam->OrIntrImaC2M(Pt2dr(0,0))),
                            aCam->NormC2M(aCam->OrIntrImaC2M(Pt2dr(aCam->Sz()))),
                            mCurDist.Centre()
                     );

     if (! C2M)
     {
        aRMax = aRMax * (1+mCurDist.DistInverse(aRMax));
     }

     mCurDist.SetRMax(aRMax);
/*
*/
     // mCurDist.ActuRMax();
}



cMultiContEQF cDistRadialeFormelle::StdContraintes()
{
    cMultiContEQF  aRes;
    AddContrCentre(aRes);
    AddContrCoeff(aRes);

    return aRes;
}

cMultiContEQF   cDistRadialeFormelle::CentreFRapInit()
{
    cMultiContEQF  aRes;
    AddContrCentre(aRes);
    return aRes;
}

cMultiContEQF  cDistRadialeFormelle::CoeffsRapInit(INT aCoef0)
{
    cMultiContEQF  aRes;
    AddContrCoeff(aRes);
    return aRes;
}

Pt2d< Fonc_Num>  cDistRadialeFormelle::operator ()(Pt2d< Fonc_Num> aP)
{
    Pt2d<Fonc_Num> fEc = aP - mFCentre;
    Fonc_Num fRho2 = ElSquare(fEc.x) +ElSquare(fEc.y);
    Fonc_Num fSom = 1;
    Fonc_Num fRho2N  = 1;

    for (int aK=0 ; aK<(INT) mFCoeffs.size() ; aK++)
    {
        fRho2N = fRho2N * fRho2;
        fSom  = fSom + fRho2N * mFCoeffs[aK];
    }
    return  mFCentre +  Pt2d<Fonc_Num>(fEc.x*fSom,fEc.y*fSom);
}

const ElDistRadiale_PolynImpair & cDistRadialeFormelle::DistInit() const
{
    return mDistInit;
}

const ElDistRadiale_PolynImpair & cDistRadialeFormelle::DistCur() const
{
    return mCurDist;
}

cDistModStdPhpgr  & cDistRadialeFormelle::DistCurStd()
{
   return mCurDistStd;
}
const cDistModStdPhpgr & cDistRadialeFormelle::DistCurStd() const
{
   return mCurDistStd;
}

cDistModStdPhpgr  & cDistRadialeFormelle::DistInitStd()
{
   return mDistInitStd;
}
const cDistModStdPhpgr & cDistRadialeFormelle::DistInitStd() const
{
   return mDistInitStd;
}



void cDistRadialeFormelle::ReinitDist(const ElDistRadiale_PolynImpair & aD)
{
    mDistInit = mCurDist = aD;
    ReinitOnCur();
}

void cDistRadialeFormelle::SetCentreFige(bool isFiged)
{
    mCentreFige = isFiged;
}

Pt2d<Fonc_Num>  cDistRadialeFormelle::FCentre() const
{
   return mFCentre;
}



/************************************************************/
/*                                                          */
/*                cParamIntrinsequeFormel                   */
/*                                                          */
/************************************************************/

     // ============== cPIFRegulDist ====================

cPIFRegulDist::cPIFRegulDist() :
   mRegulDistDxyP1     (new cP2d_Etat_PhgrF("RegDistxy1")),
   mRegulDistDxyP2     (new cP2d_Etat_PhgrF("RegDistxy2")),
   mRegulDistDxyP3     (new cP2d_Etat_PhgrF("RegDistxy3")),
   mRegulDistDxyP4     (new cP2d_Etat_PhgrF("RegDistxy4")),

   mRegulDistDxxP1     (new cP2d_Etat_PhgrF("RegDistuu1")),
   mRegulDistDxxP2     (new cP2d_Etat_PhgrF("RegDistuu2")),
   mRegulDistDxxP3     (new cP2d_Etat_PhgrF("RegDistuu3")),

   // mRegulDistDyyP1     (new cP2d_Etat_PhgrF("RegDistuu1")),
   // mRegulDistDyyP2     (new cP2d_Etat_PhgrF("RegDistuu2")),
   // mRegulDistDyyP3     (new cP2d_Etat_PhgrF("RegDistuu3")),

   mRegulDistDxP1      (new cP2d_Etat_PhgrF("RegDistu1")),
   mRegulDistDxP2      (new cP2d_Etat_PhgrF("RegDistu2")),
   mRegulDistKnownDer  (new cP2d_Etat_PhgrF("RegDistu3")),

   // mRegulDistDyP1      (new cP2d_Etat_PhgrF("RegDistu1")),
   // mRegulDistDyP2      (new cP2d_Etat_PhgrF("RegDistu2")),

   mRegulDistValP1     (new cP2d_Etat_PhgrF("RegDistValP1")),
   mRegulDistKnownVal  (new cP2d_Etat_PhgrF("RegDistKnownVal")),

   mFER_DxDy        (0),
   mFER_Dxx         (0),
   mFER_Dx          (0),
   mFER_Val         (0),
   mImPdsDef        (1,1,1.0)
{
}

cPIFRegulConseq::cPIFRegulConseq(cParamIntrinsequeFormel * aPIF0,cParamIntrinsequeFormel * aSuiv,bool WithR,bool ForGenCode) :
   mSet      (*aSuiv->Set()),
   mPIF0     (aPIF0),
   mSuiv     (aSuiv),
   mWithR    (WithR),
   mNameType ("cRegCamConseq_" + std::string(mWithR? "CompR_": "")  +  aSuiv->NameType()),
   mRayConseq(new cP3d_Etat_PhgrF("Ray"))
{
   ELISE_ASSERT(aPIF0->Set()==aSuiv->Set(),"Different set in cPIFRegulConseq");
   InitInterv();
   mLInterv.AddInterv(mPIF0->IncInterv());
   mLInterv.AddInterv(mSuiv->IncInterv() );

   if (mWithR)
   {
      mEqP3I = mSet.Pt3dIncTmp();
      mLInterv.AddInterv(mEqP3I->IncInterv() );
   }
   if (!ForGenCode)
   {
      mFctr = cElCompiledFonc::AllocFromName(mNameType);
      if (mFctr==0)
      {
          std::cout << "For Type =" << mNameType << "\n";
          ELISE_ASSERT(false,"cPIFRegulConse cannot get Fctr");
      }
      mFctr->SetMappingCur(mLInterv,&mSet);
      mSet.AddFonct(mFctr);
      mRayConseq->InitAdr(*mFctr);
   }
   ResetInterv();
}

void cPIFRegulConseq::InitInterv()
{
     mSuiv->IncInterv().SetName("Intr2");
}

void cPIFRegulConseq::ResetInterv()
{
     mSuiv->IncInterv().SetName("Intr");
}
     // ============== cParamIntrinsequeFormel ====================

std::vector<std::string>   StdVectorOfName(const std::string & aPref,int aNb)
{
   std::vector<std::string> aRes;
   for (int aK=0 ; aK<aNb ;aK++)
       aRes.push_back(aPref+":"+ToString(aK));
   return aRes;
}

cParamIntrinsequeFormel::cParamIntrinsequeFormel
(
    bool              isDC2M,
    CamStenope  *     aCamInit,
    cSetEqFormelles & aSet,
    bool ParamVar
) :
   cElemEqFormelle (aSet,false),
   isDistC2M       (isDC2M),
   isFocaleFree    (false),
   isPPFree        (false),
   isAFoc1Free     (false),
   isAFoc2Free     (false),
   mCamInit        (aCamInit),
   mCurPIF         (0),
   mCamGrid        (0),
   mFiged          (false),
   mFocaleInit     (mCamInit->Focale()),
   mCurFocale      (mFocaleInit),
   mPPInit         (mCamInit->PP()),
   mCurPP          (mPPInit),
   mAFocInit       (mCamInit->ParamAF()),
   mCurAFoc        (mAFocInit),
   mFFoc           (  ParamVar ?
		      aSet.Alloc().NewF(aCamInit->IdentCam(),"F",&mCurFocale) :
		      Fonc_Num(mFocaleInit)
		   ),
   mIndPP          (aSet.Alloc().CurInc()),
   mFPP            (
		        ParamVar ?
		        aSet.Alloc().NewPt2(aCamInit->IdentCam(),&(mCurPP.x),&(mCurPP.y),false,"PPx","PPy") :
		        Pt2d<Fonc_Num>(mPPInit.x,mPPInit.y)
                   ),
   mIndAF          (aSet.Alloc().CurInc()),
   mFAFoc           (aSet.Alloc().NewVectInc("AFoc",StdVectorOfName("D",mCurAFoc.size()),mCurAFoc)),
   mParamVar        (ParamVar),
   mTolFoc          (cContrainteEQF::theContrStricte),
   mTolPP           (cContrainteEQF::theContrStricte),
   mTolAF1          (cContrainteEQF::theContrStricte),
   mTolAF2          (cContrainteEQF::theContrStricte),
   mProjStenF       (mFFoc,mFPP.x,mFPP.y,mFAFoc),
   mReg             (0),
   mRegCons         (0)
{
  if (GlobUseRegulDist)
      mReg = new  cPIFRegulDist;
  NV_UpdateCurPIF();
}

double  cParamIntrinsequeFormel::AddObsRegulConseq(int aNbGrids,double aSigmaPix)
{
    if (! mRegCons)
    {
        // std::cout << "RRetettttturnnn \n";
        return -1;
    }
    ELISE_ASSERT(!mRegCons->mWithR,"Rot not handled in AddObsRegulConseq");
    // Pas sur de l'utilite InitStateOfFoncteur ....
    InitStateOfFoncteur(mRegCons->mFctr,0);
    // mRegCons->InitInterv();

    CamStenope * aCamCur = CurPIF();
    double aPixFact =  aCamCur->ScaleCamNorm();
    double aPds = 1/ElSquare(aSigmaPix/ aPixFact);
    aPds /= 1.0/ElSquare(aNbGrids+1);


 

    double aSomRes = 0;
    double aNbRes = 0;

    for (int aKx=0 ; aKx<=aNbGrids ; aKx++)
    {
       for (int aKy=0 ; aKy<=aNbGrids ; aKy++)
       {
           Pt2dr aPProp(  (0.5+aKx)/(1+aNbGrids) , (0.5+aKy)/(1+aNbGrids)  );
           Pt2dr aPIm = aPProp.mcbyc(Pt2dr(mCamInit->Sz()));

           Pt3dr aDirRay =   aCamCur->F2toDirRayonL3(aPIm);
           mRegCons->mRayConseq->SetEtat(aDirRay);


           std::vector<double>  aVRes = mSet.VAddEqFonctToSys(mRegCons->mFctr,aPds,false,NullPCVU);
           aSomRes += ElSquare(aVRes.at(0)) + ElSquare(aVRes.at(1));
           aNbRes ++;
       }
    }
    aSomRes = sqrt(aSomRes/aNbRes) * aPixFact;
 
    // std::cout << "Hhhhhhhhhhh " << mRegCons->mNameType << " " << aSomRes << "\n";
    return aSomRes;
    // mRegCons->ResetInterv();
}

void cParamIntrinsequeFormel::AddRegulConseq(cParamIntrinsequeFormel* aSuiv,bool WithR,bool ForGenCode)
{
   ELISE_ASSERT
   (
      NameType() == aSuiv->NameType(),
      "Mix camera type in AddRegulConseq"
   );
   mRegCons = new cPIFRegulConseq(this,aSuiv,WithR,ForGenCode);
}

void  cParamIntrinsequeFormel::Virtual_CloseEEF()
{
    IncInterv().SetName("Intr");
    mLInterv.AddInterv(IncInterv());


    if (mReg)
    {
        mReg->mNameRegDistDxDy = "cREgDistDxDy_" +  NameType();
        mReg->mNameRegDistD2   = "cREgDistDxx_"  +  NameType();
        mReg->mNameRegDistGrad = "cREgDistDx_"   +  NameType();
        mReg->mNameRegDistVal  = "cREgDistVal_"  +  NameType();

        mReg->mFER_DxDy = cElCompiledFonc::AllocFromName(mReg->mNameRegDistDxDy);
        mReg->mFER_Dxx  = cElCompiledFonc::AllocFromName(mReg->mNameRegDistD2);
        mReg->mFER_Dx   = cElCompiledFonc::AllocFromName(mReg->mNameRegDistGrad);
        mReg->mFER_Val  = cElCompiledFonc::AllocFromName(mReg->mNameRegDistVal);

        bool OkFER = mReg->mFER_DxDy && mReg->mFER_Dxx && mReg->mFER_Dx && mReg->mFER_Val;

        if (OkFER)
        {
           mReg->mFER_DxDy->SetMappingCur(mLInterv,&mSet);
           mSet.AddFonct(mReg->mFER_DxDy);

           mReg->mFER_Dxx->SetMappingCur(mLInterv,&mSet);
           mSet.AddFonct(mReg->mFER_Dxx);

       // mFER_Dyy  = cElCompiledFonc::AllocFromName(mNameRegDistD2);
           mReg->mFER_Dx->SetMappingCur(mLInterv,&mSet);
           mSet.AddFonct(mReg->mFER_Dx);

       // mFER_Dy   = cElCompiledFonc::AllocFromName(mNameRegDistGrad);
           mReg->mFER_Val->SetMappingCur(mLInterv,&mSet);
           mSet.AddFonct(mReg->mFER_Val);
        }

        if (OkFER)
        {
           mReg->mRegulDistDxyP1->InitAdr(*mReg->mFER_DxDy);
           mReg->mRegulDistDxyP2->InitAdr(*mReg->mFER_DxDy);
           mReg->mRegulDistDxyP3->InitAdr(*mReg->mFER_DxDy);
           mReg->mRegulDistDxyP4->InitAdr(*mReg->mFER_DxDy);


           mReg->mRegulDistDxxP1->InitAdr(*mReg->mFER_Dxx);
           mReg->mRegulDistDxxP2->InitAdr(*mReg->mFER_Dxx);
           mReg->mRegulDistDxxP3->InitAdr(*mReg->mFER_Dxx);

           mReg->mRegulDistDxP1->InitAdr(*mReg->mFER_Dx);
           mReg->mRegulDistDxP2->InitAdr(*mReg->mFER_Dx);
           mReg->mRegulDistKnownDer->InitAdr(*mReg->mFER_Dx);

           mReg->mRegulDistValP1->InitAdr(*mReg->mFER_Val);
           mReg->mRegulDistKnownVal->InitAdr(*mReg->mFER_Val);
        }
    }

    // std::cout << "cParamIntrinsequeFormel::Virtual_CloseEEF " << mFER_Dxx << " ## " << mFER_Dyy << "\n";
/*
    ELISE_ASSERT
    (
         (mFER_DxDy!=0) && (mFER_Dxx!=0) && (mFER_Dx!=0) && (mFER_Val!=0),
         "cParamIntrinsequeFormel::Virtual_CloseEEF  cannot load regul fonctor"
    );
*/

}

// DirRayMonde2Cam


void cParamIntrinsequeFormel::GenEqRegulDist()
{
    AssertRegIsInit();
    // bool aUP=   Dist22Gen_UsePreConditionner();

    Pt2d<Fonc_Num> aRegDistDxDy =     DistorM2C(mReg->mRegulDistDxyP1->PtF(),false) + DistorM2C(mReg->mRegulDistDxyP2->PtF(),false)
                                   - (DistorM2C(mReg->mRegulDistDxyP3->PtF(),false) + DistorM2C(mReg->mRegulDistDxyP4->PtF(),false));
    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,
        mReg->mNameRegDistDxDy,
        aRegDistDxDy.ToTab(),
        mLInterv
    );

    Pt2d<Fonc_Num> aRegDistDxx =     DistorM2C(mReg->mRegulDistDxxP1->PtF(),false) + DistorM2C(mReg->mRegulDistDxxP2->PtF(),false)
                                   - DistorM2C(mReg->mRegulDistDxxP3->PtF(),false)  * 2.0;
    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,
        mReg->mNameRegDistD2,
        aRegDistDxx.ToTab(),
        mLInterv
    );

    Pt2d<Fonc_Num> aRegDistGrad =     DistorM2C(mReg->mRegulDistDxP1->PtF(),false) - DistorM2C(mReg->mRegulDistDxP2->PtF(),false) - mReg->mRegulDistKnownDer->PtF();
    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,
        mReg->mNameRegDistGrad,
        aRegDistGrad.ToTab(),
        mLInterv
    );

    Pt2d<Fonc_Num> aRegDistVal =     DistorM2C(mReg->mRegulDistValP1->PtF(),false) - mReg->mRegulDistKnownVal->PtF();
    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,
        mReg->mNameRegDistVal,
        aRegDistVal.ToTab(),
        mLInterv
    );


    if (mRegCons)
    {
        // mEqP3I      mCam.Set().Pt3dIncTmp()
/*
        mRegCons->mLInterv.AddInterv(IncInterv());
        mRegCons->mLInterv.AddInterv(mRegCons->mSuiv->IncInterv() );
        if (mRegCons->mWithR)
           mRegCons->mLInterv.AddInterv(mRegCons->mEqP3I->IncInterv() );

        mRegCons->mSuiv->IncInterv().SetName("Intr2");
*/
        mRegCons->InitInterv();

        Pt3d<Fonc_Num>  aRay1 = mRegCons->mRayConseq->PtF();
        Pt3d<Fonc_Num>  aRay2 = aRay1;
        if (mRegCons->mWithR)
           aRay2 = aRay1 + (mRegCons->mEqP3I->PF() ^ aRay1);
        Pt2d<Fonc_Num>  aP1 = DirRayMonde2Cam(aRay1,0);
        Pt2d<Fonc_Num>  aP2 = mRegCons->mSuiv->DirRayMonde2Cam(aRay2,0);
        Pt2d<Fonc_Num> aDif = aP1 - aP2;
        cElCompileFN::DoEverything
        (
            DIRECTORY_GENCODE_FORMEL,
            mRegCons->mNameType,
            aDif.ToTab(),
            mRegCons->mLInterv
        );
            // mLInterv.AddInterv(mCam.RF().IncInterv());
            // mLInterv.AddInterv(mEqP3I->IncInterv());
        // mRegCons->mSuiv->IncInterv().SetName("Intr");
        mRegCons->ResetInterv();
    }
}


void cParamIntrinsequeFormel::AddCstrRegulDist(Pt2dr aP,double aPdsVal,double aPdsGrad,double aPdsD2)
{
   AssertRegIsInit();
   InitStateOfFoncteur(mReg->mFER_Val,0);
   InitStateOfFoncteur(mReg->mFER_Dx,0);
   InitStateOfFoncteur(mReg->mFER_DxDy,0);
   InitStateOfFoncteur(mReg->mFER_Dxx,0);
   bool Show = false;
// Val  
    std::vector<double> aVPv(2,aPdsVal);
    mReg->mRegulDistValP1->SetEtat(aP);
    mReg->mRegulDistKnownVal->SetEtat(aP);
    std::vector<double> aRes;

    aRes = mSet.VAddEqFonctToSys(mReg->mFER_Val,aVPv,false,NullPCVU);

    double aEps = CurFocale() / 50.0;



//   Grad
    std::vector<double> aVPGrad(2,aPdsGrad/(2.0*ElSquare(aEps)));
     // x 
    mReg->mRegulDistDxP1->SetEtat(aP+Pt2dr( aEps,0));
    mReg->mRegulDistDxP2->SetEtat(aP+Pt2dr(-aEps,0));
    mReg->mRegulDistKnownDer->SetEtat(Pt2dr(2*aEps,0));
    aRes = mSet.VAddEqFonctToSys(mReg->mFER_Dx,aVPGrad,false,NullPCVU);


     // y 
    mReg->mRegulDistDxP1->SetEtat(aP+Pt2dr(0, aEps));
    mReg->mRegulDistDxP2->SetEtat(aP+Pt2dr(0,-aEps));
    mReg->mRegulDistKnownDer->SetEtat(Pt2dr(0,2*aEps));
    aRes = mSet.VAddEqFonctToSys(mReg->mFER_Dx,aVPGrad,false,NullPCVU);

    if (Show) 
        std::cout << aRes.size() << "GG " <<  aRes[0]/aEps << " " << aRes[1]/aEps << "\n";

//   Deriv sec
    std::vector<double> aVPCourb(2,aPdsD2/ElSquare(aEps*aEps));
     // xx  P1 + P2 - 2* P3
    mReg->mRegulDistDxxP1->SetEtat(aP+Pt2dr( aEps,0));
    mReg->mRegulDistDxxP2->SetEtat(aP+Pt2dr(-aEps,0));
    mReg->mRegulDistDxxP3->SetEtat(aP);
    aRes= mSet.VAddEqFonctToSys(mReg->mFER_Dxx,aVPCourb,false,NullPCVU);
    if (Show) 
        std::cout << aRes.size() << "Dxxxx " <<  aRes[0]/ElSquare(aEps) << " " <<  aRes[1]/ElSquare(aEps) << "\n";

     // yy
    mReg->mRegulDistDxxP1->SetEtat(aP+Pt2dr(0, aEps));
    mReg->mRegulDistDxxP2->SetEtat(aP+Pt2dr(0,-aEps));
    mReg->mRegulDistDxxP3->SetEtat(aP);
    aRes = mSet.VAddEqFonctToSys(mReg->mFER_Dxx,aVPCourb,false,NullPCVU);
    if (Show) 
        std::cout << aRes.size() << "Dyyy " <<  aRes[0]/ElSquare(aEps) << " " <<  aRes[1]/ElSquare(aEps) << "\n";

     // yy
    mReg->mRegulDistDxyP1->SetEtat(aP+Pt2dr( aEps, aEps));
    mReg->mRegulDistDxyP2->SetEtat(aP+Pt2dr(-aEps,-aEps));
    mReg->mRegulDistDxyP3->SetEtat(aP+Pt2dr(-aEps, aEps));
    mReg->mRegulDistDxyP4->SetEtat(aP+Pt2dr( aEps,-aEps));
    aRes = mSet.VAddEqFonctToSys(mReg->mFER_DxDy,aVPCourb,false,NullPCVU);
    if (Show) 
       std::cout << aRes.size() << "DXXYY " <<  aRes[0]/ElSquare(aEps) << " " <<  aRes[1]/ElSquare(aEps) << "\n";
}

void cParamIntrinsequeFormel::AddCstrRegulGlob(int aNbEch,double aPdsVal,double aPdsGrad,double aPdsD2,Im2D_REAL4 * aFoncPds)
{
    AssertRegIsInit();
    if (aFoncPds==0) 
    {
       aFoncPds = &mReg->mImPdsDef;
    }
    TIm2D<REAL4,REAL8> aTPds(*aFoncPds);

    double aSomPds = 0.0;

    for (int aKIter = 0 ; aKIter<2 ;  aKIter++)
    {
        for (int anX=0 ; anX<=aNbEch ; anX++)
        {
             for (int anY=0 ; anY<=aNbEch ; anY++)
             {
                  Pt2dr aPProp(anX/double(aNbEch),anY/double(aNbEch));
                  Pt2dr aPPds = aPProp.mcbyc(Pt2dr(aFoncPds->sz()));
                  double aPds = aTPds.getprojR(aPPds);
                  if (aKIter==0)
                  {
                     aSomPds += aPds;
                  }
                  else
                  {
                     if (aSomPds > 1e-6)
                     {
                         Pt2dr aPIm = aPProp.mcbyc(Pt2dr(mCamInit->Sz()));
                         aPIm = CurPIF()->NormC2M(aPIm);
                         aPds /= aSomPds;
                         AddCstrRegulDist(aPIm,aPdsVal*aPds,aPdsGrad*aPds,aPdsD2*aPds);
                     }
                     
// void cParamIntrinsequeFormel::AddCstrRegulDist(Pt2dr aP,double aPdsVal,double aPdsGrad,double aPdsD2)
                  }
             }
        }
    }

}

void cParamIntrinsequeFormel::AssertRegIsInit()
{
   ELISE_ASSERT(mReg,"cParamIntrinsequeFormel::AssertRegIsIni");
}


void cParamIntrinsequeFormel::AddRapViscosite(double aTol)
{
    for (int aK=0 ; aK<NbInc() ; aK++)
    {
        AddRappViscosite
        (
             "cParamIntrinsequeFormel::AddViscosite",
             true,
             aK,
             aTol,
             true
        );
   }
}

void cParamIntrinsequeFormel::AddToListInterval(cIncListInterv & aLInterv)
{
   aLInterv.AddInterv(IncInterv());
}

void  cParamIntrinsequeFormel::PrepareEqFForPointIm
      (const cIncListInterv &,cElCompiledFonc * anEq,const Pt2dr & aPIm,bool EqDroite,int aKCam)
{
}

bool cParamIntrinsequeFormel::UseAFocal() const
{
   return mProjStenF.UseAFocal();
}

void cParamIntrinsequeFormel::AssertNoAFocalParam(const std::string & aMes)
{
   if (mProjStenF.UseAFocal())
   {
       std::cout << "Dans le contexte " << aMes << "\n";
       ELISE_ASSERT(false,"Utilisation de parametre Afocal proscrit");
   }
}

cCamStenopeGrid * cParamIntrinsequeFormel::CamGrid()
{
   return mCamGrid;
}

void cParamIntrinsequeFormel::UpdateCamGrid( double aTol)
{
   if (UseAFocal()  &&  (! AFocalAcceptNoDist))
      return;
   bool aLastFiged = mFiged;
   mFiged = AllParamIsFiged() ;
   // Si rien n'a change
   if (aLastFiged == mFiged)
      return;

   // On passe de fige a mobile, donc plus de grille
   if (aLastFiged)
   {
        delete mCamGrid;
        mCamGrid =0;
        return;
   }

   //   On se fige
   //
 
      CamStenope * aCS = CurPIF();

      Pt2di aRab(20,20);
      Pt2di aStep(10,10);
      double aRabR = 0;
      Pt2di aSz = aCS->Sz();

      double aR = euclid(aSz)/2.0;
      if (aCS->HasRayonUtile())
         aR = aCS->RayonUtile();

// std::cout << "iuytml RAY = " << aR << " SZR " <<  euclid(aSz)/2.0 << "\n";

      mCamGrid = cCamStenopeGrid::Alloc(aR+aRabR,*aCS,Pt2dr(aStep),true,true);
      if (mCamGrid)
      {

          mRayonGrid = 1e20;

// std::cout << "iuytml END CGID  \n";

           double aEcMax =0;
           for (int aKx=100 ; aKx< aCS->Sz().x ; aKx += 200)
           {
               for (int aKy=100 ; aKy< aCS->Sz().y ; aKy += 200)
               {
                   Pt2dr aP(aKx,aKy);
                   // std::cout << "IZU " << mCamGrid->IsInZoneUtile(aP) << " " << aP << "\n";
                   if (mCamGrid->IsInZoneUtile(aP))
                   {
                      // UVGCC4.6 double aEps=1e-5;

                       Pt3dr aRay = aCS->F2toDirRayonL3(aP);
                       // UVGCC4.6 Pt3dr aRayG = mCamGrid->F2toDirRayonL3(aP);


                       // UVGCC4.6 Pt3dr  aRayX = aRay + Pt3dr(aEps,0,0);
                       // UVGCC4.6 Pt3dr  aRayY = aRay + Pt3dr(0,aEps,0);

                       Pt2dr aP1 = aCS->L3toF2(aRay);
                       Pt2dr aPG = mCamGrid->L3toF2(aRay);


                       // UVGCC4.6 Pt2dr aDx = (mCamGrid->L3toF2(aRayX)-aPG)/aEps;
                       // UVGCC4.6 Pt2dr aDy = (mCamGrid->L3toF2(aRayY)-aPG)/aEps;


                       Pt2dr aDGX,aDGY;
                       // UVGCC4.6 Pt2dr aPG2 = mCamGrid->L2toF2AndDer(Pt2dr(aRay.x,aRay.y),aDGX,aDGY);

                // std::cout << aPG << aDx << aDy << "\n";
                // std::cout <<  "    " << aPG2 << aDGX << aDGY << "\n";

                       double aDist = euclid(aP1,aPG);
                       if ( aDist >aTol)
                          mRayonGrid = ElMin(mRayonGrid,euclid(aP,aSz/2.0));
                       aEcMax = ElMax(aDist,aEcMax);
                    }
               }
           }
/*

        std::cout << "GetC ------------RTOl------------ " << mRayonGrid   << " -EcMax " << aEcMax << "\n"; 
         getchar();
*/

    }


}

bool cParamIntrinsequeFormel::IsDistFiged() const
{
   return  true;
}


bool cParamIntrinsequeFormel::AllParamIsFiged() const
{
   return      IsDistFiged() 
           &&  (! isFocaleFree) 
           &&  (! isPPFree)
           &&  ((!UseAFocal()) || ((!isAFoc1Free)&&(!isAFoc2Free)))
         ;
}

INT cParamIntrinsequeFormel::IndPP() const
{
    ELISE_ASSERT(mParamVar,"cParamIntrinsequeFormel::IndPP");
    return mIndPP;
}

void cParamIntrinsequeFormel::SetAF1Free(bool aF1)
{
   if (aF1 )
      ELISE_ASSERT(mParamVar,"cParamIntrinsequeFormel::SetAF1Free");
   isAFoc1Free = aF1;
}


void cParamIntrinsequeFormel::SetAF2Free(bool aF2)
{
   if (aF2 )
      ELISE_ASSERT(mParamVar,"cParamIntrinsequeFormel::SetAF2Free");
   isAFoc2Free = aF2;
}




void cParamIntrinsequeFormel::SetFocFree(bool FF)
{
   if (FF ) 
      ELISE_ASSERT(mParamVar,"cParamIntrinsequeFormel::SetFocFree");
   isFocaleFree = FF;
}
void cParamIntrinsequeFormel::SetFocFigee(double aTol)
{
    mTolFoc = aTol;
    SetFocFree(false);
}

void cParamIntrinsequeFormel::SetPPFree(bool PFF)
{
   if (PFF) 
      ELISE_ASSERT(mParamVar,"cParamIntrinsequeFormel::SetFocFree");
   isPPFree  = PFF;
}

void cParamIntrinsequeFormel::SetPPFFige(double aTol)
{
   mTolPP = aTol;
   SetPPFree(false);
}




Pt2d<Fonc_Num> cParamIntrinsequeFormel::FPP()
{
   return mFPP;
}
Fonc_Num       cParamIntrinsequeFormel::FFoc()
{
   return mFFoc;
}


cMultiContEQF cParamIntrinsequeFormel::StdContraintes()
{

    

    cMultiContEQF aRes;

    if ( mParamVar)
    {
        if (! isFocaleFree)
        {
           //   std::cout <<  "  SFFF  \n";
           AddFoncRappInit(aRes,0,1,mTolFoc);
        }

        if (! isPPFree)
           AddFoncRappInit(aRes,1,3,mTolPP);

        if (UseAFocal())
        {
           if (! isAFoc1Free)
              AddFoncRappInit(aRes,3,4,mTolAF1);

           if (! isAFoc2Free)
              AddFoncRappInit(aRes,4,5,mTolAF2);
        }

    }

    return aRes;
}

void cParamIntrinsequeFormel::InitStateOfFoncteur(cElCompiledFonc *,int aKCam)
{
}


cParamIntrinsequeFormel::~cParamIntrinsequeFormel() {}

Pt2d<Fonc_Num> cParamIntrinsequeFormel::VirtualDist(Pt2d<Fonc_Num> aP,bool UsePC,int aKCam)
{
   return aP;
}




Pt2d<Fonc_Num> cParamIntrinsequeFormel::DistorM2C(Pt2d<Fonc_Num> aP,bool UsePC,int aK)
{
   ELISE_ASSERT(!isDistC2M,"No cParamIntrinsequeFormel is C2M");
   return VirtualDist(aP,UsePC,aK);
}

Pt2d<Fonc_Num> cParamIntrinsequeFormel::DistorC2M(Pt2d<Fonc_Num> aP,bool UsePC,int aK)
{
   ELISE_ASSERT(isDistC2M,"No cParamIntrinsequeFormel is M2C");
   return VirtualDist(aP,UsePC,aK);
}


bool cParamIntrinsequeFormel::DistIsC2M() const
{
	return isDistC2M;
}

Pt3d<Fonc_Num> cParamIntrinsequeFormel::Cam2DirRayMonde(Pt2d<Fonc_Num> aP,int aKCam)
{
   return CorrDist2DirRayMonde(DistorC2M(aP,true,aKCam));
}

Fonc_Num  cParamIntrinsequeFormel::NormGradC2M(Pt2d<Fonc_Num> )
{
    return 1.0;
}

bool cParamIntrinsequeFormel::UseSz() const
{
   return false;
}

Pt3d<Fonc_Num>   cParamIntrinsequeFormel::CorrDist2DirRayMonde(Pt2d<Fonc_Num> aP)
{
    return Pt3d<Fonc_Num> 
	   (
	      (aP.x-mFPP.x)/mFFoc,
	      (aP.y-mFPP.y)/mFFoc,
	      1.0
	   );
}

Pt2d<Fonc_Num>   cParamIntrinsequeFormel::DirRayMonde2CorrDist(Pt3d<Fonc_Num> aP)
{
///  C'EST LA ....
/*
   aP = aP/aP.z;

   return Pt2d<Fonc_Num>(mFPP.x+mFFoc*aP.x,mFPP.y+mFFoc*aP.y);
*/
   Pt2d<Fonc_Num> aRes;
   mProjStenF.Proj(aRes.x,aRes.y,aP.x,aP.y,aP.z);
   return aRes;
}


Pt2d<Fonc_Num>   cParamIntrinsequeFormel::DirRayMonde2Cam(Pt3d<Fonc_Num>  aP,int aKCam)
{
   return DistorM2C(DirRayMonde2CorrDist(aP),true,aKCam);
}

std::string cParamIntrinsequeFormel::NameType() const
{
   return std::string ( mParamVar ? "NoDist" : "NoVar" ) ;
}


cCameraFormelle * cParamIntrinsequeFormel::NewCam
                 (
                      eModeContrRot aMode,
                      ElRotation3D aRot,
                      cCameraFormelle * pCAtt,
                      const std::string & aName,
		      bool  CompEqAppui,
		      bool GenCodeAppui,
                      bool HasEqDr
                 )
{
   cCameraFormelle * aRes =  new cCameraFormelle(aMode,aRot,*this,pCAtt,aName,CompEqAppui,GenCodeAppui,HasEqDr);
   mSet.AddCamFormelle(aRes,aName);
   return aRes;
}

const tParamAFocal  &  cParamIntrinsequeFormel::CurAFoc() const
{
    return mCurAFoc;
}


REAL  cParamIntrinsequeFormel::CurFocale() const
{
    return mCurFocale;
}

Pt2dr  cParamIntrinsequeFormel::CurPP() const
{
    return mCurPP;
}


void  cParamIntrinsequeFormel::NV_UpdateCurPIF() 
{
  mCurPIF =  new CamStenopeIdeale(DistIsC2M(),CurFocale(),CurPP(),CurAFoc());
  // mCurPIF->SetDistInverse();
  mCurPIF->HeritComplAndSz(*CamInit());
}
void  cParamIntrinsequeFormel::UpdateCurPIF() {NV_UpdateCurPIF();}
CamStenope * cParamIntrinsequeFormel::CurPIF() { return mCurPIF; }

CamStenope * cParamIntrinsequeFormel::DupCurPIF() 
{ 
    CamStenopeIdeale * aS = mCurPIF;
    NV_UpdateCurPIF();
    ElSwap(aS,mCurPIF);
    return aS;
}

CamStenope * cParamIntrinsequeFormel::CamInit()
{
   return mCamInit;
}

double cParamIntrinsequeFormel::StdScaleN() const
{
   return mCamGrid ? mCamGrid->ScaleCamNorm() : mCamInit->ScaleCamNorm();
}

double cParamIntrinsequeFormel::StdScaleNNoGrid() const
{
// std::cout << "cParamIntrinsequeFormel::StdScaleNNoGrid\n";
// return 1/0.006;

// std::cout <<  "StdScaleNNoG  "  << mCamInit->ScaleCamNorm() << "\n"; getchar();

   return mCamInit->ScaleCamNorm();
}

Pt2dr  cParamIntrinsequeFormel::CorrigePFromDAdd(const Pt2dr & aP,bool UseGrid)
{
  if (UseGrid && mCamGrid)
     return aP;
   return mCamInit->DComplC2M(aP);
}


void cParamIntrinsequeFormel::Update_0F2D()
{
   UpdateCurPIF();
}



/************************************************************/
/*                                                          */
/*                cParamIFDistRadiale                       */
/*                                                          */
/************************************************************/

ElDistRadiale_PolynImpair SetToDeg(const ElDistRadiale_PolynImpair & aPol,INT aDeg)
{
   ElDistRadiale_PolynImpair aRes(aPol.RMax(),aPol.Centre());

   for (INT aK=0; aK<aDeg ; aK++)
   {
       aRes.PushCoeff(aPol.CoeffGen(aK));
   }

   return aRes;
}


cParamIFDistRadiale::cParamIFDistRadiale
(
       bool                    isDC2M,
       cCamStenopeDistRadPol * aCam,
       cSetEqFormelles & aSet,
       int aDegFig,
       cCamStenopeModStdPhpgr * aDP
) :
  cParamIntrinsequeFormel  (isDC2M,aCam,aSet,true),
  mCDistPPLie              (false),
  mFoncEqPPCDistX          (0),
  mFoncEqPPCDistY          (0),
  mDRF                     (false,true,aDegFig,SetToDeg(aCam->DRad(),5),aSet,aDP, aDP?aDP->IdentCam(): (aCam?aCam->IdentCam():"NoId::ParamIFDistRadiale")),
  mCurPIF                  (0)
{
  NV_UpdateCurPIF();
}


std::string  cParamIFDistRadiale::NameType() const
{
    return  std::string("DRad") 
	    + ToString(mDRF.DistInit().NbCoeff());
}

Pt2d<Fonc_Num> cParamIFDistRadiale::VirtualDist(Pt2d<Fonc_Num> aP,bool UsePC,int aKCam)
{
   return mDRF(aP);
}


void cParamIFDistRadiale::SetPPFree(bool )
{
   ELISE_ASSERT(false,"cParamIFDistRadiale::SetPPFree");
}

void cParamIFDistRadiale::SetLibertePPAndCDist(bool PPF,bool CDF,double aTol)
{
   if (PPF && CDF)
   {
       ELISE_ASSERT
       (
            aTol<0,
	    "Incoherent de donner une tol  pour liberer les param"
       );
   }
   mDRF.SetTolCDist(aTol);
   mTolPP = aTol;


   cParamIntrinsequeFormel::SetPPFree(PPF);
   mDRF.SetCentreFige(! CDF);
   mCDistPPLie = false;
}

void cParamIFDistRadiale::SetCDistPPLie(double aTol)
{
   mTolPP = aTol;

   cParamIntrinsequeFormel::SetPPFree(true);
   mDRF.SetCentreFige(false);
   mCDistPPLie = true;
}


void cParamIFDistRadiale::SetDRFDegreFige(INT aDF,double aTol)
{
	mDRF.SetDegreFige(aDF,aTol);
}

void cParamIFDistRadiale::SetValInitOnValCur()
{
     mDRF.SetValInitOnValCur();
     cElemEqFormelle::SetValInitOnValCur();
}


cMultiContEQF cParamIFDistRadiale::StdContraintes()
{
    cMultiContEQF aRes = cParamIntrinsequeFormel::StdContraintes();
    cMultiContEQF aRDRF = mDRF.StdContraintes();
    aRes.Add(aRDRF);

    if (mCDistPPLie)
    {
        if (mFoncEqPPCDistX==0)
	{
           INT iP = IndPP();
	   INT iC = mDRF.IndCentre();
           mFoncEqPPCDistX = cElCompiledFonc::FoncSetValsEq(&mSet,iP,iC);
           mFoncEqPPCDistY = cElCompiledFonc::FoncSetValsEq(&mSet,iP+1,iC+1);
	}
	aRes.AddAcontrainte(mFoncEqPPCDistX,mTolPP);
	aRes.AddAcontrainte(mFoncEqPPCDistY,mTolPP);
    }

    return aRes;
}

const ElDistRadiale_PolynImpair & cParamIFDistRadiale::DistInit() const
{
	return mDRF.DistInit();
}
const ElDistRadiale_PolynImpair & cParamIFDistRadiale::DistCur() const
{
	return mDRF.DistCur();
}

void    cParamIFDistRadiale::NV_UpdateCurPIF()
{
    mDRF.ActuRMax(CamInit(),false);
    mCurPIF =  new cCamStenopeDistRadPol(DistIsC2M(),CurFocale(),CurPP(),DistCur(),CurAFoc());

    // mCurPIF->SetDistInverse();
    mCurPIF->HeritComplAndSz(*CamInit());
}
void  cParamIFDistRadiale::UpdateCurPIF(){ NV_UpdateCurPIF();}
CamStenope * cParamIFDistRadiale::CurPIF() { return CurPIFPolRad(); }
cCamStenopeDistRadPol  * cParamIFDistRadiale::CurPIFPolRad() {return mCurPIF;}
CamStenope * cParamIFDistRadiale::DupCurPIF() 
{ 
    cCamStenopeDistRadPol * aS = mCurPIF;
    NV_UpdateCurPIF();
    ElSwap(aS,mCurPIF);
    return aS;
}

bool cParamIFDistRadiale::IsDistFiged() const
{
   return  mDRF.IsFiged();
}

/************************************************************/
/*                                                          */
/*                cParamIFDistStdPhgr                       */
/*                                                          */
/************************************************************/
cDistModStdPhpgr  DistForInt(const cCamStenopeModStdPhpgr & aCam)
{
    cDistModStdPhpgr aDistDeg = aCam.DModPhgrStd();
    aDistDeg.DRad() = SetToDeg(aDistDeg.DRad(),5);
    return aDistDeg;
}

cParamIFDistStdPhgr::cParamIFDistStdPhgr
(
     bool  isDC2M,
     cCamStenopeModStdPhpgr * aCam,cSetEqFormelles & aSet,
     int aDegFig
)   :
    cParamIFDistRadiale(isDC2M,aCam,aSet,aDegFig,aCam) ,
    mDStd              (mDRF.DistCurStd()),
    mInitDStd          (mDRF.DistInitStd()),
    mParamDecentreFree (false),
    mParamAffineFree   (false),
    mFP1               (aSet.Alloc().NewF(aCam->IdentCam(),"P1",&(mDStd.P1()))),
    mFP2               (aSet.Alloc().NewF(aCam->IdentCam(),"P2",&(mDStd.P2()))),
    mFb1               (aSet.Alloc().NewF(aCam->IdentCam(),"b1",&(mDStd.b1()))),
    mFb2               (aSet.Alloc().NewF(aCam->IdentCam(),"b2",&(mDStd.b2()))),
    mTol_Dec_PhgStdPIF (cContrainteEQF::theContrStricte),
    mTol_Aff_PhgStdPIF (cContrainteEQF::theContrStricte),
    mCurPIF            (0)
{
// aSet.ShowVar(); std::cout <<"=======================\n";
//aSet.ShowVar(); std::cout << mDStd.P1() << " " << mDStd.P2() << " 11111\n"; getchar();
    cDistModStdPhpgr aDistDeg = aCam->DModPhgrStd();
    aDistDeg.DRad() = SetToDeg(aDistDeg.DRad(),5);

    mDStd = aDistDeg;
    mInitDStd = aDistDeg;
    SetCDistPPLie();

    NV_UpdateCurPIF();
//aSet.ShowVar(); std::cout << "22222\n"; getchar();
}

bool cParamIFDistStdPhgr::IsDistFiged() const
{
   return    cParamIFDistRadiale::IsDistFiged()
          && (!mParamDecentreFree)
          && (!mParamAffineFree);
}

void cParamIFDistStdPhgr::SetParam_Aff_Fige(double aTol)
{
   mParamAffineFree = false;
   mTol_Aff_PhgStdPIF = aTol;
}

void cParamIFDistStdPhgr::SetParam_Dec_Fige(double aTol)
{
   mParamDecentreFree = false;
   mTol_Dec_PhgStdPIF = aTol;
}

void cParamIFDistStdPhgr::SetParam_Aff_Free()
{
   mParamAffineFree = true;
}

void cParamIFDistStdPhgr::SetParam_Dec_Free()
{
   mParamDecentreFree = true;
}



std::string  cParamIFDistStdPhgr::NameType() const
{
    return cParamIFDistRadiale::NameType() + "APFraser";
}

cDistModStdPhpgr & cParamIFDistStdPhgr::CurDist()
{
	return mDStd;
}
const cDistModStdPhpgr & cParamIFDistStdPhgr::CurDist() const
{
	return mDStd;
}

cMultiContEQF  cParamIFDistStdPhgr::StdContraintes()
{
  cMultiContEQF aRes = cParamIFDistRadiale::StdContraintes();
  double aR =  mDRF.RMax();


  if (! mParamDecentreFree)
  {
      AddFoncRappInit(aRes,NbInc()-4,NbInc()-2,mTol_Dec_PhgStdPIF/ElSquare(aR));
  }
  if (! mParamAffineFree)
  {
      AddFoncRappInit(aRes,NbInc()-2,NbInc(),mTol_Aff_PhgStdPIF/aR);
  }
  return aRes;
}

void cParamIFDistStdPhgr::NV_UpdateCurPIF()
{
    mDRF.ActuRMax(CamInit(),false);
    mCurPIF =  new cCamStenopeModStdPhpgr (DistIsC2M(), CurFocale(),CurPP(), mDStd,CurAFoc());

/*
std::cout    << & (mDStd.Coeff(0)) << " "
             << &(mDStd.P1())   << " "
             << "PHGRCPIF " << mCurPIF->DModPhgrStd().Coeff(0) << "\n";
mSet.ShowVar(); 
getchar();
*/

    // mCurPIF->SetDistInverse();
    mCurPIF->HeritComplAndSz(*CamInit());
}
void cParamIFDistStdPhgr::UpdateCurPIF() { return NV_UpdateCurPIF(); }

CamStenope * cParamIFDistStdPhgr::CurPIF() {return CurPIFStdPhgr();}
cCamStenopeModStdPhpgr *  cParamIFDistStdPhgr::CurPIFStdPhgr() {return mCurPIF;}

CamStenope * cParamIFDistStdPhgr::DupCurPIF() 
{ 
 // std::cout << "cParamIFDistStdPhgr::DupCurPIF  " << mDStd.P1() << " " << mDStd.P2() << " " << mDStd.b1() << " " << mDStd.b2() << "\n";
    cCamStenopeModStdPhpgr * aS = mCurPIF;
    NV_UpdateCurPIF();
    ElSwap(aS,mCurPIF);

/*
 mSet.ShowVar(); 
 std::cout << "cParamIFDistStdPhgr::DupCurPIF  " << mDStd.P1() << " " << mDStd.P2() << " " << mDStd.b1() << " " << mDStd.b2() <<  " " <<  aS->Focale() << "\n";
getchar();
*/

    return aS;
}



Pt2d<Fonc_Num> cParamIFDistStdPhgr::VirtualDist(Pt2d<Fonc_Num> aPF,bool UsePC,int aKCam)
{
   Pt2d<Fonc_Num> fPRad = cParamIFDistRadiale::VirtualDist(aPF,UsePC,aKCam);
   Pt2d<Fonc_Num> fDp = aPF - mDRF.FCentre();

   Fonc_Num fDx = fDp.x;
   Fonc_Num fDy = fDp.y;
   Fonc_Num fDx2 = Square(fDx);
   Fonc_Num fDy2 = Square(fDy);
   Fonc_Num fDR2 = fDx2 + fDy2;
   
   Fonc_Num fDCx = (2*fDx2+fDR2)*mFP1 + 2*fDx*fDy*mFP2 + mFb1*fDx + mFb2 * fDy;
   Fonc_Num fDCy = (2*fDy2+fDR2)*mFP2 + 2*fDx*fDy*mFP1;


   return fPRad + Pt2d<Fonc_Num>(fDCx,fDCy);
}


/************************************************************/
/*                                                          */
/*                cParamIFHomogr                            */
/*                                                          */
/************************************************************/

cParamIFHomogr::cParamIFHomogr
(
     bool                    isDC2M,
     cCamStenopeDistHomogr * aCam,
     cSetEqFormelles &        aSet,
     eModeContrHom            aMode
)   :
    cParamIntrinsequeFormel(isDC2M,aCam,aSet,false),
    mHF (aSet.NewHomF(aCam->Hom(),aMode)),
    mCurPIF (0)
{
    NV_UpdateCurPIF();
}

void cParamIFHomogr::SetFigee() { mHF->SetModeCtrl(eHomFigee);}
void cParamIFHomogr::SetLibre() { mHF->SetModeCtrl(eHomLibre);}
void cParamIFHomogr::SetStdBloqueRot() { mHF->SetModeCtrl(eHomStdBloqueRot);}

cParamIFHomogr::~cParamIFHomogr()
{
	delete mHF;
}

cMultiContEQF  cParamIFHomogr::StdContraintes()
{
    return mHF->StdContraintes();
}

std::string  cParamIFHomogr::NameType() const
{
    return "DHom";
}

Pt2d<Fonc_Num> cParamIFHomogr::VirtualDist(Pt2d<Fonc_Num> aP,bool UsePC,int aKCam)
{
	return (*mHF)(aP);
}



void cParamIFHomogr::NV_UpdateCurPIF()
{
    mCurPIF =  new cCamStenopeDistHomogr(DistIsC2M(),CurFocale(),CurPP(), mHF->HomCur(),CurAFoc());

    // mCurPIF->SetDistInverse();
    mCurPIF->HeritComplAndSz(*CamInit());
}
void cParamIFHomogr::UpdateCurPIF(){NV_UpdateCurPIF();}

CamStenope * cParamIFHomogr::CurPIF() {return CurPIFHom();}
cCamStenopeDistHomogr *  cParamIFHomogr::CurPIFHom() {return mCurPIF;}


CamStenope * cParamIFHomogr::DupCurPIF() 
{ 
    cCamStenopeDistHomogr * aS = mCurPIF;
    NV_UpdateCurPIF();
    ElSwap(aS,mCurPIF);
    return aS;
}

#if (0)
#endif


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
