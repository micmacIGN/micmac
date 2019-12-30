#include "include/V1VII.h"
#include "cGeneratedCodeTestCam.h"
#include "include/MMVII_Derivatives.h"


namespace MMVII
{
/** \file GenCodeTestCam.cpp
    \brief Make benchmark on generated formal code 

    This file contain test to generate formal class, and use them
  to make benchmark between jets and computed analyticall

*/


class cGenCodeTestCam
{
    public :
       cGenCodeTestCam();
       void Generate();
       void Init(cGeneratedCodeTestCam &aGCTC);

    private :
       cIncListInterv mLInterv;
       cSetEqFormelles     mSet;
       AllocateurDInconnues &  mAlloc;
       cMatr_Etat_PhgrF    mRot0;
       cP2d_Etat_PhgrF     mMesIm;

       cP3dFormel          mPTer;
       cP3dFormel          mCCam;
       cP3dFormel          mOmega;
       cP2dFormel          mCDist;
       cValFormel          mK2;
       cValFormel          mK4;
       cValFormel          mK6;
       cP2dFormel          mPP;
       cValFormel          mFoc;
};

class cMMV1_TestCam : public cInterfaceTestCam
{
    public :
       void  InitFromParams(const std::vector<double> &) override;
       void  Compute(std::vector<double> & Vals,std::vector<std::vector<double> > & ) override;
       void  Compute(int aNb) override;
       cMMV1_TestCam ();
    private :
       cGenCodeTestCam        mGenerator;
       cGeneratedCodeTestCam  mCamGen;
};

/* *********************************************** */
/*                                                 */
/*            cMMV1_TestCam                        */
/*                                                 */
/* *********************************************** */

void  cMMV1_TestCam::InitFromParams(const std::vector<double> & aVals)
{
   mCamGen.SetCoordCur(aVals.data());
// MesIm_x
   *(mCamGen.AdrVarLocFromString("MesIm_x")) = 0.0;
   *(mCamGen.AdrVarLocFromString("MesIm_y")) = 0.0;

   *(mCamGen.AdrVarLocFromString("R0_0_0")) = 1.0;
   *(mCamGen.AdrVarLocFromString("R0_1_0")) = 0.0;
   *(mCamGen.AdrVarLocFromString("R0_2_0")) = 0.0;

   *(mCamGen.AdrVarLocFromString("R0_0_1")) = 0.0;
   *(mCamGen.AdrVarLocFromString("R0_1_1")) = 1.0;
   *(mCamGen.AdrVarLocFromString("R0_2_1")) = 0.0;

   *(mCamGen.AdrVarLocFromString("R0_0_2")) = 0.0;
   *(mCamGen.AdrVarLocFromString("R0_1_2")) = 0.0;
   *(mCamGen.AdrVarLocFromString("R0_2_2")) = 1.0;

} 

cMMV1_TestCam::cMMV1_TestCam ()
{
     mGenerator.Init(mCamGen);
}

void  cMMV1_TestCam::Compute(std::vector<double> & aVals,std::vector<std::vector<double> > &  aDeriv)
{
   mCamGen.ComputeValDeriv();
   aVals  = mCamGen.ValSsVerif();
   aDeriv = mCamGen.CompDerSsVerif();
}

void  cMMV1_TestCam::Compute(int aNb)
{
    for (int aK=0 ; aK<aNb ; aK++)
        mCamGen.ComputeValDeriv();
}

cInterfaceTestCam * cInterfaceTestCam::AllocMMV1()
{
    return new cMMV1_TestCam;
}

/*
       cGenCodeTestCam  aGen;
       cGeneratedCodeTestCam aCamGen;
       aGen.Init(aCamGen);
       aCamGen.SetCoordCur(aVVals);
       aCamGen.ComputeValDeriv();

*/



/* *********************************************** */
/*                                                 */
/*            cGenCodeTestCam                      */
/*                                                 */
/* *********************************************** */

cGenCodeTestCam::cGenCodeTestCam() :
   mSet     (),
   mAlloc   (mSet.Alloc()),
   mRot0    ("R0",3,3),
   mMesIm   ("MesIm"),
   mPTer    (Pt3dr(0,0,1),"PGr",mSet,mLInterv),
   mCCam    (Pt3dr(0,0,0),"CCam",mSet,mLInterv),
   mOmega   (Pt3dr(0,0,0),"W",mSet,mLInterv),
   mCDist   (Pt2dr(0.01,0.02),"CD",mSet,mLInterv),
   mK2      ( 0.002,"K2",mSet,mLInterv),
   mK4      (-0.001,"K4",mSet,mLInterv),
   mK6      ( 0.001,"K6",mSet,mLInterv),
   mPP      (Pt2dr(3000,2000),"PP",mSet,mLInterv),
   mFoc     ( 5000,"F",mSet,mLInterv)
{
}

void cGenCodeTestCam::Init(cGeneratedCodeTestCam &aGCTC)
{
   aGCTC.SetMappingCur(mLInterv,&mSet);
   // aGCTC.Close(false);
}

void cGenCodeTestCam::Generate() 
{
    // mFPTer.IncInterv().SetName("Ori1");
   // En coordonnees dans le repere de la camera, en incluant une "petite" rotation
    Pt3d<Fonc_Num> aPCam = mRot0.Mat() * (mPTer.FPt()-mCCam.FPt());
    aPCam = aPCam + (mOmega.FPt() ^ aPCam);  

    Pt2d<Fonc_Num> aPi (aPCam.x/aPCam.z,aPCam.y/aPCam.z);
    Pt2d<Fonc_Num>  aCdPi = aPi-mCDist.FPt();
    Fonc_Num  aRho2 = Square(aCdPi.x) + Square(aCdPi.y);
    Fonc_Num  aRho4 = Square(aRho2);
    Fonc_Num  aRho6 = Cube(aRho2);

    Fonc_Num  aCoeffDist = mK2.FVal()*aRho2 + mK4.FVal()*aRho4 + mK6.FVal()*aRho6;
    Pt2d<Fonc_Num>  aPDist = aPi + Pt2d<Fonc_Num>(aCdPi.x*aCoeffDist,aCdPi.y*aCoeffDist);
    Pt2d<Fonc_Num> aPProj = mPP.FPt() + Pt2d<Fonc_Num>(aPDist.x*mFoc.FVal(),aPDist.y*mFoc.FVal());
    Pt2d<Fonc_Num> aResidu = aPProj-mMesIm.PtF();

    std::string aDirApp =  cMMVII_Appli::CurrentAppli().TopDirMMVII() ;
    cElCompileFN::DoEverything
    (
             aDirApp + "src/MMV1/",   // Directory ou est localise le code genere
             "cGeneratedCodeTestCam",  // donne les noms de .cpp et .h  de classe
             // std::vector<Fonc_Num> ({aPi.x,aPi.y}), //  expressions formelles 
             std::vector<Fonc_Num> ({aResidu.x,aResidu.y}), //  expressions formelles 
             mLInterv  // intervalle de reference
    );


}


void   MMV1_GenerateCodeTestCam()
{
    if (1) 
    {
       cGenCodeTestCam  aGCTC;
       aGCTC.Generate();
    }
    if (0)
    {
       double   aVVals[30];
       cGenCodeTestCam  aGen;
       cGeneratedCodeTestCam aCamGen;
       aGen.Init(aCamGen);
       aCamGen.SetCoordCur(aVVals);
       aCamGen.ComputeValDeriv();

    }
}



};
