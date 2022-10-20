
#include "MMVII_Tpl_Images.h"
#include "MMVII_AimeTieP.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ImageInfoExtract.h"


namespace MMVII
{

/** Map the interval [-1,1] to itself, the parameter :
    * Steep fix the derivate in 0
    * Exp fix the power and hence concavity/convexity

*/

class cConcavexMapP1M1 : public cFctrRR
{
   public :
      /// non virtual  F
      double NVF (double) const; 
      ///  virtual for heriting cFctrRR , 
      virtual  double F (double) const override ;
      void Show() const;
      // static cFctrRR  TheOne;  ///< the object return always 1

      cConcavexMapP1M1(double aSteep,double aExp,bool Shift0Is0);
    private :
      double mShift;
      double mFact;
      double mSteep;
      double mExp;
};

static const cTabulFonc1D & AllocTabConCavex(double aSteep,double aExp,int aNbDig)
{
   cPt3dr aParam(aSteep,aExp,aNbDig);
   static int aCpt=0;
   static cPt3dr aLastParam(-1,-1,-1);
   static cTabulFonc1D * aRes = nullptr;
   if ((aRes == nullptr) || (aParam!=aLastParam))
   {
      cMemManager::SetActiveMemoryCount(false);  // Will not unaloccate Res ...

      aCpt++;
      if (aCpt==100)
      {
           cMMVII_Appli::CurrentAppli().MMVII_WARNING("Susicious many alloc in AllocTabConCavex");
      }
      aLastParam = aParam;
      delete aRes;
      cConcavexMapP1M1 aCM(aSteep,aExp,(aSteep<0));
      aRes = new cTabulFonc1D (aCM,-1,1,5000);
      
      cMemManager::SetActiveMemoryCount(true);
   }
   return *aRes;

/*
    typedef std::map<cPt3dr,cTabulFonc1D *>  tMap;
    cMemManager::SetActiveMemoryCount(false);
    // static std::map<cTabulFonc1D,cPt3dr> aBufRes;
    static tMap aMap;
                // static cConcavexMapP1M1 aCM(1e9,0.5,true);

                // static cConcavexMapP1M1 aCM(10,0.5);
                // static cTabulFonc1D aTF(aCM,-1,1,5000);
    cMemManager::SetActiveMemoryCount(true);

    return  *(aMap[cPt3dr(0.0,0.0,0.0)]);
*/
}


double cConcavexMapP1M1::F (double aX) const {return NVF(aX);}

double cConcavexMapP1M1::NVF (double aX) const 
{
    if (aX<0) return -NVF(-aX);
    return mFact * (pow(aX+mShift,mExp) - pow(mShift,mExp));
}

cConcavexMapP1M1::cConcavexMapP1M1(double aSteep,double aExp,bool Shift0Is0) :
    mSteep (aSteep),
    mExp   (aExp)
{
    if (Shift0Is0)
    {
       mFact = 1.0;
       mShift = 0.0;
    }
    else
    {
       MMVII_INTERNAL_ASSERT_medium((mExp!=1)||(mSteep==1),"cConcavexMapP1M1");
       MMVII_INTERNAL_ASSERT_medium((mExp>=1)||(mSteep>1),"cConcavexMapP1M1");
       MMVII_INTERNAL_ASSERT_medium((mExp<=1)||(mSteep<1),"cConcavexMapP1M1");


       {
           mFact = 1;
           for (int aK= 0 ; aK<20 ; aK++)
           {
               mShift = pow(mSteep /(mFact*mExp),1/(mExp-1));
               mFact = 1/(pow(1+mShift,mExp) - pow(mShift,mExp));
               // StdOut() << "cConcavexMapP1M1::  SHF" << mShift << " FACT " << mFact << "\n";
           }
           // Show();
       }
    }
}

void cConcavexMapP1M1::Show() const
{
   double aEps= 1e-4;
   StdOut()  << " STEEP0 " << mSteep << " EXP " << mExp << "\n";
   StdOut()  << " V0 " << NVF(0) 
             << " V1 " << NVF(-1) 
             << " VM1 " << NVF(-1) 
             << " D0 " << (NVF(aEps)-NVF(-aEps)) / (2*aEps) 
             << " D1 " << (NVF(1+aEps)-NVF(1-aEps)) / (2*aEps) 
             << " DM1 " << (NVF(-1+aEps)-NVF(-1-aEps)) / (2*aEps) 
             << "\n";
   for (int aK=0 ; aK<=10 ; aK++)
   {
        double aV = aK/10.0;
        StdOut()  << "  * " << aV << " => " << NVF(aV) << "\n";
   }
   getchar();
}

void TTT()
{
    {
       cConcavexMapP1M1 aCM(5.0,0.5,false);
       aCM.Show();
    }
    {
       cConcavexMapP1M1 aCM(10.0,0.5,false);
       aCM.Show();
    }
    exit(EXIT_SUCCESS);
    // cConcavexMapP1M1(10.0,0.5);
    // cConcavexMapP1M1(2.0,0.5);
    // cConcavexMapP1M1(0.5,2);
    // cConcavexMapP1M1(1.0,1.0);
}

/* ================================= */
/*          cAimePCar                */
/* ================================= */
cAimeDescriptor::cAimeDescriptor() :
   mILP (cPt2di(1,1))
{
}

cAimeDescriptor cAimeDescriptor::DupLPIm() const
{
   cAimeDescriptor aRes(*this);
   aRes.mILP = mILP.Dup();
   return aRes;
}

cIm2D<tU_INT1>   cAimeDescriptor::ILP() const {return mILP;}
const std::vector<double> &  cAimeDescriptor::DirPrinc() const {return mDirPrinc;}
std::vector<double> &  cAimeDescriptor::DirPrinc() {return mDirPrinc;}

double  cAimeDescriptor::DistanceFromIShift(const cAimeDescriptor & aAD2,int aShift,const cSetAimePCAR & aSet) const
{
   const cDataIm2D<tU_INT1>  & aDIm1(mILP.DIm());
   cPt2di aSz1= aDIm1.Sz();
   tU_INT1 * const * aData1 = aDIm1.ExtractRawData2D();

   const cDataIm2D<tU_INT1>  & aDIm2(aAD2.mILP.DIm());
   // === cPt2di aSz2= aDIm2.Sz();
   tU_INT1 * const * aData2 = aDIm2.ExtractRawData2D();

   MMVII_INTERNAL_ASSERT_tiny(aSz1==aDIm2.Sz(),"cAimeDescriptor::Distance");
   int aNbX = aSz1.x();
   aShift = mod(aShift,aNbX);
   int aIRes = 0;

   for (int aY=0 ; aY<aSz1.y() ; aY++)
   {
       const tU_INT1 * aLine1 = aData1[aY];
       const tU_INT1 * aLine2 = aData2[aY];
       for (int aX1= 0; aX1<aNbX ; aX1++)
       {
           aIRes += Square(aLine1[aX1]-aLine2[(aX1+aShift)%aNbX]);
       }
   
   } 
   double aRes = aIRes;
   aRes /= aSz1.x() * aSz1.y();
   aRes /= Square(aSet.Ampl2N());

   return aRes;
}

double  cAimeDescriptor::DistanceFrom2RPeek(double aX1,const cAimeDescriptor & aAD2,double aX2,const cSetAimePCAR & aSet) const
{
    return DistanceFromIShift(aAD2,round_ni(aX2-aX1),aSet);
}

double cAimeDescriptor::DistanceFromStdPeek(int aIPeek,const cAimeDescriptor & aAD2,const cSetAimePCAR & aSet) const
{
    return DistanceFrom2RPeek(mDirPrinc.at(aIPeek),aAD2,aAD2.mDirPrinc.at(aIPeek),aSet);
}


cWhitchMin<int,double>  cAimeDescriptor::DistanceFromBestPeek(const cAimeDescriptor & aAD2,const cSetAimePCAR & aSet) const
{
    cWhitchMin<int,double> aWMin(-1,1e60);
    for (int aK=0 ; aK<int(mDirPrinc.size()) ; aK++)
    {
        aWMin.Add(aK,DistanceFromStdPeek(aK,aAD2,aSet));
    }
    return aWMin; 
}



void AddData(const cAuxAr2007 & anAux,cAimeDescriptor & aDesc)
{
   AddData(cAuxAr2007("ILP",anAux)  , aDesc.ILP().DIm());
   AddData(cAuxAr2007("Dirs",anAux) , aDesc.DirPrinc());
}



/* ================================= */
/*          cAimePCar                */
/* ================================= */


cAimeDescriptor & cAimePCar::Desc() {return mDesc; }
const cAimeDescriptor & cAimePCar::Desc() const {return mDesc; }
cPt2dr&           cAimePCar::Pt()   {return mPt;}
const cPt2dr&     cAimePCar::Pt() const  {return mPt;}
cPt2dr&           cAimePCar::PtIm()   {return mPtIm;}
const cPt2dr&     cAimePCar::PtIm() const  {return mPtIm;}

void AddData(const cAuxAr2007 & anAux,cAimePCar & aPC)
{
     AddData(cAuxAr2007("Pt",anAux),aPC.Pt());
     AddData(cAuxAr2007("Desc",anAux),aPC.Desc());
}

cAimePCar cAimePCar::DupLPIm() const
{
   cAimePCar aRes;
   aRes.mPt = mPt;
   aRes.mPtIm = mPtIm;
   aRes.mDesc = mDesc.DupLPIm();
   return aRes;
}

double  cAimePCar::L1Dist(const cAimePCar& aP2) const
{
   return mDesc.ILP().DIm().L1Dist(aP2.mDesc.ILP().DIm());
}


/* ================================= */
/*          cProtoAimeTieP           */
/* ================================= */

template<class Type> 
   cProtoAimeTieP<Type>::cProtoAimeTieP
   (
        cGP_OneImage<Type> * aGPI,
        const cPt2di & aPInit,
        bool           ChgMaj
   ) :
   mGPI        (aGPI),
   mChgMaj     (ChgMaj),
   mPImInit    (aPInit),
   mPRImRefined   (ToR(mPImInit)),
   mPFileInit  (mGPI->Im2File(ToR(mPImInit))),
   mNumAPC     (-1)
{
}

template<class Type> 
   cProtoAimeTieP<Type>::cProtoAimeTieP
   (
        cGP_OneImage<Type> * aGPI,
        const cPt2dr & aPInit
   ) :
   mGPI          (aGPI),
   mChgMaj       (false),
   mPImInit      (ToI(aPInit)),
   mPRImRefined  (aPInit),
   mPFileInit    (mGPI->Im2File(aPInit)),
   mPFileRefined (mPFileInit),
   mNumAPC       (-1)
{
}



template<class Type> int   cProtoAimeTieP<Type>::NumOct()   const {return mGPI->Oct()->NumInPyr();}
template<class Type> int   cProtoAimeTieP<Type>::NumIm()    const {return mGPI->NumInOct();}
template<class Type> float cProtoAimeTieP<Type>::ScaleInO() const {return mGPI->ScaleInO();}
template<class Type> float cProtoAimeTieP<Type>::ScaleAbs() const {return mGPI->ScaleAbs();}
template<class Type>  const cGaussianPyramid<Type> & cProtoAimeTieP<Type>::Pyram() const {return mGPI->Pyr();}
template<class Type>  const cGP_Params&  cProtoAimeTieP<Type>::Params() const {return Pyram().Params();}

///  ALP  Aime Log Pol

double CalcOrient(const cDataIm2D<tREAL4>&  aDIm,eModeNormOr aMode)
{
    // cDataIm2D<tREAL4> & aDIm(aIm.DIm());
    cPt2di aSz = aDIm.Sz();
    int aNbRho  = (aMode==eModeNormOr::eMNO_MaxGradR) ? (aSz.y()-1) : aSz.y();
    int aNbTeta  = aSz.x();

    cIm1D<tREAL4>  aHisto(aNbTeta,nullptr,eModeInitImage::eMIA_Null);
    cDataIm1D<tREAL4> & aDHisto(aHisto.DIm());
    // Parse the image and fill the histogramme in teta
    for (const auto & aP : cRect2(cPt2di(0,0),cPt2di(aNbTeta,aNbRho)))
    {
        double aVal = aDIm.GetV(aP);
        switch(aMode)
        {
             case eModeNormOr::eMNO_MaxGray :
             break;
             // tangential gradient, diff along teta
             case eModeNormOr::eMNO_MaxGradT :
                 aVal -= aDIm.GetV(cPt2di((aP.x()+1)%aNbTeta,aP.y()));
             break;
             // radial  gradient, diff along teta
             case eModeNormOr::eMNO_MaxGradR :
                 aVal -= aDIm.GetV(cPt2di(aP.x(),aP.y()+1));
             break;
             default :
                  MMVII_INTERNAL_ASSERT_strong(false,"Unhandled enum val in eModeNormOr");
             break;
        }
        aDHisto.AddV(aP.x(),std::abs(aVal));
    }

    // Get integer max value
    int aIRes = WhitchMax(aDHisto).x();
    // refine it by parabol fitting
    double aDelta =  StableInterpoleExtr
                     (
                         aDHisto.CircGetV(aIRes-1),
                         aDHisto.CircGetV(aIRes  ),
                         aDHisto.CircGetV(aIRes+1)
                     );
   

    // Ensure it is in [0, NbTeta[
    double aRes =  mod_real(aIRes + aDelta,aNbTeta);
    return aRes;
}

template<class Type> bool   cProtoAimeTieP<Type>::FillAPC(const cFilterPCar& aFPC,cAimePCar & aPC,bool ForTest)
{
// static int aCpt=0; aCpt++;  StdOut() << "BUG cProtoAimeTieP " << aCpt << " " << ForTest << "\n";
// bool Bug=(aCpt==65);

   int aNbTeta = aFPC.LPS_NbTeta();
   int aNbRho = aFPC.LPS_NbRho();
   double aMulV = aFPC.LPS_Mult();
   bool   aCensusMode = aFPC.LPS_CensusMode();
    
   double  aRho0 = aFPC.LPC_Rho0();
   int     aShiftI0 = aFPC.LPC_DeltaI0();
   int     aDeltaIm = aFPC.LPC_DeltaIm();



   tGPI * aImCarac = mGPI;
   tGPI * aImDetect   = aImCarac->ImOriHom();  ///< Image original with same level of detection
   const std::vector<tGPI*> &   aVIm =  aImDetect->Pyr().VMajIm() ; ///< Vector of Major Images

   int aK0 = aImDetect->NumMaj() + aShiftI0; ///< We may wish to have more resolved images (or less ?)
   MMVII_INTERNAL_ASSERT_medium(aK0>=0,"Num Maj assertion in FillAPC");


   int aKLim = aK0+ (aNbRho -1) * aDeltaIm; ///< K of lowest resolved image

   if (aKLim >= int(aVIm.size())) // If out of the pyramid
      return false;
    
   cPt2dr aCenter =  mPFileRefined;  // Center 
   cPt2di aSzLP(aNbTeta,aNbRho); // Sz of Log Pol image
   const std::vector<cPt2dr> & aVDirTeta0 = aFPC.VDirTeta0();   // vector of direction
   const std::vector<cPt2dr> & aVDirTeta1 = aFPC.VDirTeta1();   // other vector of direction, may be interlaced

   cIm2D<tREAL4>       aILPr(aSzLP);  // Real Log Pol images
   cDataIm2D<tREAL4>&  aDILPr = aILPr.DIm(); // Data real log pol im

   int IndRhoLP=0;

   cComputeStdDev<double> aRawStat; // To normalize value
   double aVCentral=0.0;  // To be  used in census mode

   for (int aKIm=aK0 ; aKIm<=aKLim  ; aKIm+= aDeltaIm)
   {
        tGPI &  anIk  =  *(aVIm.at(aKIm));  // Image at corresponding level
        cPt2dr  aCk   =  anIk.File2Im(aCenter);  // Centre in geometry of image
        double  aRhok =  aRho0 * anIk.ScaleInO();  // Rho in fact R0 * ScaleAbs / ScaleOfOct
        cDataIm2D<Type> &  aDImk (anIk.ImG().DIm());  // Data Image at level

        if (ForTest)
        {
             // Check if all corner are inside
             for (int aK=0 ; aK<4 ; aK++)
             {
                 cPt2dr aP = aCk + ToR(TAB4Corner[aK]) * aRhok;
                 if (! aDImk.InsideBL(aP))
                 {
                    return false;
                 }
             }
        }
        // Memorize central value 4 census
        if (aCensusMode && (aKIm==aK0))
        {
             aVCentral = aDImk.GetVBL(aCk);
             if (ForTest && (aVCentral==0))
             {
                return false;
             }
        }

        if (!ForTest)
        {
             for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++)
             {
                 const cPt2dr & aDir = (aKIm%2) ? aVDirTeta1.at(aKTeta) : aVDirTeta0.at(aKTeta);
                 cPt2dr aP = aCk + aDir * aRhok; // Point in LogPol
                 double aV = aDImk.GetVBL(aP);
                 aDILPr.SetV(cPt2di(aKTeta,IndRhoLP),aV);
                 if (! aCensusMode)
                 {
                    aRawStat.Add(1.0,aV);
                 }
             }
        }
        IndRhoLP++;
// StdOut() << "RoohhhhK " << aRhok << " ABS=" << anIk.ScaleAbs()<< "\n";
   }
// StdOut() << "xxxxxRoohhhhK NBT=" << aNbTeta << " \n"; getchar();
   // Now, in test mode, we know that all the circle will be inside, OK then ...
   if (ForTest)
   {
       return true;
   }

   // Compute the main orientations from real image
   if (aFPC.IsForTieP())
   {
      for (int aK=0 ; aK<int(eModeNormOr::eNbVals) ; aK++)
      {
         aPC.Desc().DirPrinc().push_back(CalcOrient(aDILPr,eModeNormOr(aK))) ;
      }
   }
  
   // Memorize the localization
   aPC.Pt() = aCenter;
   aPC.PtIm() = mPRImRefined;
 
   //  Now convert the image to a 8 bit unsigned one
   cDataIm2D<tU_INT1> & aDILPi =  aPC.Desc().ILP().DIm();
   aDILPi.Resize(aSzLP);
   cComputeStdDev<double> aStat = aCensusMode ? aRawStat :  aRawStat.Normalize();

   const cTabulFonc1D &  aTFD  = AllocTabConCavex(aFPC.LPQ_Steep0(),aFPC.LPQ_Exp(),5000);
   for (const auto & aP : aDILPi)
   {
        double aV0 = aDILPr.GetV(aP);
        double aValRes=0;
        if (aCensusMode)
        {
            // double aVN  = aV0/aVCentral;
            double aRatio =  NormalisedRatioPos(aV0,aVCentral);
            aRatio = aTFD.F(aRatio);
            aValRes = 128 + cSetAimePCAR::TheCensusMult * aRatio;
        }
        else
        {
            double aVN  = aStat.NormalizedVal(aV0);
            aValRes = 128 + aMulV  * aVN;
        }
        aDILPi.SetVTrunc(aP,aValRes);
   }

   return true;
}

template<class Type> bool cProtoAimeTieP<Type>::TestFillAPC(const cFilterPCar& aFPC)
{
    cAimePCar  aAPC;
    bool aRes =  FillAPC(aFPC,aAPC,true);

    return aRes;
}

/* ================================= */
/*          cSetAimePCAR             */
/* ================================= */

const double cSetAimePCAR::TheCensusMult = 128.0;

cSetAimePCAR::cSetAimePCAR(eTyPyrTieP aType,bool IsMax) :
  mType  ((int)aType),
  mIsMax (IsMax)
{
}

cSetAimePCAR::cSetAimePCAR():
   cSetAimePCAR(eTyPyrTieP::eNbVals,true)
{
}


eTyPyrTieP              cSetAimePCAR::Type()   {return eTyPyrTieP(mType);}
int &                   cSetAimePCAR::IType()  {return mType;}
bool&                   cSetAimePCAR::IsMax()  {return mIsMax;}
std::vector<cAimePCar>& cSetAimePCAR::VPC()    {return mVPC;}
bool&          cSetAimePCAR::Census()       {return mCensus;}
const bool&    cSetAimePCAR::Census() const {return mCensus;}
double&        cSetAimePCAR::Ampl2N()       {return mAmpl2N;}
const double&  cSetAimePCAR::Ampl2N() const {return mAmpl2N;}


void AddData(const cAuxAr2007 & anAux,cSetAimePCAR & aSPC)
{
    AddData(cAuxAr2007("Type",anAux),aSPC.IType());
    AddData(cAuxAr2007("Max",anAux),aSPC.IsMax());
    AddData(cAuxAr2007("VPC",anAux),aSPC.VPC() );
    AddData(cAuxAr2007("Census",anAux),aSPC.Census() );
    AddData(cAuxAr2007("Ampl2N",anAux),aSPC.Ampl2N() );
}

void cSetAimePCAR::InitFromFile(const std::string & aName) 
{
    ReadFromFile(*this,aName);
}


void cSetAimePCAR::SaveInFile(const std::string & aName) const
{
 //  StdOut() << "MMPPPDD " << aName << " "  << (const_cast<cSetAimePCAR *>(this))->VPC().size()  << "\n";
     MMVII::SaveInFile(*this,aName);

     if (0)
     {
          for (int aK=0; aK<100 ; aK++) 
              StdOut() << "MMv1_SaveInFile\n";
          MMv1_SaveInFile<cSetAimePCAR>(*this,aName);  // generate an error as "it should"
     }
}



/*  ====  INSTANCIATION  ======= */

template class cProtoAimeTieP<tREAL4>;
template class cProtoAimeTieP<tINT2>;



};
