#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{


/* ================================= */
/*          cAimePCar                */
/* ================================= */
cAimeDescriptor::cAimeDescriptor() :
   mILP (cPt2di(1,1))
{
}

cIm2D<tU_INT1>   cAimeDescriptor::ILP() {return mILP;}
const std::vector<double> &  cAimeDescriptor::DirPrinc() const {return mDirPrinc;}
std::vector<double> &  cAimeDescriptor::DirPrinc() {return mDirPrinc;}

double  cAimeDescriptor::DistanceFromIShift(const cAimeDescriptor & aAD2,int aShift,const cSetAimePCAR & aSet) const
{
   const cDataIm2D<tU_INT1>  & aDIm1(mILP.DIm());
   cPt2di aSz1= aDIm1.Sz();
   tU_INT1 * const * aData1 = aDIm1.ExtractRawData2D();

   const cDataIm2D<tU_INT1>  & aDIm2(aAD2.mILP.DIm());
   cPt2di aSz2= aDIm2.Sz();
   tU_INT1 * const * aData2 = aDIm2.ExtractRawData2D();

   MMVII_INTERNAL_ASSERT_tiny(aSz1==aSz2,"cAimeDescriptor::Distance");
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
cPt2dr&           cAimePCar::Pt()   {return mPt;}

void AddData(const cAuxAr2007 & anAux,cAimePCar & aPC)
{
     AddData(cAuxAr2007("Pt",anAux),aPC.Pt());
     AddData(cAuxAr2007("Desc",anAux),aPC.Desc());
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
   mPFileInit  (mGPI->Im2File(ToR(mPImInit))),
   mNumAPC     (-1)
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
   std::vector<cPt2dr> aVDirTeta;   // vector of direction
   for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++)
   {
       aVDirTeta.push_back(FromPolar(1.0,(2*M_PI*aKTeta)/aNbTeta));
   }
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

        // Memorize central value 4 census
        if (aCensusMode && (aKIm==aK0))
        {
             aVCentral = aDImk.GetVBL(aCk);
             if (ForTest && (aVCentral==0))
             {
                return false;
             }
        }

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
        else
        {
             for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++)
             {
                 cPt2dr aP = aCk + aVDirTeta.at(aKTeta) * aRhok; // Point in LogPol
                 double aV = aDImk.GetVBL(aP);
                 aDILPr.SetV(cPt2di(aKTeta,IndRhoLP),aV);
                 if (! aCensusMode)
                 {
                    aRawStat.Add(1.0,aV);
                 }
             }
        }
        IndRhoLP++;
   }
   // Now, in test mode, we now that all the circle will be inside, OK then ...
   if (ForTest)
   {
       return true;
   }

   // Compute the main orientations from real image
   for (int aK=0 ; aK<int(eModeNormOr::eNbVals) ; aK++)
   {
      aPC.Desc().DirPrinc().push_back(CalcOrient(aDILPr,eModeNormOr(aK))) ;
   }
  
   // Memorize the localization
   aPC.Pt() = aCenter;
 
   //  Now convert the image to a 8 bit unsigned one
   cDataIm2D<tU_INT1> & aDILPi =  aPC.Desc().ILP().DIm();
   aDILPi.Resize(aSzLP);
   cComputeStdDev<double> aStat = aCensusMode ? aRawStat :  aRawStat.Normalize();

   for (const auto & aP : aDILPi)
   {
        double aV0 = aDILPr.GetV(aP);
        double aValRes=0;
        if (aCensusMode)
        {
            // double aVN  = aV0/aVCentral;
            aValRes = 128 + cSetAimePCAR::TheCensusMult * NormalisedRatio(aV0,aVCentral);
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
