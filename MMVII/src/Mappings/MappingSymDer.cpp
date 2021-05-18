#include "include/MMVII_all.h"

using namespace NS_SymbolicDerivative ;


namespace MMVII
{

/* ============================================= */
/*      cDataMapCalcSymbDer<Type>                */
/* ============================================= */


template <class Type,const int DimIn,const int DimOut> 
  void  cDataMapCalcSymbDer<Type,DimIn,DimOut>::CheckDim(tCalc  * aCalc,bool Derive)
{
   MMVII_INTERNAL_ASSERT_strong(DimIn==aCalc->NbUk(),"Input dim in calculator");
   MMVII_INTERNAL_ASSERT_strong(DimOut==aCalc->NbElem(),"Output dim in calculator");
   MMVII_INTERNAL_ASSERT_strong(aCalc->WithDer()==Derive,"Derive dim in calculator");
}

template <class Type,const int DimIn,const int DimOut> 
  void  cDataMapCalcSymbDer<Type,DimIn,DimOut>::SetObs(const std::vector<Type> & aVObs)
{
   mVObs = aVObs;
}

template <class Type,const int DimIn,const int DimOut> 
  cDataMapCalcSymbDer<Type,DimIn,DimOut>::cDataMapCalcSymbDer
  (
         tCalc  * aCalcVal,
         tCalc  * aCalcDer,
         const std::vector<Type> & aVObs
  ) :
     mCalcVal  (aCalcVal),
     mCalcDer  (aCalcDer),
     mVObs     (aVObs)
{
    CheckDim(mCalcVal,false);
    CheckDim(mCalcDer,true);
    // mCalcVal.NbUk()
}


template <class Type,const int DimIn,const int DimOut> 
  const typename cDataMapCalcSymbDer<Type,DimIn,DimOut>::tVecOut &
      cDataMapCalcSymbDer<Type,DimIn,DimOut>::Values(tVecOut & aRes,const tVecIn & aVecIn) const
{
   aRes.clear();
   std::vector<Type> aVUk(DimIn);

   MMVII_INTERNAL_ASSERT_strong(mCalcVal->NbInBuf()==0,"Buff not empty");
   tU_INT4 aSzBuf = mCalcVal->SzBuf();
   for (tU_INT4 aK0=0 ;aK0<aVecIn.size() ; aK0+=aSzBuf)
   {
       tU_INT4 aK1 = std::min(tU_INT4(aVecIn.size()),aK0+aSzBuf);
       for (tU_INT4 aK=aK0 ; aK<aK1 ; aK++)
       {
           for (int aD=0 ; aD<DimIn ; aD++)
               aVUk[aD] = aVecIn[aK][aD];
           mCalcVal->PushNewEvals(aVUk,mVObs);
       }
       mCalcVal->EvalAndClear();
       for (tU_INT4 aK=aK0 ; aK<aK1 ; aK++)
       {
           tPtOut aPRes;
           for (int aD=0 ; aD<DimOut ; aD++)
           {
               aPRes[aD] = mCalcVal->ValComp(aK-aK0,aD);
           }
           aRes.push_back(aPRes);
       }
   }
   return aRes;
}

template <class Type,const int DimIn,const int DimOut> 
  typename cDataMapCalcSymbDer<Type,DimIn,DimOut>::tCsteResVecJac
      cDataMapCalcSymbDer<Type,DimIn,DimOut>::Jacobian(tResVecJac aRes,const tVecIn & aVecIn) const 
{
   tVecOut * aVecOut=aRes.first;
   tVecJac * aVecJac=aRes.second;

   std::vector<Type> aVUk(DimIn);

   MMVII_INTERNAL_ASSERT_strong(mCalcDer->NbInBuf()==0,"Buff not empty");
   tU_INT4 aSzBuf = mCalcDer->SzBuf();
   for (tU_INT4 aK0=0 ;aK0<aVecIn.size() ; aK0+=aSzBuf)
   {
       tU_INT4 aK1 = std::min(tU_INT4(aVecIn.size()),aK0+aSzBuf);
       for (tU_INT4 aK=aK0 ; aK<aK1 ; aK++)
       {
           for (int aD=0 ; aD<DimIn ; aD++)
               aVUk[aD] = aVecIn[aK][aD];
           mCalcDer->PushNewEvals(aVUk,mVObs);
       }

       mCalcDer->EvalAndClear();
       for (tU_INT4 aK=aK0 ; aK<aK1 ; aK++)
       {
           tPtOut aPRes;
           for (int aDOut=0 ; aDOut<DimOut ; aDOut++)
           {
               tPtIn aLineJac;
               aPRes[aDOut] = mCalcDer->ValComp(aK-aK0,aDOut);
               for (int aDIn=0 ; aDIn<DimIn ; aDIn++)
               {
                  aLineJac[aDIn] = mCalcDer->DerComp(aK-aK0,aDOut,aDIn);
               }
               SetLine(aDOut,(*aVecJac)[aK],aLineJac);
               // SetCol((*aVecJac)[aK],aDOut,aLineJac);
           }
           aVecOut->push_back(aPRes);
       }
   }
   return tCsteResVecJac(aVecOut,aVecJac);
}

template class cDataMapCalcSymbDer<tREAL8,2,2> ;
 // cDataIIMFromMap<Type,Dim>;


/*
template <class Type,const int Dim> class  cDataDistCloseId : cDataIterInvertMapping<Type,Dim>
{
     public :
       cDataDistCloseId(
     private :
};
*/


/* ============================================= */
/*                   TEST                        */
/* ============================================= */

bool BUGINVMAP =false;

void TestJacob(cDataMapCalcSymbDer<double,2,2> * aMCS,const cPt2dr & aP)
{
double aEps= 1e-5;
// cPt2dr aP0 = aMCS->cDataMapping<double,2,2>::Value(aP) ;
cPt2dr aPPx = aMCS->cDataMapping<double,2,2>::Value(aP +cPt2dr(aEps,0.0)) ;
cPt2dr aPmx = aMCS->cDataMapping<double,2,2>::Value(aP +cPt2dr(-aEps,0.0)) ;
cPt2dr aPPy = aMCS->cDataMapping<double,2,2>::Value(aP +cPt2dr(0.0,aEps)) ;
cPt2dr aPmy = aMCS->cDataMapping<double,2,2>::Value(aP +cPt2dr(0.0,-aEps)) ;

double aRho = Norm2(aP);
StdOut() << " GX: " << (aPPx-aPmx)/(2*aEps)
         << " GY: " << (aPPy-aPmy)/(2*aEps)
         << " 7R6 " << 7 * pow(aRho,6)
         << "\n";
}


void BenchSymDerMap(cParamExeBench & aParam)
{
   cPt3di aDeg(3,1,1);
   const std::vector<cDescOneFuncDist>  & aVecD =  DescDist(aDeg);

   for (int aKTest=0 ; aKTest<100 ; aKTest++)
   {
       cCalculator<double> * anEqVal =  EqDist(aDeg,false,1+RandUnif_N(50));
       cCalculator<double> * anEqDer =  EqDist(aDeg, true,1+RandUnif_N(50));
       int aNbPar = anEqVal->NbObs();
       std::vector<double> aVParam(aNbPar,0.0);

       // Basic Test, with all param=0, distortion=identity
       {
           cDataMapCalcSymbDer<double,2,2> aMCS(anEqVal,anEqDer,aVParam);
           cPt2dr aP = cPt2dr::PRandC();
           cPt2dr aQ = aMCS.Value(aP);
           MMVII_INTERNAL_ASSERT_bench(Norm2(aP-aQ)<1e-5,"Value in MCS");
       }

       // compute a distortion with random value
       double aSomJac=0;
       for (int aKPar=0 ; aKPar<aNbPar ;aKPar++)
       {
           double aV = (RandUnif_C() * HeadOrTail()) / aVecD.at(aKPar).MajNormJacOfRho(1.0);
           aVParam[aKPar] = aV;
           aSomJac += std::abs(aV) * aVecD.at(aKPar).MajNormJacOfRho(1.0) ;
       }
       aSomJac = std::max(aSomJac,1e-5) ; // dont divide 0 if all null
       for (int aKPar=0 ; aKPar<aNbPar ;aKPar++)
       {
           aVParam[aKPar] = 0.2* (aVParam[aKPar]/ aSomJac);
           if (false) // Print params
              std::cout << "KPar " << anEqVal->NamesObs().at(aKPar) << " : "  << aVParam[aKPar] << "\n";
       }

       // cDataMapCalcSymbDer<double,2,2> aMCS(anEqVal,anEqDer,aVParam);
       auto aMCS = new cDataMapCalcSymbDer<double,2,2>(anEqVal,anEqDer,aVParam);
       cMapping<double,2,2>            aMapCS(aMCS);

       auto aDId = new cMappingIdentity<double,2> ;
       cMapping<double,2,2>            aMapId(aDId);

       cDataIIMFromMap aIMap(aMapCS,aMapId,1e-6,5);

       int aNbPts  = 1+RandUnif_N(100);
       std::vector<cPt2dr> aVIn;
       for (int aKPts=0 ; aKPts<aNbPts ; aKPts++)
       {
           aVIn.push_back(cPt2dr::PRandInSphere());
       }
       aIMap.cDataMapping<double,2,2>::Values(aVIn);


       const std::vector<cPt2dr> & aVOut = aIMap.cDataInvertibleMapping<double,2>::Inverses(aVIn);
       double aMaxD=0.0;
       double aMaxDisto=0.0;
       for (int aKPts=0 ; aKPts<int(aVIn.size()) ; aKPts++)
       {
           cPt2dr aDif =  aVIn[aKPts] - aMCS->cDataMapping<double,2,2>::Value(aVOut[aKPts]) ;
           aMaxD = std::max(aMaxD,Norm2(aDif));
           aMaxDisto = std::max(aMaxDisto,Norm2(aVIn[aKPts]-aVOut[aKPts]));
 
           // StdOut() << "Kp: " << aKPts << " PTS "<< aVIn[aKPts]<< " Dif " << aDif << "\n";
           // TestJacob(aMCS,aVOut[aKPts]);

       }
       // StdOut() << "MOYD " << aMaxD  << " " << aMaxDisto << "\n";
       MMVII_INTERNAL_ASSERT_bench(aMaxD<1e-5,"Distorsion inverse");
       // getchar();

       // StdOut() << aP << aQ << aP - aQ <<   " NBPAR " << aNbPar << "\n";

       delete anEqVal;
       delete anEqDer;
   }

}

};
