#include "MMVII_Geom2D.h"
#include "MMVII_SysSurR.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Interpolators.h"


namespace MMVII
{




/* *********************************************************** */
/*                                                             */
/*                         cEllipse                            */
/*                                                             */
/* *********************************************************** */

	//  =================== Create/Read/Write  ======================

/**   Create an ellipse from it equation : v0 XX + v1 XY + v2 YY + v3 X + v4 Y = 1
 *    C0 is a global translation (that has been added for numerical stability)
 *
 *                        (v0  v1) (X)             (X)
 *     E(X,Y) =    (X Y)  (v1  v2) (Y)  +  (v3 v4) (Y)  -1 =   tP  QF P +L P   -1
 *
 *        =>  S = 1/2  Q-1 P  because  E(X,Y) = t(P-S) Q (P-S) +  CSte
 */

cEllipse::cEllipse(cDenseVect<tREAL8> aDV,const cPt2dr & aC0) :
    mV    (aDV.Dup()),
    mNorm (std::sqrt(Square(mV(0)) + 2 * Square(mV(1))  + Square(mV(2)))),
    mC0   (aC0),
    mQF   (M2x2FromLines(cPt2dr(mV(0),mV(1)),cPt2dr(mV(1),mV(2))))
{

    cPt2dr aSol = SolveCol(mQF,cPt2dr(mV(3),mV(4)))/2.0;
    mCenter  = aC0-aSol;
    mCste = -1-QScal(aSol,mQF,aSol);

    cResulSymEigenValue<tREAL8>  aRSEV = mQF.SymEigenValue();

     mLGa = aRSEV.EigenValues()(0);
     mLSa = aRSEV.EigenValues()(1);

     mOk = (mLGa >0) && (mLSa>0) && (mCste<0) ;
     if (!mOk) return;

     mLGa = std::sqrt((-mCste)/mLGa);
     mLSa = std::sqrt((-mCste)/mLSa);
                     
     GetCol(mVGa,aRSEV.EigenVectors(),0);
     GetCol(mVSa,aRSEV.EigenVectors(),1);

     // There is no warantee on orientaion  from jacobi !!
     if ((mVGa ^ mVSa) < 0)
        mVSa = - mVSa;

     mRayMoy = std::sqrt(mLGa*mLSa);
     mSqRatio = std::sqrt(mLGa/mLSa);
}


cDenseVect<tREAL8>  ParamOfEllipse(tREAL8 aTeta,tREAL8 aLGa,tREAL8 aLSa)
{
  
     cDenseMatrix<tREAL8>  aMatR =      MatrRot(-aTeta);
     cDenseMatrix<tREAL8>  aMatDiag =   MatDiag(cPt2dr(1.0/Square(aLGa),1.0/Square(aLSa)));

      cDenseMatrix<tREAL8> aM = aMatR.Transpose() * aMatDiag  * aMatR;

      std::vector<tREAL8>  aV{aM.GetElem(0,0), aM.GetElem(0,1),aM.GetElem(1,1),0.0,0.0};
      return  cDenseVect<tREAL8>(aV);
}

cEllipse::cEllipse(const cPt2dr & aCenter,tREAL8 aTeta,tREAL8 aLGa,tREAL8 aLSa):
    cEllipse(ParamOfEllipse(aTeta,aLGa,aLSa),aCenter)
{
}

void cEllipse::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("Vect",anAux),mV);
     MMVII::AddData(cAuxAr2007("C0",anAux),mC0);
     if (anAux.Input())
     {
          *this = cEllipse(mV,mC0);
     }
}

void AddData(const  cAuxAr2007 & anAux,cEllipse & anEl)
{
     anEl.AddData(anAux);
}


	//  =================== Accessor  ======================

const cPt2dr & cEllipse::VGa() const {return mVGa;}
const cPt2dr & cEllipse::VSa() const {return mVSa;}
bool   cEllipse::Ok() const   {return mOk;}
tREAL8 cEllipse::LGa() const  {return mLGa;}
tREAL8 cEllipse::LSa() const  {return mLSa;}
tREAL8 cEllipse::RayMoy() const  {return mRayMoy;}
const cPt2dr &  cEllipse::Center() const {return mCenter;}
double cEllipse::TetaGa() const { return ToPolar(mVGa).y(); }




cPt2dr  cEllipse::PtOfTeta(tREAL8 aTeta,tREAL8 aMulRho) const
{
    return  mCenter+ mVGa *(cos(aTeta)*mLGa*aMulRho) + mVSa *(sin(aTeta)*mLSa*aMulRho);
}



cPt2dr  cEllipse::ToCoordLoc(const cPt2dr & aP0) const
{
     cPt2dr aP = (aP0-mCenter)/mVGa;

     return cPt2dr(aP.x()/mLGa,aP.y()/mLSa);
}

cPt2dr  cEllipse::FromCoordLoc(const cPt2dr & aP0) const
{
   return mCenter + VectFromCoordLoc(aP0);
}

cPt2dr  cEllipse::VectFromCoordLoc(const cPt2dr & aP0) const
{
   return  mVGa * cPt2dr(aP0.x()*mLGa,aP0.y()*mLSa);
}


cPt2dr cEllipse::ToRhoTeta(const cPt2dr & aP0) const
{
     return ToPolar(ToCoordLoc(aP0));
}

cPt2dr  cEllipse::Tgt(const cPt2dr &aPInit) const
{
     return VectFromCoordLoc(Rot90(ToCoordLoc(aPInit)));
}

cPt2dr  cEllipse::NormalInt(const cPt2dr &aPInit) const
{
     return Rot90(Tgt(aPInit));
}


cPt2dr  cEllipse::PtAndGradOfTeta(tREAL8 aTeta,cPt2dr &aGrad,tREAL8 aMulRho) const
{
    // Tgt = DP/Dteta =  (-mLGa sin(aTeta)  ,  mLSa cos(teta)
    // Norm = Tgt * P(0,-1) =>    mLSa cos(teta) , mLGa sin(aTeta)

   tREAL8 aCos = cos(aTeta);
   tREAL8 aSin = sin(aTeta);

   aGrad = VUnit( mVGa *(mLSa * aCos)  +  mVSa*(mLGa*aSin));
   return mCenter+ mVGa *(aCos*mLGa*aMulRho) + mVSa *(aSin *mLSa*aMulRho);
}

          //====================  Distances =============================

double cEllipse::SignedQF_D2(const cPt2dr& aP) const
{
    cPt2dr aQ = aP-mCenter;
    tREAL8 aRes =   QScal(aQ,mQF,aQ)  + mCste;

    return aRes / mNorm;
}

double cEllipse::QF_Dist(const cPt2dr & aP) const 
{
    return std::sqrt(std::abs(SignedQF_D2(aP)));
}

double cEllipse::ApproxSigneDist(const cPt2dr & aP0) const
{
    cPt2dr aP = (aP0-mCenter) / mVGa;
    aP = cPt2dr(aP.x()/mSqRatio,aP.y()*mSqRatio);

    return Norm2(aP) - mRayMoy;
}

double cEllipse::ApproxDist(const cPt2dr & aP0) const
{
	return std::abs(ApproxSigneDist(aP0));
}

double cEllipse::EuclidDist(const cPt2dr& aP) const
{
     return Norm2(aP-ProjOnEllipse(aP));
}

double cEllipse::SignedEuclidDist(const cPt2dr& aP) const
{
     cPt2dr aPProj = ProjOnEllipse(aP);

     tREAL8 aD = Norm2(aPProj-aP);

     bool Inside =  Scal(NormalInt(aPProj),aP-aPProj) >0;

     return Inside ? -aD  : aD ;
}
          //====================  PROJECTIONS =============================

cPt2dr  cEllipse::ProjOnEllipse(const cPt2dr & aPtAbs) const
{
    // Precaution because if A==B the polynon will be ill formed
    if (mSqRatio <= 1+3e-3)
    {
         if (mSqRatio <= 1+1e-7) 
            return  ProjNonEuclOnEllipse(aPtAbs);
         cPt2dr aRhoTeta =  ToRhoTeta(aPtAbs);

	 double aStep    = std::max(mSqRatio-1.0,1e-5)*4.0;
	 double aTetaCur = aRhoTeta.y();

	 int aNbStepMax = 4;
	 for (int aK=0 ; aK< aNbStepMax ; aK++)
	 {
	      double aD0 = SqN2(PtOfTeta(aTetaCur)-aPtAbs);
	      double aDm1 = SqN2(PtOfTeta(aTetaCur-aStep)-aPtAbs);
	      double aDp1 = SqN2(PtOfTeta(aTetaCur+aStep)-aPtAbs);

	      std::optional<double>  aDOpt =  InterpoleExtr(aDm1,aD0,aDp1);
	      if (aDOpt)
	      {
                  double aNewTeta = aTetaCur + *aDOpt * aStep;
		  if (SqN2(PtOfTeta(aNewTeta)-aPtAbs) >= aD0)
		  {
			  aK = aNbStepMax; // no improvement stop here
		  }
		  else
		  {
                       aTetaCur = aNewTeta;
		       aStep *= std::max(0.1,std::min(1.0,std::abs(*aDOpt)));
		  }
	      }
	      else
	      {
                   aK = aNbStepMax;  // plate parabol nothing todo ..
	      }
	 }

         return PtOfTeta(aTetaCur);
}

    cPt2dr aPLoc  = (aPtAbs-mCenter)/mVGa;

/*   PEllipse :  (A Cos(teta) , B Sin(teta))      PLoc (x,y)
     Tgt       :  (-A Sin(teta), B Cos(teta)

          (PE - PLoc ) . Tgt = 0

	  (A Cos(teta) - x)  (  -A Sin(teta))
	  (B Sin(teta) -y) .  (  B  Cos(teta))

	  0 =  -A^2  CS + Ax S + B^2 CS -yB C

	  0  =   C( (B^2-A^2) S - yB)  + AxS      

          A^2x^2 S^2  = (1-S^2) ( (B^2-A^2) S - yB) ^2 
 
*/

    cPolynom<tREAL8>  aC2  ({1.0,0.0,-1.0});  // 1 - S^2
    cPolynom<tREAL8>  AxS2 ({0.0,0.0,Square(mLGa*aPLoc.x())}); //  A^2x^2 S^2
    cPolynom<tREAL8>  B2A2SmYB ({-mLSa*aPLoc.y(),Square(mLSa)-Square(mLGa)});  // (B^2-A^2) S - yB

    cPolynom<tREAL8>  aPol = aC2 * B2A2SmYB * B2A2SmYB - AxS2;

    std::vector<tREAL8> aVRoots = aPol.RealRoots(1e-10,20);


    // StdOut() << "VRRRRRR " << aVRoots << std::endl;
    cWhichMin<cPt2dr,tREAL8>  aBestP(aPLoc,1e60);
    for (const auto & aSinus : aVRoots)
    {
        // MMVII_INTERNAL_ASSERT_tiny(std::abs(aSinus)<1.000001,"CHANGE TO DO IN ShowEllipse !!!");
         for (const auto  aSign : {-1.0,1.0})
         {
              tREAL8 aCos = aSign * std::sqrt(std::max(0.0,1-Square(aSinus)));

	      cPt2dr aProjLoc(mLGa*aCos,mLSa*aSinus);

	      // aBestP.Add(cPt2dr(aCos,aSinus),SqN2(aProjLoc-aPLoc));
	      aBestP.Add(aProjLoc,SqN2(aProjLoc-aPLoc));
         }
    }
    return mCenter + mVGa * aBestP.IndexExtre();

    /*
    cPt2dr aPCS = aBestP.IndexExtre();
    cPt2dr aPEllipse  (mLGa * aPCS.x()  , mLSa * aPCS.y());

    return mCenter + mVGa * aPEllipse;
    */
}

cPt2dr  cEllipse::ProjNonEuclOnEllipse(const cPt2dr & aPt) const
{
     return FromCoordLoc(VUnit(ToCoordLoc(aPt)));
}

          // ===================   BENCH ============================
	 
void cEllipse::BenchEllispe()
{
     for (int aK=0 ; aK<50 ; aK++)
     {
         static int aCpt=0; aCpt++;  
	 // StdOut() << "cccCPT=" << aCpt << std::endl; 
	 bool Bug=aCpt==-474; // No bug 4 now

         bool  IsCircle = (aK%5)==2;
         bool  IsAllmostCircle = (aK%5)==3;
         bool  IsAllmostCircle2 = (aK%5)==4;

         cPt2dr aC =  cPt2dr::PRandC() * 20.0;
	 tREAL8 aTeta = RandInInterval(-M_PI,M_PI);

	 tREAL8 aLGa  = RandInInterval(0.25,4.0);  // Random value >0
	 tREAL8 aLSa  = aLGa * RandInInterval(0.2,0.9);  // Random value, smaller enough (for stability of axe direction)
	 if (IsCircle) aLSa = aLGa;
	 if (IsAllmostCircle) aLSa =  aLGa * RandInInterval(0.9,0.95);
	 if (IsAllmostCircle2) aLSa =  aLGa * RandInInterval(0.999,0.9999);

	 cEllipse  anEl(aC,aTeta,aLGa,aLSa);

	 MMVII_INTERNAL_ASSERT_bench(Norm2(anEl.Center()-aC)<1e-5,"BenchEllispe");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anEl.LSa()-aLSa)<1e-5,"BenchEllispe");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anEl.LGa()-aLGa)<1e-5,"BenchEllispe");

	 tREAL8 aDif = std::abs(anEl.TetaGa() - aTeta);
	 aDif = std::min(aDif,M_PI-aDif);  // direction of axe are line/direction undefined up to M_PI
	 if (! IsCircle)
	    MMVII_INTERNAL_ASSERT_bench(aDif<1e-5,"BenchEllispe");

         for (int aKp=0 ; aKp<50 ; aKp++)
	 {
	     // Test   PtOfTeta o ToRhoTeta  = Id
             cPt2dr aP0 =  aC + cPt2dr::PRandUnitDiff(cPt2dr(0,0)) * 10.0;
             cPt2dr aP1 =  anEl.ToRhoTeta(aP0);
             cPt2dr aP2 =  anEl.PtOfTeta(aP1.y(),aP1.x());

	     MMVII_INTERNAL_ASSERT_bench(Norm2(aP0 -aP2)<1e-5,"BenchEllispe");

	     //  Compare tangent with a numerical calculation
	     tREAL8 aEps = 1e-3;
	     cPt2dr aTgtN = (anEl.PtOfTeta(aP1.y()+aEps,aP1.x()) - anEl.PtOfTeta(aP1.y()-aEps,aP1.x())) / (2*aEps);
	     cPt2dr aTgt = anEl.Tgt(aP0);
	     MMVII_INTERNAL_ASSERT_bench(Norm2(aTgt -aTgtN)<1e-3,"BenchEllispe Tgt");

	     // Test FromCoordLoc o ToCoordLoc = Id
             aP1 =  anEl.FromCoordLoc(aP0);
             aP2 =  anEl.ToCoordLoc(aP1);
	     MMVII_INTERNAL_ASSERT_bench(Norm2(aP0 -aP2)<1e-5,"BenchEllispe");
	 }

         for (int aKp=0 ; aKp<10 ; aKp++)
	 {
             cPt2dr aP1 =  anEl.PtOfTeta(RandInInterval(-M_PI,M_PI),RandInInterval(0.99,0.999));
             cPt2dr aP2 =  anEl.PtOfTeta(RandInInterval(-M_PI,M_PI),RandInInterval(1.001,1.01));
             cPt2dr aP3 =  anEl.PtOfTeta(RandInInterval(-M_PI,M_PI),1.0);

	     MMVII_INTERNAL_ASSERT_bench(anEl.SignedQF_D2(aP1)<0,"BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(anEl.SignedQF_D2(aP2)>0,"BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anEl.SignedQF_D2(aP3))<1e-5,"BenchEllispe");

	     MMVII_INTERNAL_ASSERT_bench(anEl.ApproxSigneDist(aP1)<0,"P1:BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(anEl.ApproxSigneDist(aP2)>0,"P2:BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anEl.ApproxSigneDist(aP3))<1e-5,"P3:BenchEllispe");

	     MMVII_INTERNAL_ASSERT_bench(anEl.SignedEuclidDist(aP1)<0,"P1:BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(anEl.SignedEuclidDist(aP2)>0,"P2:BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anEl.SignedEuclidDist(aP3))<1e-5,"P3:BenchEllispe");
	 }

         for (int aKp=0 ; aKp<10 ; aKp++)
	 {
             cPt2dr aP1 =  anEl.PtOfTeta(RandInInterval(0.1,10),RandInInterval(-M_PI,M_PI));
	     cPt2dr aPP = anEl.ProjOnEllipse(aP1);
	     if (Bug)
	     {
		     StdOut() << anEl.ToRhoTeta(aP1) << " " <<   Scal(anEl.Tgt(aPP),aP1-aPP)  << aP1-aPP  << std::endl;
		     StdOut() <<  "Ratiooo=" << aLSa/aLGa   << std::endl;
	     }

	     MMVII_INTERNAL_ASSERT_bench(std::abs(anEl.ToRhoTeta(aPP).x()-1.0)<1e-5,"BenchEllispe");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(anEl.Tgt(aPP),aP1-aPP))<1e-3,"BenchEllispe");
	     // getchar();
	 }

	 /* Check experimentaly that ApproxDist ~ EuclidDist and that the approximation can be bounded as a fonction
	  * of ellipse excentricity*/
	 tREAL8              aMaxRatio=0.0;
	 cWeightAv<tREAL8>  aWAvg;
         for (int aKp=0 ; aKp<100 ; aKp++)
	 {
             cPt2dr aP1 =  anEl.PtOfTeta(RandInInterval(0.9,1.1),RandInInterval(-M_PI,M_PI));
             tREAL8  aRatio= anEl.ApproxDist(aP1) / anEl.EuclidDist(aP1) ;
	     UpdateMax(aMaxRatio,aRatio);
	     aWAvg.Add(1.0,aRatio);
	 }

        if (IsCircle)
         {
             MMVII_INTERNAL_ASSERT_bench(std::abs(aMaxRatio-1)<1e-5,"BenchEllispe RATIO");
         }
         else
         {
              tREAL8  aStdR =  (aLGa / aLSa) - 1;
              tREAL8  aStdM = (aMaxRatio - 1.0) / aStdR;
              tREAL8  aStdA = (aWAvg.Average()- 1.0)  / aStdR;
              MMVII_INTERNAL_ASSERT_bench(aStdM<1.0,"BenchEllispe RATIO");
              MMVII_INTERNAL_ASSERT_bench(aStdA<1.0,"BenchEllispe RATIO");
         }

     }
}


/*  *********************************************************** */
/*                                                              */
/*               cEllipseEstimate                               */
/*                                                              */
/*  *********************************************************** */
cEllipse_Estimate::cEllipse_Estimate(const cPt2dr & aC0) :
    mSys  (new cLeasSqtAA<tREAL8> (5)),
    mC0   (aC0)
{
}

cLeasSqtAA<tREAL8> & cEllipse_Estimate::Sys() {return *mSys;}

cEllipse_Estimate::~cEllipse_Estimate()
{
    delete mSys;
}

void cEllipse_Estimate::AddPt(cPt2dr aP)
{
     aP = aP-mC0;

     cDenseVect<tREAL8> aDV(5);
     aDV(0) = Square(aP.x());
     aDV(1) = 2 * aP.x() * aP.y();
     aDV(2) = Square(aP.y());
     aDV(3) = aP.x();
     aDV(4) = aP.y();

     mSys->PublicAddObservation(1.0,aDV,1.0);

     mVObs.push_back(aP);
}

cEllipse cEllipse_Estimate::Compute()
{
     auto  aSol = mSys->Solve();

     return cEllipse(aSol,mC0);
     /// return  aRes;
}

/*  *********************************************************** */
/*                                                              */
/*               cExtractedEllipse                              */
/*                                                              */
/*  *********************************************************** */


cExtractedEllipse::cExtractedEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse) :
    mSeed      (aSeed),
    mEllipse   (anEllipse),
    mDist      (10.0),
    mDistPond  (10.0),
    mEcartAng  (10.0),
    mValidated (false)
{
}

/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */


cExtract_BW_Ellipse::cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest) :
        cExtract_BW_Target(anIm,aPBWT,aMasqTest)
{
}

void cExtract_BW_Ellipse::AnalyseAllConnectedComponents(const std::string & aNameIm)
{
    for (auto & aSeed : mVSeeds)
    {
        aSeed.mMarked4Test = mDMasqT.GetV(aSeed.mPixW);
	bool OkCC = AnalyseOneConnectedComponents(aSeed);
        if (aSeed.mMarked4Test && (!OkCC) )
	{
            StdOut() << "###  For Marked point " << aSeed.mPixW << " AnalyseCC = " << OkCC << "    ###" << std::endl;
	}
        if (OkCC)
        {
            bool OkFront = ComputeFrontier(aSeed);

            if (aSeed.mMarked4Test && (!OkFront))
	    {
                StdOut() << "###  For Marked point " << aSeed.mPixW << " AnalyseFront = " << OkFront << "  ###" << std::endl;
	    }

            if (OkFront)
            {
                AnalyseEllipse(aSeed,aNameIm);
            }
        }
    }
}

class cRadialBlurrCompute
{
        public :
	   typedef cIm1D<tREAL8>     tHist;
	   typedef cDataIm1D<tREAL8> tDHist;

           cRadialBlurrCompute
           (
	        const cSeedBWTarget & aSeed,
	        tREAL8 aDMax,
		size_t aNbInit
	   );
           cRadialBlurrCompute
           (
	        tREAL8 aBlack,
	        tREAL8 aWhite,
	        tREAL8 aDMax,
		size_t aNbInit
	   );




	   void Add(tREAL8 aDist,tREAL8 aGray);
	   void Compute();

	   cRadialBlurrCompute MakeNormalized(tREAL8 aNbSigma,size_t aNbInit);
	   void Show();

        private :
	   inline tREAL8 Dist2RIndex(tREAL8 aDist) const;
	   inline tREAL8 RIndex2Dist(tREAL8 aInd) const;
	   inline tREAL8 GradIndex2Dist(tREAL8 aInd) const;

	   tREAL8   mBlack;
	   tREAL8   mWhite;
	   tREAL8   mDMax;
           size_t   mNbInit;
	   size_t   mNbInCurv;
	   tHist    mPopOfD;
	   tDHist&  mDPopOfD;
	   tHist    mGrayOfD;
	   tDHist&  mDGrayOfD;
	   tHist    mDensity;
	   tDHist&  mDDensity;
	   int      mSignGrow;     /// 1 if growing, else -1
	   tREAL8   mSigmaFNN;
           tREAL8   mAvg ;         /// Avg but also normalization
           tREAL8   mStdDev;       /// Standard Dev  but also normalization
           cComputeStdDev<tREAL8>  mComputeSD;
	   std::vector<cPt2dr>  mVDG;  ///<  vector  Dist/Gray
};

tREAL8 cRadialBlurrCompute::Dist2RIndex(tREAL8 aDist)   const  {return (aDist + mDMax) * mNbInit;}
tREAL8 cRadialBlurrCompute::RIndex2Dist(tREAL8 aInd)    const {return aInd / mNbInit  -mDMax ;}
tREAL8 cRadialBlurrCompute::GradIndex2Dist(tREAL8 aInd) const {return RIndex2Dist(aInd+0.5);}

cRadialBlurrCompute::cRadialBlurrCompute
(
     tREAL8 aBlack,
     tREAL8 aWhite,
     tREAL8 aDMax,
     size_t aNbInit
)  :
   mBlack     (aBlack),
   mWhite     (aWhite),
   mDMax      (aDMax),
   mNbInit    (aNbInit),
   mNbInCurv  (round_ni(mDMax * mNbInit * 2)),
   mPopOfD    (mNbInCurv,nullptr,eModeInitImage::eMIA_Null),
   mDPopOfD   (mPopOfD.DIm()),
   mGrayOfD   (mNbInCurv,nullptr,eModeInitImage::eMIA_Null),
   mDGrayOfD  (mGrayOfD.DIm()),
   mDensity   (mNbInCurv,nullptr,eModeInitImage::eMIA_Null),
   mDDensity  (mDensity.DIm()),
   mSigmaFNN  (0.25),
   mAvg       (0.0),
   mStdDev    (1.0)
{
}

//  0.1 1  => 0.45726
//  1   1  => 0.500075
//  2   1  => 0.606591


cRadialBlurrCompute::cRadialBlurrCompute
(
     const cSeedBWTarget & aSeed,
     tREAL8 aDMax,
     size_t aNbInit
)  :
   cRadialBlurrCompute(aSeed.mBlack,aSeed.mWhite,aDMax,aNbInit)
{
}

void cRadialBlurrCompute::Add(tREAL8 aDist,tREAL8 aGray)
{
    aDist = (aDist-mAvg) / mStdDev;
    tREAL8 anInd = Dist2RIndex(aDist);
    if ( mDPopOfD.InsideBL(cPt1dr(anInd)))
    {
       mDPopOfD.AddVBL(anInd,1.0);         // increase population
       mDGrayOfD.AddVBL(anInd,aGray);      // increase gray
       mVDG.push_back(cPt2dr(aDist,aGray));  //  memorise for future reuse
    }
}

void cRadialBlurrCompute::Compute()
{
   ExpFilterOfStdDev(mDGrayOfD,3,mSigmaFNN*mNbInit);
   ExpFilterOfStdDev(mDPopOfD ,3,mSigmaFNN*mNbInit);

   //  Put in "GrayOfD" the  average (by divide by population)
   for (size_t aD=0 ; aD<mNbInCurv ; aD++)
   {
       // if (mDPopOfD.GetV(aD)>0)
       {
           tREAL8  aAvg = SafeDiv(mDGrayOfD.GetV(aD), mDPopOfD.GetV(aD));
           mDGrayOfD.SetV(aD,aAvg);
       }
   }
   mSignGrow = (mDGrayOfD.GetV(mNbInCurv-1)>mDGrayOfD.GetV(0)) ? 1 : -1;
   
   for (size_t aD=1 ; aD<mNbInCurv ; aD++)
   {
       // if ( (mDPopOfD.GetV(aD)>0) && (mDPopOfD.GetV(aD-1)>0) )
       {
            tREAL8 aGrad = mSignGrow * (mDGrayOfD.GetV(aD)- mDGrayOfD.GetV(aD-1)) ;
            aGrad = std::max(0.0,aGrad);
            mDDensity.SetV(aD-1,aGrad);
            // mDGrayOfD.SetV(aD-1,aGrad);
            mComputeSD.Add(aGrad,GradIndex2Dist(aD-1));
       }
   }

   mAvg    = mComputeSD.SomWV()  / mComputeSD.SomW();
   mStdDev = mComputeSD.StdDev(1e-10);
   // Convulation by mSigmaFNN has artificially increased the blurring, we correct it
   mStdDev  = std::sqrt(std::max(0.0,Square(mStdDev)-Square(mSigmaFNN)));
   // StdOut() << " * C0 => AVG=" <<  mAvg  <<  " DEV=" << mStdDev  << std::endl;

}

cRadialBlurrCompute cRadialBlurrCompute::MakeNormalized(tREAL8 aNbSigma,size_t aNbInit)
{

   cRadialBlurrCompute aRes(mBlack,mWhite,aNbSigma,aNbInit);

   aRes.mAvg = mAvg;
   aRes.mStdDev = mStdDev;

   for (const auto & aDG : mVDG)
       aRes.Add(aDG.x(),aDG.y());
   aRes.Compute();

   StdOut() << " * C0 => " <<  mAvg  <<  " " << mStdDev  << "  NN " << aRes.mAvg << " " << aRes.mStdDev << std::endl;

   return aRes;
}


/*
   cIm1D<tREAL8> aPopDist(aNbInCurv,nullptr,eModeInitImage::eMIA_Null);
   cDataIm1D<tREAL8>& aDPopDist = aPopDist.DIm();
   cIm1D<tREAL8> aGrayDist(aNbInCurv,nullptr,eModeInitImage::eMIA_Null);
   cDataIm1D<tREAL8>& aDGrayDist = aGrayDist.DIm();
   */



tREAL8  GraySimul(tREAL8 aD,cSeedBWTarget & aSeed)
{
    aD = std::min(0.0,(aD-1) * 0.5);  //  1-> 0  -1 -> -1
    aD =  CubAppGaussVal(aD);

    return  aSeed.mBlack + (aSeed.mWhite-aSeed.mBlack) * aD;
}

bool  cExtract_BW_Ellipse::AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm)
{
	// --  1 estimate the ellispe fitting the frontier
     cEllipse_Estimate anEEst(mCentroid);
     for (const auto  & aPFr : mVFront)
         anEEst.AddPt(aPFr);
     cEllipse anEl = anEEst.Compute();
     if (! anEl.Ok())
     {
        CC_SetMarq(eEEBW_Lab::eElNotOk);
        return false;
     }


     double aSomD = 0;
     [[maybe_unused]] double aSomRad = 0;   // Average radiometry diff, unused
     tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;
     for (const auto  & aPFr : mVFront)
     {
         aSomD += anEl.ApproxDist(aPFr);
         // aSomD += std::abs(anEl.EuclidDist(aPFr));  slower, and approx was validated
	 aSomRad += std::abs(mDIm.GetVBL(aPFr)-aGrFr);
     }

     aSomD /= mVFront.size();

     tREAL8 aSomDPond =  aSomD / (1+anEl.RayMoy()/50.0);

     int aNbPts = round_ni(4*(anEl.LGa()+anEl.LSa()));
     tREAL8 aSomTeta = 0.0;
     for (int aK=0 ; aK<aNbPts ; aK++)
     {
            double aTeta = (aK * 2.0 * M_PI) / aNbPts;
	    cPt2dr aGradTh;
	    cPt2dr aPt = anEl.PtAndGradOfTeta(aTeta,aGradTh);

	    if (! mDGx.InsideBL(aPt))
	    {
                  CC_SetMarq(eEEBW_Lab::eElNotOk);
                  return false;
	    }
            cPt2dr aGradIm (mDGx.GetVBL(aPt),mDGy.GetVBL(aPt));
	    // Case rather rare, bur would generate error
	    if (IsNull(aGradIm))
               return false;

	    aSomTeta += std::abs(ToPolar(aGradIm/-aGradTh).y());
     }
     aSomTeta /= aNbPts;

     if (aSomDPond>0.2)
     {
        CC_SetMarq(eEEBW_Lab::eBadEl);
	return false;
     }


     cExtractedEllipse  anEE(aSeed,anEl);
     anEE.mDist      = aSomD;
     anEE.mDistPond  = aSomDPond;
     anEE.mEcartAng  = aSomTeta;
     anEE.mVFront    = mVFront;

     if (aSomDPond>0.1)
     {
         CC_SetMarq(eEEBW_Lab::eAverEl);
     }
     else
     {
         if (aSomTeta>0.05)
	 {
            CC_SetMarq(eEEBW_Lab::eBadTeta);
	 }
	 else
	 {
            anEE.mValidated = true;
	 }
     }

     mListExtEl.push_back(anEE);

if (true)
{
#if (0)
   // si on veut afficher la fonion bicub
   {
      static bool First = false;
      if (First)
      {
           aSeed.mBlack = 0.0;
           aSeed.mWhite = 100.0;
	   for (tREAL8 aX=-2; aX<=2 ; aX +=0.2)
               StdOut() << "X: " << aX  << " " << GraySimul(aX,aSeed) << std::endl;
      }
      First = false;
   }

    tREAL8   aDMax = 4.0;
    size_t   aNbDig = 5;
    cRadialBlurrCompute aRBC(aSeed,aDMax,aNbDig);

   cPt2di  aPMargin(10,10);
   cBox2di aBox(aSeed.mPInf-aPMargin,aSeed.mPSup+aPMargin);
   aBox = aBox.Inter(mDIm);

   tREAL8 aMajD = aDMax * (anEl.LGa()/anEl.LSa());

   for (const auto & aPix : cRect2(aBox))
   {
       cPt2dr aPR = ToR(aPix);
       if (anEl.ApproxDist(aPR)< aMajD)
       {
          tREAL8 aDS = anEl.SignedEuclidDist(aPR)   ;

	  tREAL8 aGray = mDIm.GetV(aPix);
	  // aGray = GraySimul(aDS,aSeed);

	  // aRBC.Add(aDS*0.5+0.2,aGray);
	  aRBC.Add(aDS,aGray);
       }
   }

   aRBC.Compute();
   aRBC.MakeNormalized(3.0,aNbDig);
   getchar();


   // tREAL8   aDMax = 4.0;
   // size_t   aNbDig = 5;
   int      aNbInCurv = round_ni(aDMax * aNbDig * 2);

   // cPt2di  aPMargin(10,10);
   // cBox2di aBox(aSeed.mPInf-aPMargin,aSeed.mPSup+aPMargin);

   cIm1D<tREAL8> aPopDist(aNbInCurv,nullptr,eModeInitImage::eMIA_Null);
   cDataIm1D<tREAL8>& aDPopDist = aPopDist.DIm();
   cIm1D<tREAL8> aGrayDist(aNbInCurv,nullptr,eModeInitImage::eMIA_Null);
   cDataIm1D<tREAL8>& aDGrayDist = aGrayDist.DIm();

   cIm1D<tREAL8>    aGradGray(aNbInCurv,nullptr,eModeInitImage::eMIA_Null);
   cDataIm1D<tREAL8>& aDGradGray = aGradGray.DIm();
 
   // aBox = aBox.Inter(mDIm);

   // Accumulute the gray as a function of signed distanceto ellipse
   for (const auto & aPix : cRect2(aBox))
   {
       cPt2dr aPR = ToR(aPix);
       if (anEl.ApproxDist(aPR)< aDMax+1)  
       {
          tREAL8 aDS = anEl.SignedEuclidDist(aPR);
	  if (std::abs(aDS)< aDMax)
	  {
                tREAL8 aIndex = (aDS + aDMax) * aNbDig;
		if (aDPopDist.InsideBL(cPt1dr(aIndex)))
		{
                    aDPopDist.AddVBL(aIndex,1.0);
                    aDGrayDist.AddVBL(aIndex,mDIm.GetV(aPix));
		}
	  }
       }
   }
   //  Normalise to have average gray as value & compute grad of this value
   int aGrow = (aDGrayDist.GetV(aNbInCurv-1)>aDGrayDist.GetV(0)) ? 1 : -1;
   cComputeStdDev<tREAL8>  aStdD;

   for (int aD=0 ; aD<aNbInCurv ; aD++)
   {
       tREAL8  aAvg = SafeDiv(aDGrayDist.GetV(aD), aDPopDist.GetV(aD));
       aDGrayDist.SetV(aD,aAvg);
//============================
       if (aD>=1)   // compute grad
       {
          tREAL8 aGrad = aGrow *(aDGrayDist.GetV(aD)- aDGrayDist.GetV(aD-1)) ;
	  aGrad = std::max(0.0,aGrad);
          aDGradGray.SetV(aD-1,aGrad);
	  tREAL8 aPixC = aD-0.5; // The grad is computede bewteen D & D-1, do it correspond to pix D-0.5
          aStdD.Add(aGrad,aPixC/aNbDig-aDMax);
       }
   }

   tREAL8 aSomGrad = aStdD.SomW();
   aStdD = aStdD.Normalize(1e-5);
   tREAL8 aAvg = aStdD.SomWV();
   tREAL8 aDev = std::sqrt(aStdD.SomWV2()) ;

   // cGaussLaw  aGL(aAvg,aDev);
   std::unique_ptr<cAvgDevLaw> aGL (cAvgDevLaw::CubAppGaussLaw(aAvg,aDev));
   cComputeStdDev<tREAL8>  aStdGauss;

   for (int aDInt=0 ; aDInt<aNbInCurv-1 ; aDInt++)
   {
       tREAL8 aD = (aDInt+0.5)/aNbDig - aDMax;
       tREAL8  aGausW = aGL->NormalizedValue(aD);
       aStdGauss.Add(aGausW,aD);
   }
   tREAL8 aSomGauss = aStdGauss.SomW();
   tREAL8 aSomDif = 0.0;
   for (int aDInt=0 ; aDInt<aNbInCurv-1 ; aDInt++)
   {
       tREAL8 aGrad = aDGradGray.GetV(aDInt);
       tREAL8  aGausW = aGL->NormalizedValue((aDInt+0.5)/aNbDig - aDMax);
       aSomDif += std::abs(aGausW/aSomGauss - aGrad/aSomGrad);
   }

   aStdGauss = aStdGauss.Normalize(1e-5);

   StdOut()  << "AVG=" << aAvg  << " StdD="   << aDev  << "\n" << std::endl;

   /*
	   << " GAUS" << aStdGauss.SomWV() << " " << std::sqrt(aStdGauss.SomWV2())
	     << "  DIF=" << aSomDif << "\n\n";
	     */

   if (aSeed.mMarked4Test)
   {
       StdOut()  <<  " =========================== " << std::endl;
       for (const auto & aD : aDPopDist)
       {
           StdOut()  <<  " D: " <<  FixDigToStr( (aD.x()/double(aNbDig) - aDMax) , 3)
		     <<  " G: " <<  (aDGrayDist.GetV(aD)-  aSeed.mBlack) / (aSeed.mWhite-aSeed.mBlack)
		     <<  " Grag: " <<  (aDGradGray.GetV(aD) / aSomGrad) * 100.0
		     <<  " Gauss: " <<  aGL->NormalizedValue((aD.x()+0.5)/aNbDig - aDMax)/aSomGauss * 100.0
                      << "\n";
       }

       for (int aNb : {10,100,1000,10000})
       {
         cComputeStdDev<tREAL8>  aStdCub;
	 //int aNb=11;
         for (int aK=-aNb; aK<=aNb; aK++)
	 {
              tREAL8 aX = aK/double(aNb);
              tREAL8 aW = CubAppGaussVal(aX);
	      aStdCub.Add(aW,aX);
	      // StdOut() << "X:" << aX  << " W:" << aW << std::endl;
	 }
	 StdOut()  << "CUB DEV " << aStdCub.StdDev() << std::endl;
       }
       getchar();

   }

/*    Grad = A exp(-AX2 + BX4)
 *
 *    Log(Grad) =  A + BX2 + CX4
   cDataIm1D<tREAL8>& aDPopDist = aPopDist.DIm();
   cIm1D<tREAL8> aGrayDist(aNbInCurv,nullptr,eModeInitImage::eMIA_Null);
   cDataIm1D<tREAL8>& aDGrayDist = aGrayDist.DIm();
   cDataIm1D<tREAL8>& aDGradGray = aGradGray.DIm();
 */

   {
      int aNbDeg = 4;
      cLeasSqtAA<tREAL8> aSys(aNbDeg+1);
      for (int aDInt=0 ; aDInt<aNbInCurv ; aDInt++)
      {
         tREAL8 aPop = aDGradGray.GetV(aDInt);
	 if (aPop>0)
	 {
              tREAL8 aX = (aDInt+0.5)/aNbDig-aDMax;
              cDenseVect<tREAL8> aV(aNbDeg+1);
              for (int aK=0 ; aK<=aNbDeg ; aK++)
                  aV(aK) = pow(aX,2*aK);
	      aSys.AddObservation(aPop,aV,log(aPop));
	 }
      }
      cDenseVect<tREAL8> aSol = aSys.Solve();
      cWeightAv<tREAL8>  aAvgD;

      for (int aDInt=0 ; aDInt<aNbInCurv ; aDInt++)
      {
         tREAL8 aPop = aDGradGray.GetV(aDInt);
         // cDenseVect<tREAL8> aV(aNbDeg+1);
	 tREAL8 aSom = 0;
         tREAL8 aX = (aDInt+0.5)/aNbDig-aDMax;
         for (int aK=0 ; aK<=aNbDeg ; aK++)
             aSom += pow(aX,2*aK) * aSol(aK);

	 aAvgD.Add(aPop,std::abs(exp(aSom)-aPop));
         if (aSeed.mMarked4Test)
	 {
             StdOut() <<  "EEE=" << aDGradGray.GetV(aDInt) << " " << exp(aSom) << std::endl;
	 }
      }
      StdOut() << "sss=" <<  aSol.ToStdVect()   << " DDD=" << aAvgD.Average() << std::endl;
   }
#endif
}

   return true;
}

const std::list<cExtractedEllipse> & cExtract_BW_Ellipse::ListExtEl() const {return mListExtEl;}

void  cExtractedEllipse::ShowOnFile(const std::string & aNameIm,int aZoom,const std::string& aPrefName) const
{

    static int aCptIm = 0;
    aCptIm++;
    const cSeedBWTarget &  aSeed = mSeed;
    const cEllipse &       anEl  = mEllipse;


	cPt2di  aPMargin(6,6);
	cBox2di aBox(aSeed.mPInf-aPMargin,aSeed.mPSup+aPMargin);

	cRGBImage aRGBIm = cRGBImage::FromFile(aNameIm,aBox,aZoom);  ///< Allocate and init from file
	cPt2dr aPOfs = ToR(aBox.P0());

// MMVII_INTERNAL_ASSERT_tiny(false,"CHANGE TO DO IN ShowEllipse !!!");

        aRGBIm.DrawEllipse(cRGBImage::Blue,anEl.Center() - aPOfs,anEl.LGa(),anEl.LSa(),anEl.TetaGa());

        for (const auto  & aPFr : mVFront)
	{
            aRGBIm.SetRGBPoint(aPFr-aPOfs,cRGBImage::Red);
	    //StdOut() <<  "DDDD " <<  anEl.ApproxSigneDist(aPFr) << std::endl;
	}

	aRGBIm.ToFile(aPrefName + "_Ellipses_" + ToStr(aCptIm) + ".tif");
}

};
