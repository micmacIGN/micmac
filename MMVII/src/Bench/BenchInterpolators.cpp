#include "MMVII_Interpolators.h"

namespace MMVII
{



/**  Basic test on bilinear interpolator :   check that "cLinearInterpolator" works,
 *  "GetValueInterpol" work, and "Tabulated Interpolator" works.
 *
 *  Does it by comparing "hand crafted" GetVBL with :
 *     - cLinearInterpolator
 *     - Tabulated-raw
 *     - Tabulated-lin
 */

template <class Type>  void  TplInterpol_CmpLinearGetVBL(const cPt2di & aSz)
{
     // [0]  generate a random image
     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();

     // [1]  create 3 intepolator 
     cLinearInterpolator aBil1;  // analytic formula
     // tabulated : high tabulation to be accurate enough
     cInterpolator1D *  aTabBil1 =  cInterpolator1D::TabulatedInterp(aBil1,1e5,false);
     // tabulation with linear interpolation, more accurated
     cInterpolator1D *  aTabBiBil = cInterpolator1D::TabulatedInterp(aBil1,100,true);


     // [2]  generate random points, and compare the values of different interpolation mode
     int aNbTest = 0;
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));
     while (aNbTest<1000)
     {
         cPt2dr  aPt = aBoxR.GeneratePointInside();
	 if (aDIm.InsideBL(aPt))
	 {
             tREAL8 aV1 = aDIm.GetVBL(aPt);
             tREAL8 aV2 = aDIm.GetValueInterpol(aPt,aBil1);
             tREAL8 aV3 = aDIm.GetValueInterpol(aPt,*aTabBil1);
             tREAL8 aV4 = aDIm.GetValueInterpol(aPt,*aTabBiBil);

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-6,"Interpol 12 ");

	     tREAL8 aDif13 = std::abs(aV1-aV3) / tNumTrait<Type>::AmplRandomValueCenter();
	     MMVII_INTERNAL_ASSERT_bench(aDif13<1e-4,"Interpol bil 13 ");

	     tREAL8 aDif14 = std::abs(aV1-aV4) / tNumTrait<Type>::AmplRandomValueCenter();
	     MMVII_INTERNAL_ASSERT_bench(aDif14<1e-6,"Interpol bil 14 ");
             aNbTest ++;
	 }
     }

     delete aTabBil1;
     delete aTabBiBil;
}


/*  Test GetValueInterpol with a less basic interolator : cubic
 *  we use Cubic(-0.5) as for this value the interpolation of a linear
 *  function is exact, so we have an external ground truth
 *
 *  The test is made with analytic and tabulated interpolator
 *
 */
template <class Type>  void  TplInterpol_CmpCubicFoncLinear(const cPt2di & aSz)
{
	// ----- [0]  parameters for linear function "Ax+By+C"
     tREAL8 A = -2;
     tREAL8 B = 3;
     tREAL8 C = -123;

	// -----  [1]   Generate the linear function  in the image
     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();

     for (const auto & aP : aDIm)
     {
         aDIm.SetV(aP,A*aP.x()+B*aP.y() + C);
     }

	// -----  [2]   Generate the linear function  in the image
     // bicub that interpolate exactly linear function
     cCubicInterpolator aI3(-0.5);
     // tabulated version, create with "AllocFromNames" 
     cDiffInterpolator1D * aTabI3 = cDiffInterpolator1D::AllocFromNames({"Tabul","1000","Cubic","-0.5"});


	// -----  [3]   Test itsefl
     int aNbTest = 0;
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));
     while (aNbTest<5000)
     {
         cPt2dr  aPt = aBoxR.GeneratePointInside();
         if (aDIm.InsideInterpolator(aI3,aPt))
         {
             aNbTest++;

	     tREAL8 aV0 = A*aPt.x()+B*aPt.y() + C;   // Analytical value
	     tREAL8 aV1 = aDIm.GetValueInterpol(aPt,aI3);  // interpolation with analytical interpolator
	     tREAL8 aV2 = aDIm.GetValueInterpol(aPt,*aTabI3);  // interpolation with tabulated interpolator

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV0-aV1)<1e-6,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV0-aV2)<1e-6,"Interpol ");
         }
     }
     delete aTabI3;
}


/**  Test interpolation with a low-frequency function for the computation of derivative :
 *
 *     # compute derivative of interpolator with  "GetValueAndGradInterpol"
 *     # compute numerical derivative (with "epsilon-value")
 *
 *     Also make some test with sin-card apodized. Theoretically  (shanon-niquist ?) with sufficiently
 *     accurate sinc-card we should be abble to reconstruct exactly the continuous function ...
 */

template <class Type>  void  TplInterpol_FuncLowFreq(const cPt2di & aSz,const cDiffInterpolator1D & anInterpol)
{
     // [0] : generate the low frequency image  F(x,y) = sin(x/aPx) * cos(y/aPy)
     tREAL8 aPerX = 10.0;
     tREAL8 aPerY = 15.0;

     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();

     for (const auto & aP : aDIm)
     {
         tREAL8 aVal = std::sin(aP.x()/aPerX) *  std::cos(aP.y()/aPerY) ;
	 aDIm.SetV(aP,aVal);
     }

     //cCubicInterpolator aI3(aCoeff);

     cDiffInterpolator1D * aTabInt= cDiffInterpolator1D::TabulatedInterp(anInterpol,10000);
     cSinCApodInterpolator  aSinCInt(40.0,20.0);
     cDiffInterpolator1D * aTabSinC= cDiffInterpolator1D::TabulatedInterp(aSinCInt,10000);

     int aNbTest = 0;
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));

     cStdStatRes  aStatDifSinC;  // statistic on diff  I (interpol sinc)  / F(x,y)
     cStdStatRes  aStatDifDxSinC;  // statistic on diff I(interpol sinc)/dx  / dF/dx
     cStdStatRes  aStatDifDySinC;   // statistic on diff I(interpol sinc)/dx  / dF/dy

     while (aNbTest<5000)
     {
         tREAL8 aEpsDif = 1e-4;
         cPt2dr  aPt = aBoxR.GeneratePointInside();
         if (aDIm.InsideInterpolator(aSinCInt,aPt,aEpsDif*1.01))
         {
             aNbTest++;
             // compute value and derivative of interopolated function using "GetValueAndGradInterpol"
             auto [aValue,aGrad] = aDIm.GetValueAndGradInterpol(aPt,anInterpol);

	     // compute value and derative with finite difference methods

	     tREAL8 aV0 = aDIm.GetValueInterpol(aPt,anInterpol);

	     cPt2dr aEpsX(aEpsDif,0);
	     tREAL8 aVDx = (aDIm.GetValueInterpol(aPt+aEpsX,anInterpol)-aDIm.GetValueInterpol(aPt-aEpsX,anInterpol)) / (2*aEpsDif);

	     cPt2dr aEpsY(0,aEpsDif);
	     tREAL8 aVDy = (aDIm.GetValueInterpol(aPt+aEpsY,anInterpol)-aDIm.GetValueInterpol(aPt-aEpsY,anInterpol)) / (2*aEpsDif);

	     // check that value with finite difference are close enough to value with GetValueAndGradInterpol
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aValue  - aV0)<1e-6,"Interpol VAndDer");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aGrad.x() - aVDx)<1e-5,"Interpol VAndDer X");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aGrad.y() - aVDy)<1e-5,"Interpol VAndDer Y");

	     //  test that the value got with tabulated are close enough, btw not so close for derivative 
             auto [aVTab,aGTab] = aDIm.GetValueAndGradInterpol(aPt,*aTabInt);

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aGrad.x()-aGTab.x() )<1e-4,"Interpol VAndDer Tabulated");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aGrad.y()-aGTab.y() )<1e-4,"Interpol VAndDer Tabulated");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aValue-aVTab )<1e-6,"Interpol VAndDer Tabulated");

	     // memorize the difference between "real ground truth sinusoidal" function and interpolation
	     // by sinus-cardinal.
	     // as interoplation with sin card with high value can cost a lot, dont do it at each step
	     if ((aNbTest%20)==0)
	     {
                  auto [aVTSinC,aGradTSinC]  = aDIm.GetValueAndGradInterpol(aPt,*aTabSinC);

                  tREAL8 aVTh = std::sin(aPt.x()/aPerX) *  std::cos(aPt.y()/aPerY) ;
                  tREAL8 aDerThX = std::cos(aPt.x()/aPerX) *  std::cos(aPt.y()/aPerY)  / aPerX;
                  tREAL8 aDerThY = std::sin(aPt.x()/aPerX) *  -std::sin(aPt.y()/aPerY)  / aPerY;

	          aStatDifSinC.Add(std::abs(aVTSinC-aVTh));
	          aStatDifDxSinC.Add(std::abs(aGradTSinC.x()-aDerThX));
	          aStatDifDySinC.Add(std::abs(aGradTSinC.y()-aDerThY));
	     }
	 }
     }
     if (0)
     {
        StdOut() << "AVG DIF SINC =" << aStatDifSinC.Avg() << " " << aStatDifSinC.Max() << "\n";
        StdOut() << "AVG DIF SINC =" << aStatDifDxSinC.Avg() << " " << aStatDifDxSinC.Max() << "\n";
        StdOut() << "AVG DIF SINC =" << aStatDifDySinC.Avg() << " " << aStatDifDySinC.Max() << "\n";
     }

     // test coincidence interpolation/"real function" as predict shannon-nyquist, btw the cannot be strict
     // as its rather theorictal value with slow convergencre
     MMVII_INTERNAL_ASSERT_bench(std::abs(aStatDifSinC.Max()  )<2e-4,"SinCard exact reconstitution");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aStatDifDxSinC.Max()  )<2e-3,"SinCard exact reconstitution for X-derivate");
     MMVII_INTERNAL_ASSERT_bench(std::abs(aStatDifDySinC.Max()  )<2e-3,"SinCard exact reconstitution for Y-derivate");


     delete aTabInt;
     delete aTabSinC;
}

template <class Type>  void  TplInterpol_FuncLowFreq(const cPt2di & aSz)
{
     cCubicInterpolator aI3(-0.5);
     // const cDiffInterpolator1D& aDifInt = aI3;
     TplInterpol_FuncLowFreq<Type>(aSz,aI3);

     cMPD2Interpol aMPD2;
     TplInterpol_FuncLowFreq<Type>(aSz,aMPD2);
}


/** Do some "intrisiq" test on interpolators, they are not related to
 * any interpolation but to some theoretical properties of kernels
 */


void BenchIntrinsiqOneInterpol
     (
          const cInterpolator1D & anInt,
	  bool  IsPartUnit,   // Partition of unity 
	  bool  IsSym         // Is it symetric
     )
{
    if (IsPartUnit)
    {
        // if its a "partition" of unity it should complies with sum of value congruent to 1
	// equals 1 for any phase
        for (tREAL8 aPh=-0.1 +0.01 *RandUnif_C() ; aPh<1.1 ; aPh += 0.0002317845) // test +- randomly "many" phase
	{
            tREAL8 aSumW =  0.0;
            for (int aK= -anInt.SzKernel()-3 ; aK<anInt.SzKernel()+3 ; aK++)
	    {
                 aSumW += anInt.Weight(aK+aPh);
	    }
            MMVII_INTERNAL_ASSERT_bench(std::abs(aSumW-1.0)<1e-5,"Interpol Unity Partition");
	}
    }
    if (IsSym)
    {
        // test the symetry of the kernel
        for (tREAL8 anX=-0.1 ; anX<anInt.SzKernel()+10 ; anX += 0.00531)
	{
            tREAL8 aDif = std::abs(  anInt.Weight(anX)-anInt.Weight(-anX)  );
	    MMVII_INTERNAL_ASSERT_bench(std::abs(aDif)<1e-8,"Interpol VAndDer Tabulated");
	}
    }
}

// variant with Ptr for deleting object
void PtrBenchIntrinsiqOneInterpol ( const cInterpolator1D * anInt, bool  IsPartUnit,bool  IsSym)
{
    BenchIntrinsiqOneInterpol(*anInt,IsPartUnit,IsSym);
    delete anInt;
}


void BenchIntrinsiqOneDiffInterpol
     (
          const cDiffInterpolator1D & anInt,
	  bool  IsPartUnit,   // Partition of unity 
	  bool  IsSym         // Is it symetric
     )
{
     BenchIntrinsiqOneInterpol(anInt,IsPartUnit,IsSym);
}


/**  Check that the "MPD's" interpolator commply with their specification. The specif is that
 *   for any phase in [-0.5,0.5] the weight will be shared on 3 point [-1,0,1] , let A,B,C
 *   be the 3 weights.
 */

void BenchIntrinsiqMPDInterpol(const cInterpolator1D & anInt,tREAL8 anExp)
{
     // first it must be an interpolator
     BenchIntrinsiqOneInterpol(anInt,true,true);

     int aNb=100; // number of phase to sample regularly [0,0.5]
     for (int aK=0 ; aK<=aNb ; aK++)
     {
          tREAL8 aPh = (aK*0.5)/tREAL8(aNb) ;  // phase
	  tREAL8 aDA = 1+aPh;   // distance phase to -1
	  tREAL8 aDB = aPh;     // distance phase to -0
	  tREAL8 aDC = 1-aPh;   // distance phase to 1

	  tREAL8 aA = anInt.Weight(aDA);  // weight -1
	  tREAL8 aB = anInt.Weight(aDB);  // weight 0
	  tREAL8 aC = anInt.Weight(aDC);  // weight 1

	  tREAL8 aSigmaRef = std::pow(0.5,anExp); // theoretical sigma
	  tREAL8 aSigma    = aA*std::pow(aDA,anExp) + aB*std::pow(aDB,anExp) + aC*std::pow(aDC,anExp);  // sigma of interpol


	  //  sum of weight are equals to 1
	  MMVII_INTERNAL_ASSERT_bench(std::abs(aA+aB+aC-1.0)<1e-5,"Interpol MPD : Weigh=1");
	  //  centroid is equal to the targeted phase
	  MMVII_INTERNAL_ASSERT_bench(std::abs(-aA+aC-aPh)<1e-5,"Interpol MPD : Weigh=1");
	  //  variance is constant
	  MMVII_INTERNAL_ASSERT_bench(std::abs(aSigma-aSigmaRef)<1e-5,"Interpol MPD : Weigh=1");

	  // finnaly check that for 1/2 pixel we just do the average between 2 closest pixel
	  if (aK==aNb)
	  {
              // StdOut() << "ABC=" << aA << " " << aB << " " << aC << "\n";
	      MMVII_INTERNAL_ASSERT_bench(std::abs(aA)+std::abs(aB-0.5)+std::abs(aC-0.5)<1e-5,"Interpol MPD : Weigh=1");
	  }
     }
}




void  BenchInterpol(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Interpol")) return;

     // This one, obviously should not pass the "unity partition test" 
     //  --  BenchIntrinsiqOneInterpol(cSinCApodInterpolator(5,5),true,true);
   
     // On the other hand, the tabulation of sinus cardinal apodized should pass all the test
     // as there was some problem initially with tabulation of SinC, we make test with various params
     for (const auto & aNbTabul : {1000,1001,1002,1003})
     {
          PtrBenchIntrinsiqOneInterpol(cDiffInterpolator1D::TabulatedInterp(cSinCApodInterpolator(5,5),aNbTabul),true,true);
          PtrBenchIntrinsiqOneInterpol(cDiffInterpolator1D::TabulatedInterp(cSinCApodInterpolator(5.5,5),aNbTabul),true,true);
          PtrBenchIntrinsiqOneInterpol(cDiffInterpolator1D::TabulatedInterp(cSinCApodInterpolator(4.5,5),aNbTabul),true,true);
     }

     // bench different analytical or tabulated interolator
     PtrBenchIntrinsiqOneInterpol(new cLinearInterpolator(),true,true);
     BenchIntrinsiqOneDiffInterpol(cMPD2Interpol(),true,true);


     cDiffInterpolator1D * aTabMPD2= cDiffInterpolator1D::TabulatedInterp(cMPD2Interpol(),1000);
     BenchIntrinsiqOneDiffInterpol(*aTabMPD2,true,true);
     delete aTabMPD2;

     //  also make the test of creation by name
     PtrBenchIntrinsiqOneInterpol(cDiffInterpolator1D::AllocFromNames({"Tabul","1000","MPD2"}),true,true);


     // bench specific to MPD interpol 
     BenchIntrinsiqMPDInterpol(cMPD2Interpol(),2.0); // test with analytical MPD2
     for (const auto & anExp : {1.0,2.0,3.0})
     {
        // test with analytical  MPDK (very slow)
        BenchIntrinsiqMPDInterpol(cMPDKInterpol(anExp),anExp);
        // test with tabulated  MPDK 
        auto aPtrTabInt = cDiffInterpolator1D::TabulatedInterp(cMPDKInterpol(anExp),1000);
        BenchIntrinsiqMPDInterpol(*aPtrTabInt,anExp);
	delete aPtrTabInt;
     }

     //  Bench on bicub interpola
     //
     for (const auto & aP : {0.0,1.0,2.0,3.0})
     {
         cCubicInterpolator  anI3(aP);

	 BenchIntrinsiqOneDiffInterpol(anI3,true,true);

	 MMVII_INTERNAL_ASSERT_bench(std::abs(1-anI3.Weight(0))<1e-8,"Interpol ");   // F(0)=1
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(1))<1e-8,"Interpol ");     // F(1)=0
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(2))<1e-8,"Interpol ");     // F(2)=0
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(2.1))<1e-8,"Interpol ");    // F(X>2) = 0
										 //
	 for (const auto aEps : {1e-2,1e-3,1e-4})
	 {
             // Derivate in 2 = 0
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(2-aEps))<4*Square(aEps) ,"Interpol ");
	     // diffrentuale in 1  dF/dx+ = dF/dx-
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(1-aEps) + anI3.Weight(1+aEps))<4*Square(aEps) ,"Interpol ");
	 }
	 // compare  "analytical derivatives" with finite diff
         cEpsDiffFctr aEpsI3(anI3,1e-5);

	 cTabulatedDiffInterpolator aTabDif(anI3,10000);
	 cDiffInterpolator1D * aPtrI3 = cDiffInterpolator1D::TabulatedInterp(new cCubicInterpolator(aP),10000);

	 for (int aK=0 ; aK<1000 ; aK++)
	 {
             tREAL8 anX = RandInInterval(-3,3);

	     tREAL8 aD1 = anI3.DiffWeight(anX);
	     tREAL8 aF1 = anI3.Weight(anX);

	     tREAL8 aD2 = aEpsI3.DiffWeight(anX);
	     tREAL8 aF2 = aEpsI3.Weight(anX);

	     auto [aF3,aD3] = anI3.WAndDiff(anX);

	     // StdOut() << "DDDD X=" << anX << " "  << aD1 << " DifF=" << aD3 << " Dif=" << aD1 -aD3 << "\n";
	     // StdOut() << "FFFF X=" << anX << " "  << aF1 << " DifF=" << aF3 << " Dif=" << aF1 -aF3 << "\n\n";

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aD2)<1e-4 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF1-aF2)<1e-8 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aD3)<1e-4 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF1-aF3)<1e-8 ,"Interpol ");


	     tREAL8 aD4 = aTabDif.DiffWeight(anX);
	     tREAL8 aF4 = aTabDif.Weight(anX);
	     auto [aF5,aD5] = aTabDif.WAndDiff(anX);
	     auto [aF6,aD6] = aPtrI3->WAndDiff(anX);

	     //  these one should be very clos if not ==0, because they are same bilin formula
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF4-aF5)<1e-8 ,"Interpol TabDif ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD4-aD5)<1e-8 ,"Interpol  TabDif");

	     // StdOut() << "D455  " <<  aD3-aD5  << " " << aF3 - aF5 << "\n";
	     // Not so close : analyticall formula & bilin tab
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD3-aD5)<1e-3 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF3-aF5)<1e-3 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD6-aD5)<1e-8 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF6-aF5)<1e-8 ,"Interpol ");
	 }
	 delete aPtrI3;
     }

     TplInterpol_CmpLinearGetVBL<tU_INT1>(cPt2di(230,127));
     TplInterpol_CmpLinearGetVBL<tREAL4>(cPt2di(144,201));


     TplInterpol_CmpCubicFoncLinear<tINT2>(cPt2di(199,147));
     TplInterpol_CmpCubicFoncLinear<tREAL8>(cPt2di(100,188));

     TplInterpol_FuncLowFreq<tREAL8>(cPt2di(800,700));

     aParam.EndBench();
}


/* *************************************************** */
/*                                                     */
/*                ::                                   */
/*                                                     */
/* *************************************************** */


tREAL8 CubAppGaussVal(const tREAL8& aV)
{
   tREAL8 aAbsV = std::abs(aV);
   if (aAbsV>1.0) return 0.0;

   tREAL8 aAV2 = Square(aAbsV);

   return 1.0 + 2.0*aAbsV*aAV2 - 3.0*aAV2;
   // cubic formula , return (mA+2) * x3-(mA+3)*x2+1;   for A=0 -> 2x3-3x2+1  . 
}

/**   Compute the weighting for ressampling one pixel of an image with a mapping M.
 *  Formalisation :
 *
 *      - we have pixel out  Co
 *      - we have an image weighing arround Co  W(P) = BiCub((P-Co)/aSzK)
 *      - let S be the support of W(P) we compute the box of M-1(S)
 *
 */

cRessampleWeigth  cRessampleWeigth::GaussBiCub(const cPt2dr & aCenterOut,const cAff2D_r & aMapO2I, double aSzK)
{
     cRessampleWeigth aRes;

     // [1] compute the box in input image space 
     cPt2dr aSzW = cPt2dr::PCste(aSzK);
     cBox2dr aBoxOut(aCenterOut-aSzW,aCenterOut+aSzW);
     cBox2di aBoxIn =  ImageOfBox(aMapO2I,aBoxOut).Dilate(1).ToI();

     cAff2D_r  aMapI2O = aMapO2I.MapInverse();

     double aSomW = 0.0;
     for (const auto & aPixIn : cRect2(aBoxIn))
     {
         cPt2dr aPixOut = aMapI2O.Value(ToR(aPixIn));
         double aW =  CubAppGaussVal(Norm2(aPixOut-aCenterOut)/aSzK);
         if (aW>0)
         {
            aRes.mVPts.push_back(aPixIn);
            aRes.mVWeight.push_back(aW);
            aSomW += aW;
         }
     }

     // if not empty  , som W = 1
     if (aSomW>0)
     {
        for (auto & aW : aRes.mVWeight)
        {
            aW /= aSomW;
        }
     }

     return aRes;
}
#if (0)
#endif

};

