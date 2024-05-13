#include "MMVII_Image2D.h"
#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_Interpolators.h"
// #include "../TutoBenchTrianguRSNL/TrianguRSNL.h"

#include "MMVII_TplSymbImage.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{


/********************************************/
/*                                          */
/*              cGaussF2D                   */
/*                                          */
/********************************************/

/**  class for implemanting basic 2D-gauss law */

class  cGaussF2D
{
     public :
           double  Val(const cPt2dr & aP) const; ///< Value of  law
           cGaussF2D(const cPt2dr & aC,double aSigm) ; ///< Ctsr from center and variance

	   /** return law resulting from composition  Val(P) = G(aHom(aP)) */
           cGaussF2D(const cGaussF2D & G2,const  cHomot2D<tREAL8> & aHom );
     private :
           cPt2dr  mC;   ///< Center of law
           double  mSigm; ///< variance oflaw
};


cGaussF2D::cGaussF2D(const cPt2dr & aC,double aSigm) :
    mC    (aC),
    mSigm (aSigm)
{
}
double  cGaussF2D::Val(const cPt2dr & aP) const
{
   double aN2 = SqN2(aP-mC) / (2*Square(mSigm));
   return  exp(-aN2) ;
}

cGaussF2D::cGaussF2D(const cGaussF2D & aG ,const  cHomot2D<tREAL8> &  aH) : 
    cGaussF2D
    (
           (aG.mC -  aH.Tr()) / aH.Sc() ,
           aG.mSigm/aH.Sc()
    )
{
}

/********************************************/
/*                                          */
/*              cTestDeformIm               */
/*                                          */
/********************************************/

class cTestDeformIm
{
     public :
         cTestDeformIm
         (
	      int aSzGlob,
              double aEps, // Epsilon of perturbation of ground truth
	      bool Show,
	      cDiffInterpolator1D * anInterp,  // if given replace bilinear by interpol
              tREAL8 aTargetResidual
         );
         ~cTestDeformIm();
         /// Make one iteration of adding of non linear least square
	 bool OneIterationFitModele(bool IsLast);
     private :

	 bool               mShow; // print result, export image ...
         int                mSzGlob; ///< variable used for dimensionning the rest
         double             mAmplRad;  ///<  amplitude of radiometry
	 cDiffInterpolator1D * mInterpol;  ///< interpolator, if exist use linear/grad instead of  
         tREAL8              mTargetResidual;

         cPt2di             mSzIm;   ///<  size of image
         cIm2D<tREAL8>      mIm;     ///<  image matched
         cDataIm2D<tREAL8>& mDIm;    ///< data-image
         cPt2dr             mCenterLaw;  ///< center of the law
         double             mSigmaIm;   ///< variance of the law

         double             mTrRad;    ///<  Ground truth translation on radiometry
         double             mScRad;    ///<  Ground truth scale on radiometry

         cHomot2D<tREAL8>   mGT_I2Mod;     ///<  Ground truth homotethy   Image  -> Modele
         cHomot2D<tREAL8>   mGT_Mod2Im;    ///<  Ground truth homotethy   Modele -> Image

         cGaussF2D          mGaussIm;      ///<  Gaussian in image space
         cGaussF2D          mGaussModel;   ///<  Gaussian in model space

         std::vector<cPt2dr> mVPtsMod;     ///<  points sampling the model
         std::vector<double> mValueMod;    ///<  values of points in mVPtsMod

	 cResolSysNonLinear<tREAL8>* mSys;   ///< Non Linear Sys for solving problem
         cCalculator<double> *       mEqHomIm;  ///< calculator giving access to values and derivatives
};

cTestDeformIm::cTestDeformIm(int aSzGlob,double aEps,bool Show,cDiffInterpolator1D *anInterpol,tREAL8 aTargetResidual) :
   mShow       ( Show),
   mSzGlob     ( aSzGlob),  //  geometric value are proportional to aSzGlob (or w/o dimension)
   mAmplRad    (255.0),     //  radiometric value are proportional to mAmplRad (or w/o dimension)
   mInterpol   (anInterpol),
   mTargetResidual (aTargetResidual),
   mSzIm       ( mSzGlob,mSzGlob), 
   mIm         ( mSzIm),
   mDIm        ( mIm.DIm()),
   mCenterLaw   ( ToR(mSzIm)/2.0),  //put center at midle of image
   mSigmaIm    (mSzGlob/5.0),
   mTrRad      (mAmplRad/20.0),
   mScRad      (0.7),
   mGT_I2Mod   (cPt2dr(10,16),2),  // arbitrary value for homotethy  Image -> Modele
   mGT_Mod2Im  (mGT_I2Mod.MapInverse()),  
   mGaussIm    (mCenterLaw,mSigmaIm),
   mGaussModel (mGaussIm,mGT_Mod2Im),  // gaussian for modelby composition of gausian image & homothety
   mSys        (nullptr),
   mEqHomIm    (nullptr)
{
    //  -1 - parse all the pixel to inialize
    for (const auto & aPixIm :  mDIm)
    {
         cPt2dr aPMod = mGT_I2Mod.Value(ToR(aPixIm)); // corresponding point in the model
	 double aVModele = mGaussModel.Val(aPMod)*mAmplRad;
         
	 tREAL8 aMargin = 3.0;
	 bool isOk =     mInterpol                                            ?
		     mDIm.InsideInterpolator(*mInterpol,ToR(aPixIm),aMargin)  :
		     (mDIm.Interiority(aPixIm)>aMargin)                       ;
	 if (isOk) // maybe too strict , but to be sure to be inside ...
	 {
            mVPtsMod.push_back(aPMod); // add point in the grid of model
            mValueMod.push_back(aVModele); // add value of model
	 }

         // set "perfect" value of image
         mDIm.SetV(aPixIm, mTrRad + mScRad*aVModele);

      //  check validity on gaussian composed with homotethy
         double aV1 = mGaussModel.Val(aPMod);
         double aV2 = mGaussIm.Val(mGT_Mod2Im.Value(aPMod));
         MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-6,"Gauss compos");
    }
    if (mShow)
       mDIm.ToFile("GaussDef.tif");

    //  -2 - compute initial parameters  by randomization of ground truth
    cDenseVect<double> aVInit(5);
    aVInit(0) =  mScRad * (1+ aEps * RandUnif_C());
    aVInit(1) = mTrRad + (mAmplRad * mScRad * aEps * RandUnif_C());
    aVInit(2) =  mGT_Mod2Im.Sc() * (1+ aEps * RandUnif_C());
    aVInit(3) =  mGT_Mod2Im.Tr().x() + mSzGlob * aEps * RandUnif_C();
    aVInit(4) =  mGT_Mod2Im.Tr().y() + mSzGlob * aEps * RandUnif_C();


    //  -3 - allocate system and calculator
    mSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense,aVInit);
    mEqHomIm =      mInterpol                          ?
	        EqDeformImLinearGradHomotethy(true,1)  :
	        EqDeformImHomotethy(true,1)            ; // true-> with derivative,  1=sz of buffer

    StdOut() << "cTestDeformImcTestDeformImcTestDeformIm\n";
}

cTestDeformIm::~cTestDeformIm() 
{
    delete mSys;
    delete mEqHomIm;
    delete mInterpol;
}

bool cTestDeformIm::OneIterationFitModele(bool IsLast)
{
   //----------- index of unkown, basic here because the unknown are the same for each equation
   std::vector<int>  aVecInd{0,1,2,3,4};
   int aNbObsIm =  mInterpol  ? FormalGradInterpolIm2D_NbObs  : FormalBilinIm2D_NbObs;
   //----------- allocate vec of obs : 6 for image, 3 for model
   std::vector<double> aVObs(aNbObsIm+3,0.0);

   //----------- extract current parameters
   cDenseVect<double> aVCur = mSys->CurGlobSol();
   cHomot2D<tREAL8>   aCurHomM2I(cPt2dr(aVCur(3),aVCur(4)),aVCur(2)); // current homotethy
   double aCurScR = aVCur(0);  // current scale on radiometry
   double aCurTrR = aVCur(1);  // current translation on radiometry
			    
   //----------- declaration of indicator of convergence
   double aSomDif = 0; // sum of difference between model and image
   double aSomMod = 0; // sum of value of model, to normalize the difference
   double aNbOut = 0; //  number of points out of image

   // Parse all the point to add the observations on each point
   for (size_t aKPt = 0 ; aKPt<mVPtsMod.size() ; aKPt++)
   {
	 cPt2dr aPMod = mVPtsMod[aKPt];  // point of model
	 cPt2dr aPIm  = aCurHomM2I.Value(aPMod); // image of aPMod by current homotethy
	 bool isOk =      mInterpol                                 ?
                      mDIm.InsideInterpolator(*mInterpol,aPIm,0.0)  :
                      mDIm.InsideBL(aPIm)                           ;

	 if (isOk)  // avoid error 
	 {
	     // put observations in vectors
	          //  observations on image and point-image
             if (mInterpol)  
                 FormalGradInterpol_SetObs(aVObs,0,aPIm,mDIm,*mInterpol);  
	     else
                 FormalBilinIm2D_SetObs(aVObs,0,aPIm,mDIm);  

	          //  observation point model and value model
	     aVObs[aNbObsIm+0] = aPMod.x();
	     aVObs[aNbObsIm+1] = aPMod.y();
	     aVObs[aNbObsIm+2] =  mValueMod[aKPt];

	     // Now add observation
	     mSys->CalcAndAddObs(mEqHomIm,aVecInd,aVObs);

	      // compute indicator
	     tREAL8 aVRef =   mInterpol ? mDIm.GetValueInterpol(*mInterpol,aPIm) : mDIm.GetVBL(aPIm) ;
	     double aDif =  aVRef  - (aCurTrR+aCurScR*mValueMod[aKPt]); // residual
	     aSomMod += mValueMod[aKPt];
	     aSomDif += std::abs(aDif);
	 }
	 else
            aNbOut ++;
   }
   //   Update all parameter taking into account previous observation
   mSys->SolveUpdateReset();

   if (mShow)
      StdOut() << " DifImageModele=" << aSomDif / aSomMod << " NbOut=" << aNbOut << " TargRes=" << mTargetResidual  << std::endl;
   //  If we are at end, check that the model is equal (up to numerical accuracy)  to the target 
   bool Ok = (aNbOut==0) && ((aSomDif/aSomMod) < mTargetResidual) ;
   if (IsLast)
   {
       MMVII_INTERNAL_ASSERT_bench(aNbOut==0,"Gauss compos");
       MMVII_INTERNAL_ASSERT_bench( (aSomDif/aSomMod) < mTargetResidual ,"Gauss compos");
   }
   return Ok;
}


/********************************************/
/*              ::MMVII                     */
/********************************************/

void BenchDeformIm(cParamExeBench & aParam)
{
    if (! aParam.NewBench("DeformIm")) return;

    for (int aK=0 ; aK<4 ; aK++)
    {
        cDiffInterpolator1D * anInterpol = nullptr;  // If K==0 -> no interpol, bi-linear mode
	tREAL8 aTargetResidual = 1e-10;  // residual vary with interpolator (pseudo seem to have accuracy ?)
        
        // if K!=0 alloc various interpolator using the allocator by name(to test it)
        {
	   if (aK==1)
              anInterpol = cDiffInterpolator1D::AllocFromNames({"Tabul","1000","Cubic","-0.5"});
	   else if (aK==2)
              anInterpol = cDiffInterpolator1D::AllocFromNames({"Tabul","10000","SinCApod","10.0","10.0"});
	   else if (aK==3)
	   {
              aTargetResidual = 1e-5;
              anInterpol = cDiffInterpolator1D::AllocFromNames({"Tabul","1000","MMVIIK","2.0"});
	   }
        }

        cTestDeformIm aTDI(100,0.15,aParam.Show(),anInterpol,aTargetResidual);
        int aNbIter = 20 ;
	bool Ok = false;
        for (int aKIter=0 ; (aKIter<aNbIter) && (!Ok) ; aKIter++)
        {
            Ok = aTDI.OneIterationFitModele(aKIter==(aNbIter-1));
        }
    }

    aParam.EndBench();
}


};   // namespace MMVII
