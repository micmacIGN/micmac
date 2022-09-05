#include "include/MMVII_all.h"
#include "include/MMVII_TplSymbImage.h"

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
         cTestDeformIm(int aSzGlob,double aEps,bool Show);
         ~cTestDeformIm();
         /// Make one iteration of adding of non linear least square
	 void OneIterationFitModele(bool IsLast);
     private :

	 bool               mShow; // print result, export image ...
         int                mSzGlob; ///< variable used for dimensionning the rest
         double             mAmplRad;  ///<  amplitude of radiometry

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

cTestDeformIm::cTestDeformIm(int aSzGlob,double aEps,bool Show) :
   mShow       ( Show),
   mSzGlob     ( aSzGlob),  //  geometric value are proportional to aSzGlob (or w/o dimension)
   mAmplRad    (255.0),     //  radiometric value are proportional to mAmplRad (or w/o dimension)
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
         cPt2dr aPMod = mGT_I2Mod.Value(ToR(aPixIm));
	 double aVModele = mGaussModel.Val(aPMod)*mAmplRad;

	 if (mDIm.Interiority(aPixIm)>3)
	 {
            mVPtsMod.push_back(aPMod);
            mValueMod.push_back(aVModele);
	 }

         mDIm.SetV(aPixIm, mTrRad + mScRad*aVModele);

      // little check on gaussian composed with homotethy
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
    mEqHomIm = EqDeformImHomotethy(true,1);

}

cTestDeformIm::~cTestDeformIm() 
{
    delete mSys;
    delete mEqHomIm;
}

void cTestDeformIm::OneIterationFitModele(bool IsLast)
{
   //----------- index of unkown, basic here because the unknown are the same for each equation
   std::vector<int>  aVecInd{0,1,2,3,4};
   //----------- allocate vec of obs : 6 for image, 3 for model
   std::vector<double> aVObs(9,0.0);

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
	 if (mDIm.InsideBL(aPIm))  // avoid error 
	 {
	     // put observations in vectors
	          //  observations on image and point-image
             FormalBilinIm2D_SetObs(aVObs,0,aPIm,mDIm);  

	          //  observation point model and value model
	     aVObs[6] = aPMod.x();
	     aVObs[7] = aPMod.y();
	     aVObs[8] =  mValueMod[aKPt];

	     // Now add observation
	     mSys->CalcAndAddObs(mEqHomIm,aVecInd,aVObs);

	      // compute indicator
	     double aDif = mDIm.GetVBL(aPIm) - (aCurTrR+aCurScR*mValueMod[aKPt]);
	     aSomMod += mValueMod[aKPt];
	     aSomDif += std::abs(aDif);
	 }
	 else
            aNbOut ++;
   }
   //   Update all parameter taking into account previous observation
   mSys->SolveUpdateReset();

   if (mShow)
      StdOut() << " Dif=" << aSomDif / aSomMod << " NbOut=" << aNbOut  << "\n";
   //  If we are at end, check that the model is equal (up to numerical accuracy)  to the target 
   if (IsLast)
   {
       MMVII_INTERNAL_ASSERT_bench(aNbOut==0,"Gauss compos");
       MMVII_INTERNAL_ASSERT_bench( (aSomDif/aSomMod) < 1e-10 ,"Gauss compos");
   }
}


/********************************************/
/*              ::MMVII                     */
/********************************************/

void BenchDeformIm(cParamExeBench & aParam)
{
    if (! aParam.NewBench("DeformIm")) return;

    for (int aK=0 ; aK<1 ; aK++)
    {
        cTestDeformIm aTDI(100,0.15,aParam.Show());
        int aNbIter = 8 ;
        for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
        {
            aTDI.OneIterationFitModele(aKIter==(aNbIter-1));
        }
    }

    aParam.EndBench();
}


};   // namespace MMVII
