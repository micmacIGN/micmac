
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include<map>

/** \file 
    \brief 
*/


namespace MMVII
{

/*  5 boolean config tested as they are coherent with transitivity  :  111  100 010  001  000
 *
 *   Let be the current likehood of a triplet
 *       pA  pB  pC  
 *
 *   pA is the probability/likehood that edge A is "on"
 *
 *   For each of the five coherent config, we can estimate a likelihood making a indepannet assumption
 *
 *      L(111) =  pA pB pC
 *      L(100) =  pA (1-pB)  (1- pC)
 *
 *   We will compute first a global likelihood to be equal to 0 or 1 for each edge,
 *
 *   Then  will do for example
 *
 *      L(A=1) += L(100) 
 *      L(B=0) += L(100)
 *
 */

class cConfigCyclGP
{
    public :

       cConfigCyclGP(size_t aFlagBit);

	bool     In[3];
	bool     mOk;
	tREAL8   mWeight;
};

cConfigCyclGP::cConfigCyclGP(size_t aFlagBit)
{
     for (size_t aB=0 ; aB<3 ; aB++)
	In[aB]  =   ((aFlagBit & (1<<aB)) != 0) ;

     mOk = (NbBits(aFlagBit) != 2);

     mWeight = mOk ? 1.0 : (-5.0/3.0);
}

class cGraphPartition
{
     public :
	typedef cIm2D<tREAL8>        tIm;
	typedef cDataIm2D<tREAL8>    tDIm;

	cGraphPartition(tIm  aIm0);
     private :

	void MakeOneIteration(tREAL8 aAlpha);
	void AddOneTriplet(int aX,int aY,int aZ);

	cPt2di  mSz;
	size_t  mNb;

	tIm                           mGr0;
	tDIm  &                       mDGr0;

	tIm                           mICurGr;
	tDIm  &                       mDCurGr;
	tIm                           mILikH0;
	tDIm  &                       mDLikH0;
	tIm                           mILikH1;
	tDIm  &                       mDLikH1;
	std::vector<cConfigCyclGP>    mConfigs;
	tREAL8                        mSomNew;

};



cGraphPartition:: cGraphPartition(tIm  aIm0) :
    mSz      (aIm0.DIm().Sz()),
    mNb      (mSz.x()),
    mGr0     (aIm0.Dup()),
    mDGr0    (mGr0.DIm()),
    mICurGr  (aIm0.Dup()),
    mDCurGr  (mICurGr.DIm()),
    mILikH0  (mSz),
    mDLikH0  (mILikH0.DIm()),
    mILikH1  (mSz),
    mDLikH1  (mILikH1.DIm())
{
    for (size_t aFlag=0 ; aFlag<8 ; aFlag++)
        mConfigs.push_back(cConfigCyclGP(aFlag));


    int aNb=100;
    for (int aK=1 ; aK<= aNb ; aK++)
    {
         MakeOneIteration(aK/double(aNb));
	 StdOut() << "KKKK " << aK << "\n";
	 if (aK%10==0)
            mDCurGr.ToFile("MatRelax_" + ToStr(aK) + ".tif");
    }
}

void cGraphPartition::AddOneTriplet(int aX,int aY,int aZ)
{
    cPt2di   aEdges[3] = {{aX,aY},{aX,aZ},{aY,aZ}};
    tREAL8 mVal[3];

    mVal[0]  =  mDCurGr.GetV(aEdges[0]);
    mVal[1]  =  mDCurGr.GetV(aEdges[1]);
    mVal[2]  =  mDCurGr.GetV(aEdges[2]);

    for (const auto & aConfig : mConfigs)
    {
	tREAL8 aLikH =    (aConfig.In[0] ?     mVal[0] : (1-mVal[0])) 
	               *  (aConfig.In[1] ?     mVal[1] : (1-mVal[1])) 
	               *  (aConfig.In[2] ?     mVal[2] : (1-mVal[2]))  ;

	tREAL8 aModif = aConfig.mWeight * aLikH;

	for (size_t aK : {0,1,2})
	{
            tDIm & aLikIm = aConfig.In[aK] ? mDLikH1 :  mDLikH0;

	    aLikIm.AddVal(aEdges[aK],aModif);
	}
	mSomNew += std::abs(aModif);
    }
}

void cGraphPartition::MakeOneIteration(tREAL8 aAlpha)
{
	// Iterate all edges
     mDLikH0.InitNull();
     mDLikH1.InitNull();
     mSomNew = 0;
     for (size_t aX=0 ; aX<mNb ; aX++)
     {
         for (size_t aY=aX+1 ; aY<mNb ; aY++)
         {
             for (size_t aZ=aY+1 ; aZ<mNb ; aZ++)
             {
		     AddOneTriplet(aX,aY,aZ);
	     }
         }
     }

     // compute the dynamic input
     tREAL8 aSumCur=0;

     for (size_t aX=0 ; aX<mNb ; aX++)
     {
         for (size_t aY=aX+1 ; aY<mNb ; aY++)
	 {
             cPt2di aP(aX,aY);
             aSumCur += mDCurGr.GetV(aP);
	 }
     }

     tREAL8 aMul = aAlpha * ( aSumCur /mSomNew);

     ///StdOut() << "MMMM" << aMul  << " " << aSumCur << " " << mSomNew << "\n";

     // transferate the accumulated vals + reset in interval [0 1]
     for (size_t aX=0 ; aX<mNb ; aX++)
     {
         for (size_t aY=aX+1 ; aY<mNb ; aY++)
	 {
             cPt2di aP(aX,aY);
	     // StdOut()  << " AVV " <<   mDCurGr.GetV(aP)  << "\n";
             mDCurGr.AddVal(aP,aMul*(mDLikH1.GetV(aP)-mDLikH0.GetV(aP)));


	     tREAL8 aV = mDCurGr.GetV(aP);

	     if ((aV>0) && (aV<1))
		     StdOut()  << aP<< " " << aV << "\n";

	     mDCurGr.SetV(aP,std::max(0.0,std::min(1.0,aV)));
	     // StdOut()  << " APP " <<   mDCurGr.GetV(aP)  << "\n";
	     // StdOut()  << "Likkk " <<  aMul*(mDLikH1.GetV(aP)-mDLikH0.GetV(aP)) << "\n";
	 }
     }
     getchar();


     // symetrize
     for (size_t aX=0 ; aX<mNb ; aX++)
     {
         for (size_t aY=aX+1 ; aY<mNb ; aY++)
	 {
             mDCurGr.SetV(cPt2di(aY,aX),mDCurGr.GetV(cPt2di(aX,aY)));
	 }
     }
}



/* ==================================================== */
/*                                                      */
/*          cAppli_TestGraphPart                        */
/*                                                      */
/* ==================================================== */


/** Application for make some test on graph partitionning algorithms */

class cAppli_TestGraphPart : public cMMVII_Appli
{
     public :
	typedef cDenseMatrix<tREAL8> tMat;
	typedef cDataIm2D<tREAL8>    tDIm;
	typedef cDenseVect<tREAL8>   tVec;

        cAppli_TestGraphPart(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :

	 // return  E ^ alph M  using ~ (1+aM/N) ^ N  and with N= 2^ k have fast result
	 tMat  ExponentialMatrix(const tMat & aMat,tREAL8 aAlpha,int aK);


	 tMat  CosProd(const tMat &aM1,const tMat & aM2);
	 tVec  VecColNormLines(const tMat &aM1);
	 tVec  VecLineNormCol(const tMat &aM2);


         size_t  mNbVertex;
         size_t  mNbClass;
         cPt2dr  mProbaEr;
         cIm1D<tINT4>       mGTClass;
         cDataIm1D<tINT4>*  mDGTC;
	 tMat  mMat0;
};

/*                       x
 *                       c1
 * y   l1 l2 ...   *     c2
 *
 */

cAppli_TestGraphPart::tVec  cAppli_TestGraphPart::VecColNormLines(const tMat &aM1)
{
    size_t aSzY = aM1.Sz().y();
    tVec  aCol1(aSzY);
    for (size_t aKy=0 ; aKy< aSzY ; aKy++)
    {
        tVec  aLineY = aM1.ReadLine(aKy);
	aCol1(aKy) = aLineY.L2Norm();
    }
    return aCol1;
}

cAppli_TestGraphPart::tVec  cAppli_TestGraphPart::VecLineNormCol(const tMat &aM2)
{
    size_t aSzX = aM2.Sz().x();
    tVec  aLine2(aSzX);
    for (size_t aKx=0 ; aKx< aSzX ; aKx++)
    {
        tVec  aColX = aM2.ReadCol(aKx);
        aLine2(aKx) = aColX.L2Norm() ;
    }
    return aLine2;
}


cAppli_TestGraphPart::tMat  cAppli_TestGraphPart::CosProd(const tMat &aM1,const tMat & aM2)
{
    tMat aRes = aM1 * aM2;

    tVec aCol1 = VecColNormLines(aM1);
    tVec  aLine2 = VecLineNormCol(aM2);

    tDIm  & aDRes = aRes.DIm();

    for (const auto & aPix : aDRes)
    {
        aDRes.SetV(aPix,aDRes.GetV(aPix) / (mNbVertex*aCol1(aPix.y()) * aLine2(aPix.x())));
    }

    // cDense

    return aRes;
}

cAppli_TestGraphPart::tMat  cAppli_TestGraphPart::ExponentialMatrix(const tMat & aMat,tREAL8 aAlpha,int aK)
{
    tMat aRes = tMat::Identity(aMat.Sz().x()) +  aMat * (aAlpha / (1<<aK));

    while (aK)
    {
        aRes = aRes *aRes;
        aK--;
    }
    return aRes;
}


cCollecSpecArg2007 & cAppli_TestGraphPart::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
         << Arg2007(mNbVertex,"Number of vertex")

   ;
}

cCollecSpecArg2007 & cAppli_TestGraphPart::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mNbClass,"NbClass","Number of classes)",{eTA2007::HDV})
         << AOpt2007(mProbaEr,"ProbEr","Probability of error for same/diff classes )",{eTA2007::HDV})
   ;
}


cAppli_TestGraphPart::cAppli_TestGraphPart
(
      const std::vector<std::string> &  aVArgs,
      const cSpecMMVII_Appli & aSpec
) :
  cMMVII_Appli    (aVArgs,aSpec),
  mNbClass        (5),
  mProbaEr        (0.1,0.1),
  mGTClass        (1),
  mMat0           (1,1)
{
}

int cAppli_TestGraphPart::Exe()
{
   mGTClass =  cIm1D<tINT4>(mNbVertex);
   mDGTC = &(mGTClass.DIm());

   for (size_t aKv = 0 ; aKv < mNbVertex ; aKv++)
   {
       size_t  aClass = (aKv * mNbClass) / mNbVertex;

       // aClass = aKv % mNbClass;

       mDGTC->SetV(aKv,aClass);
   }

   mMat0  = tMat(mNbVertex,mNbVertex);
   tDIm & aDIm = mMat0.DIm();


   for (const auto & aPix : aDIm)
   {
       if (aPix.x() > aPix.y())
       {
            size_t aCx = mDGTC->GetV(aPix.x());
            size_t aCy = mDGTC->GetV(aPix.y());

	    bool Value = (aCx==aCy);
	    double aProbaFalse = Value ? mProbaEr.x() : mProbaEr.y() ;

	    if (RandUnif_0_1() < aProbaFalse)
		    Value = !Value;

	    aDIm.SetV(aPix,Value);
       }
       else if (aPix.x() == aPix.y())
          aDIm.SetV(aPix,1);
   }
   mMat0.SelfSymetrizeBottom();

   tMat aId = tMat::Identity(mMat0.Sz().x());

   // mMat0 = (CosProd(mMat0,aId) + CosProd(aId,mMat0)) * 0.5;

   mMat0.DIm().ToFile("MatrInit.tif");

   tMat aM2 = CosProd(mMat0,mMat0);
   aM2.DIm().ToFile("Mat2Cos.tif");

   tMat aM3 = CosProd(aM2,mMat0);
   aM3.DIm().ToFile("Mat3Cos.tif");

   tMat aM4 = CosProd(aM3,mMat0);
   aM4.DIm().ToFile("Mat4Cos.tif");


   //cGraphPartition aGP(mMat0.Im());


   cResulSymEigenValue<tREAL8> aRSE = mMat0.SymEigenValue();
   for (int aK=0 ; aK<10 ; aK++)
	   StdOut() << " * EV=" << aRSE.EigenValues()(mNbVertex-aK-1) << "\n";

   /*
   tMat mMat2 = mMat0 * mMat0 * (1.0/ tREAL8(mNbVertex)) ;
   mMat2.DIm().ToFile("Mat2.tif");

   tMat mMat3 = mMat2 * mMat0 * (1.0/ tREAL8(mNbVertex)) ;
   mMat3.DIm().ToFile("Mat3.tif");

   tMat mMat4 = mMat3 * mMat0 * (1.0/ tREAL8(mNbVertex)) ;
   mMat4.DIm().ToFile("Mat4.tif");


   ExponentialMatrix(mMat0,1.0,2).DIm().ToFile("MatExp1.tif");
   */

   // mMat0 = mMat0 * mMat0 * (1.0/ tREAL8(mNbVertex));
   // mMat0 = mMat0 * mMat0 * (1.0/ tREAL8(mNbVertex));
   // mMat0.DIm().ToFile("Mat4.tif");
   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestGraphPart(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TestGraphPart(aVArgs,aSpec));
}




cSpecMMVII_Appli  TheSpecTestGraphPart
(
     "TestGraphPart",
      Alloc_TestGraphPart,
      "This command is to make some test on graph partionning",
      {eApF::Perso},
      {eApDT::Console},
      {eApDT::Image},
      __FILE__
);

};

