
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include<map>

/** \file 
    \brief 
*/


namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_TestGraphPart                             */
/*                                                      */
/* ==================================================== */


/** Application for concatenating videos */

class cAppli_TestGraphPart : public cMMVII_Appli
{
     public :
        cAppli_TestGraphPart(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
	 typedef cDenseMatrix<tREAL8> tMat;
	 typedef cDataIm2D<tREAL8>    tDIm;

         size_t  mNbVertex;
         size_t  mNbClass;
         cPt2dr  mProbaEr;
         cIm1D<tINT4>       mGTClass;
         cDataIm1D<tINT4>*  mDGTC;

	 tMat  mMat0;

};



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

       mDGTC->SetV(aKv,aClass);
   }

   mMat0  = tMat(mNbVertex,mNbVertex);
   tDIm & aDIm = mMat0.DIm();


   for (const auto & aPix : aDIm)
   {
       if (aPix.x() >= aPix.y())
       {
            size_t aCx = mDGTC->GetV(aPix.x());
            size_t aCy = mDGTC->GetV(aPix.y());

	    bool Value = (aCx==aCy);
	    double aProbaFalse = Value ? mProbaEr.x() : mProbaEr.y() ;

	    if (RandUnif_0_1() < aProbaFalse)
		    Value = !Value;

	    aDIm.SetV(aPix,Value);
       }
   }
   mMat0.SelfSymetrizeBottom();
   mMat0.DIm().ToFile("MatrInit.tif");

   mMat0 = mMat0 * mMat0 * (1.0/ tREAL8(mNbVertex)) ;
   mMat0.DIm().ToFile("Mat2.tif");

   cResulSymEigenValue<tREAL8> aRSE = mMat0.SymEigenValue();
   for (int aK=0 ; aK<10 ; aK++)
	   StdOut() << " * EV=" << aRSE.EigenValues()(mNbVertex-aK-1) << "\n";


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

