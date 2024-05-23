#include "MMVII_Image2D.h"
#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Mappings.h"


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*              cAppli_StackDep                          */
/*                                                      */
/* ==================================================== */

class cOneDepOfStack
{
    public :
       typedef cIm2D<tREAL4> tIm;
       typedef cDataIm2D<tREAL4> tDIm;

       cOneDepOfStack
       (
	   int aK1,int aK2,
	   cIm2D<tU_INT1>      aMasqNoDepl,
           const std::string & aP1,
           const std::string & aP2,
           const std::string & aScore,
	   cDiffInterpolator1D * anInterp
       );

       int    mK1;
       int    mK2;
       tIm    mImPx1;
       tDIm & mDImPx1;
       tIm    mImPx2;
       tDIm & mDImPx2;
       tIm    mImScore;
       tDIm & mDImScore;
       cTabulatMap2D_Id<tREAL4> mMap;

};

cOneDepOfStack::cOneDepOfStack
(
     int aK1,
     int aK2,
     cIm2D<tU_INT1>      aMasqNoDepl,
     const std::string & aP1,
     const std::string & aP2,
     const std::string & aScore,
     cDiffInterpolator1D * anInterp
) :
    mK1       (aK1),
    mK2       (aK2),
    mImPx1    (tIm::FromFile(aP1)),
    mDImPx1   (mImPx1.DIm()),
    mImPx2    (tIm::FromFile(aP2)),
    mDImPx2   (mImPx2.DIm()),
    mImScore  (tIm::FromFile(aScore)),
    mDImScore (mImScore.DIm()),
    mMap      (mImPx1,mImPx2,anInterp)
{
     mDImPx1.AssertSameArea(aMasqNoDepl.DIm());
     mDImPx2.AssertSameArea(aMasqNoDepl.DIm());
     mDImScore.AssertSameArea(aMasqNoDepl.DIm());

     // preprocessing average =0 
     {
         cWeightAv<tREAL8,cPt2dr> aAvgDep;
         const auto & aDMask = aMasqNoDepl.DIm();
         for (const auto &  aPix : aDMask)
         {
              if (aDMask.GetV(aPix))
	      {
                  aAvgDep.Add(1.0,cPt2dr(mDImPx1.GetV(aPix),mDImPx2.GetV(aPix)));
	      }
         }
	 cPt2dr aAvg = aAvgDep.Average();

         for (const auto &  aPix : aDMask)
         {
             mDImPx1.AddVal(aPix,-aAvg.x());
             mDImPx2.AddVal(aPix,-aAvg.y());
         }
     }
}




class cAppli_StackDep : public cMMVII_Appli
{
     public :

        typedef cIm2D<tREAL4> tIm;
        typedef cDataIm2D<tREAL4> tDIm;

        cAppli_StackDep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
	int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :

	std::string NameIm(int aK1,int aK2,const std::string & aPost) const;
	std::string NamePx1(int aK1,int aK2) const;
	std::string NamePx2(int aK1,int aK2) const;
	std::string NameScore(int aK1,int aK2) const;

	void Do1Pixel(const cPt2di & aPix);
        void   AddObs1Disp(const cPt2dr & aDisp,int aK);
	void InitPairs();


        std::vector<std::string>      mArgInterpol;
        cDiffInterpolator1D *         mInterpol;
	bool                          mDoL2;
	cLinearOverCstrSys<tREAL8>*   mSys;
        int                           mNbIm ;
        int                           mNbVar ;
        int                           mKRef ;
	std::string                   mSpecImIn;
	std::vector<cOneDepOfStack*>  mVecDepl;
	std::vector<cOneDepOfStack*>  mVecFromRef;
	std::vector<cOneDepOfStack*>  mVecToRef;


	std::string                   mFilePairs;
	t2MapStrInt                   mMapS2Im;
	cBijectiveMapI2O<cPt2di>      mMapCple2Match;

        std::vector<int>  mX_NumUk;
        std::vector<int>  mY_NumUk;

	std::vector<tIm> mImSol;

};

cAppli_StackDep::cAppli_StackDep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec):
	cMMVII_Appli   (aVArgs,aSpec),
	mNbIm          (10)
{
}

cCollecSpecArg2007 & cAppli_StackDep::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return anArgObl
              << Arg2007(mArgInterpol,"Argument interpolator ")
              << Arg2007(mFilePairs,"File for pairs of images")
              << Arg2007(mDoL2,"L2/L1 compensation")
           ;
}

cCollecSpecArg2007 & cAppli_StackDep::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return     anArgOpt
    ;
}


std::string cAppli_StackDep::NameIm(int aK1,int aK2,const std::string & aPost) const
{
        int aNum = mMapCple2Match.Obj2I(cPt2di(aK1,aK2));
	return  ToStr(aNum,2) + "_" + aPost + ".tif";
}

std::string cAppli_StackDep::NamePx1(int aK1,int aK2) const {return NameIm(aK1,aK2,"px1");}
std::string cAppli_StackDep::NamePx2(int aK1,int aK2) const {return NameIm(aK1,aK2,"px2");}
std::string cAppli_StackDep::NameScore(int aK1,int aK2) const {return NameIm(aK1,aK2,"corrscore");}


void   cAppli_StackDep::AddObs1Disp(const cPt2dr & aDisp,int aKDepl)
{
    {
         cSparseVect<tREAL8> aSVX;
         aSVX.AddIV(mX_NumUk.at(aKDepl),1.0);
         mSys->PublicAddObservation(1.0,aSVX,aDisp.x());
    }
    {
         cSparseVect<tREAL8> aSVY;
         aSVY.AddIV(mY_NumUk.at(aKDepl),1.0);
         mSys->PublicAddObservation(1.0,aSVY,aDisp.y());
    }
}



void  cAppli_StackDep::Do1Pixel(const cPt2di & aPix)
{
    mSys->Reset();

    for (const auto &  aDepl : mVecDepl )
    {
        if (aDepl->mK1==mKRef)
	{
           cPt2dr aDisp = aDepl->mMap.Value(ToR(aPix)) - ToR(aPix);
           AddObs1Disp(aDisp,aDepl->mK2);
	}
	else if (aDepl->mK2==mKRef)
	{
            cPt2dr aDisp = aDepl->mMap.Inverse(ToR(aPix)) -ToR(aPix);
            AddObs1Disp(aDisp,aDepl->mK1);
	}
	else
	{
             cOneDepOfStack * aDispK1 =   mVecFromRef.at(aDepl->mK1);
	     cPt2dr aPix1 = aDispK1->mMap.Value(ToR(aPix));
	     cPt2dr aPix2 = aDepl->mMap.Value(aPix1);
              // FakeUseIt(aPix2);
              AddObs1Disp(aPix2-ToR(aPix),aDepl->mK2);
	}
    }

    auto aSol = mSys->Solve();

    for (int aKV=0 ;  aKV<mNbVar ; aKV++)
    {
        mImSol.at(aKV).DIm().SetV(aPix,aSol(aKV));
    }
    mSys->Reset();
}


void cAppli_StackDep::InitPairs()
{
    if (! IsInit(&mFilePairs))  
       return;

    cReadFilesStruct aRFS(mFilePairs,"SSS",0,-1,'#');
    aRFS.Read();

    for (size_t aKL=0 ; aKL<aRFS.VStrings().size() ; aKL++)
    {
        const auto & aLine  = aRFS.VStrings().at(aKL);
	const auto & aN1 = aLine.at(1);
	const auto & aN2 = aLine.at(2);
        mMapS2Im.Add(aN1,true);  // true : OK  Exist
        mMapS2Im.Add(aN2,true);  // true : OK  Exist
				 
	int aI1 = mMapS2Im.Obj2I(aN1);
	int aI2 = mMapS2Im.Obj2I(aN2);

	mMapCple2Match.Add(cPt2di(aI1,aI2));
    }
    mNbIm =  mMapS2Im.size();
}


int cAppli_StackDep::Exe()
{
    mInterpol = cDiffInterpolator1D::AllocFromNames(mArgInterpol);
    InitPairs();
    cIm2D<tU_INT1> aMasq = cIm2D<tU_INT1>::FromFile("mask.tif");

    mVecFromRef.resize(mNbIm,nullptr);
    mVecToRef.resize(mNbIm,nullptr);
    mNbVar = 2*(mNbIm - 1);
    mKRef = mNbIm /2 ;

    for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
    {
         int aK0 = 2*aKIm;
	 if (aKIm==mKRef)
            aK0 = -2; // rubbish
	 else if (aKIm>mKRef)
            aK0 -=2 ; // skip one number for KREF

	 mX_NumUk.push_back(aK0);
	 mY_NumUk.push_back(aK0+1);
	 
	 if (aKIm!=mKRef)
	 {
             mImSol.push_back(tIm(aMasq.DIm().Sz()));
             mImSol.push_back(tIm(aMasq.DIm().Sz()));
	 }
    }

    mSys = mDoL2 ? new cLeasSqtAA<tREAL8>(mNbVar) : AllocL1_Barrodale<tREAL8>(mNbVar);

    for (int aK1=0 ; aK1<mNbIm ; aK1++)
    {
        for (int aK2=0 ; aK2<mNbIm ; aK2++)
        {
            if (aK1!=aK2)
            {
               bool  Ok = false;
               if (ExistFile(NamePx1(aK1,aK2)) && ExistFile(NamePx2(aK1,aK2)) && ExistFile(NameScore(aK1,aK2)))
               {
                  cOneDepOfStack * aDepl = new cOneDepOfStack(aK1,aK2,aMasq,NamePx1(aK1,aK2),NamePx2(aK1,aK2) ,NameScore(aK1,aK2),mInterpol);
		  mVecDepl.push_back(aDepl);
		  Ok = true;
		  if (aK1==mKRef)
                       mVecFromRef.at(aK2) = aDepl;
		  if (aK2==mKRef)
                       mVecToRef.at(aK1) = aDepl;
               }
	       if (aK1==mKRef)
	       { 
		   //  && (aK1!=aK2))
		    StdOut() << " OK " << aK1 << " " << aK2 << " " << Ok << "\n";

	       }
            }
        }
    }


    for (const auto & aPix : aMasq.DIm())
        Do1Pixel(aPix);

    for (int aKS=0 ; aKS<mNbIm-1 ; aKS++)
    {
        int aKI = aKS;
	if (aKS>=mKRef)
            aKI++;

	std::string aName = "Merge_" + ToStr(mKRef) + "_to_" +  ToStr(aKI) ;
        mImSol.at(2*aKS).DIm().ToFile(aName+"_x.tif");
        mImSol.at(2*aKS+1).DIm().ToFile(aName+"_y.tif");
    }

    delete mInterpol;
    delete mSys;
    DeleteAllAndClear(mVecDepl);
    return EXIT_SUCCESS;
};




/* ==================================================== */
/*                                                      */
/*                                                      */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_StackDep(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_StackDep(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_StackDep
(
     "DeplStack",
      Alloc_StackDep,
      "Stack a serie multi date displacment",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);



}; // MMVII




