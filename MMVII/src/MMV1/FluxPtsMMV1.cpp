#include "V1VII.h"

#include "MMVII_Geom2D.h"

/** \file  FluxPtsMMV1.cpp
    \brief file for using MMV1 flux; converting them a vect of point

    MMV1 has a very powerfull and elegant image processing toolbox (maybe I am not 100% objective ;-)
    MMVII will probably not have this kind of library (at leats in first version) as it  quite
    complicated to maintain and understand. By the way,  for many filter as long as I do not know
    exactly what I want, it's much faster to implement them with MMV1. 
*/

namespace MMVII
{

class cConfig_Freeman_Or;
class cCurveBySet;
class cCircle_CurveBySet;
class cEllipse_CurveBySetc;
class cLine_CurveBySet;
class cDilateSetPoints;

/* ********************************************************* */
/*                                                           */
/*                    cConfig_Freeman_Or                     */
/*                                                           */
/* ********************************************************* */

class cConfig_Freeman_Or
{
   public :
      typedef size_t tIndex;

      cConfig_Freeman_Or(bool v8,bool trigo);
      const cPt2di & KPt(tIndex aK) const   { return mVPts.at(aK);}
      size_t NbPts() const {return mNbPts;}
      tIndex   IndSucc(tIndex aK) const {return mSucc.at(aK);}
      tIndex   IndPred(tIndex aK) const {return mPred.at(aK);}
      tIndex   IndSym(tIndex aK) const  {return mSym.at(aK);}
      bool Trigo() const {return mTrigo;}

      tIndex  Pt2Ind(const cPt2di & aPt) const
      {
            return  const_cast<cConfig_Freeman_Or*>(this)->AddrP2I(aPt);
      }

      ///  Such that =1 between 2 successive neighbour
      int  NormNeighSucc(const cPt2di &) const;

      ///  Such that =1 for each point
      int  NormSphere(const cPt2di &) const;

      /// return aK  such that aPt is between   P(K)  and P(K+1)
      size_t  IndexEnglob(const cPt2dr &) const;

       void Bench() const;

   private :
      tIndex & AddrP2I(const cPt2di & aPt) {return  mPt2Ind.at(aPt.x()+1).at(aPt.y()+1);}

      bool                               mV8;
      bool                               mTrigo;
      int                                mSignOri ;
      size_t                             mNbPts;
      tIndex                             mIndex00;
      const cPt2di *                     mRawPts;
      std::vector<cPt2di>                mVPts;
      std::vector<tIndex>                mSucc;
      std::vector<tIndex>                mPred;
      std::vector<tIndex>                mSym;
      std::vector<std::vector<size_t>>   mPt2Ind;
};

cConfig_Freeman_Or::cConfig_Freeman_Or(bool v8,bool trigo) :
  mV8      (v8),
  mTrigo   (trigo),
  mSignOri (mTrigo ? 1 : -1),
  mNbPts   (mV8 ? 8 : 4),
  mIndex00 (mNbPts),
  mRawPts  (mV8 ? &FreemanV8[0] :  &FreemanV4[0] ),
  mVPts    (mRawPts,mRawPts+mNbPts),
  mSucc    (mNbPts),
  mPred    (mNbPts),
  mSym     (mNbPts),
  mPt2Ind  (3,std::vector<size_t>(3))
{
    AddrP2I(cPt2di(0,0)) = mIndex00;

    for (size_t aK=0 ; aK<mNbPts ; aK++)
    {
        AddrP2I(mVPts.at(aK)) = aK;
        mSucc[aK] = (aK+1) % mNbPts;
        mPred[(aK+1) % mNbPts ] = aK;
    }
    for (size_t aK=0 ; aK<mNbPts ; aK++)
        mSym.at(aK) = AddrP2I(-KPt(aK));

    if (!mTrigo)
    {
         for (size_t aK=1 ; aK<mNbPts-(aK) ; aK++)
         {
              std::swap(mVPts.at(aK),mVPts.at(mNbPts-aK));
         }
    }
}

int  cConfig_Freeman_Or::NormNeighSucc(const cPt2di & aP) const
{
    return mV8 ? Norm1(aP) : NormInf(aP);
}
int  cConfig_Freeman_Or::NormSphere(const cPt2di & aP) const
{
    return mV8 ? NormInf(aP) : Norm1(aP);
}

size_t  cConfig_Freeman_Or::IndexEnglob(const cPt2dr &aP) const
{
    MMVII_INTERNAL_ASSERT_tiny(mTrigo,"Have to decide what is the meaning of IndexEnglob for anti-trigo");

    for (size_t aK=0 ; aK<mNbPts ; aK++)
        if ( ((ToR(mVPts.at(aK))^aP)>=0) && ((ToR(mVPts.at(IndSucc(aK)))^aP)<0)  )
           return aK;

    MMVII_INTERNAL_ERROR("cConfig_Freeman_Or::IndexEnglob");
    return mNbPts;
}

void cConfig_Freeman_Or::Bench() const
{
    // StdOut() << " cConfig_Freeman_Or::bbBench " << mVPts << "  A="<< DbleAreaPolygOriented(mVPts) << "\n";
    MMVII_INTERNAL_ASSERT_bench(DbleAreaPolygOriented(mVPts)== mSignOri * (int)mNbPts ,"Norm in cConfig_Freeman_Or::Bench");
    for (size_t aK=0 ; aK<mNbPts ; aK++)
    {
        cPt2di aP1 =  KPt(aK);
        cPt2di aP2 =  KPt(IndSucc(aK));
        MMVII_INTERNAL_ASSERT_bench(NormNeighSucc(aP1-aP2)==1,"Norm in cConfig_Freeman_Or::Bench");
        MMVII_INTERNAL_ASSERT_bench(NormSphere(aP1)==1,"Norm in cConfig_Freeman_Or::Bench");

        if (mTrigo)
        {
            MMVII_INTERNAL_ASSERT_bench( IndexEnglob(ToR(aP1))==aK,"Norm in cConfig_Freeman_Or::Bench");
            MMVII_INTERNAL_ASSERT_bench( IndexEnglob(ToR(aP1+aP2))==aK,"Norm in cConfig_Freeman_Or::Bench");
        }
    }
}


/* ********************************************************* */
/*                                                           */
/*                    cCurveBySet                            */
/*                                                           */
/* ********************************************************* */

/** Base class for computing digital curves as frontiers of set defined by it level */

class cCurveBySet
{
     public :
         cCurveBySet(cPt2di  aP0,size_t  aKDir0,bool EightDir,bool Trigo,std::optional<cPt2di>);
	 virtual std::vector<cPt2di>  Compute();

         void Bench(const std::vector<cPt2di> &,tREAL8 aDMin,tREAL8 aDMax) const;
     protected :
         /// Correc Direction if not coherent to reach the frontier
         void  CorrecSenseDir();

         /// Research a pair or point where we reach transition outside/inside
         void ResearchFrontier();

         void InitFirstDir();

	 // convention InsideNess <0 when we are inside the bounded set
	 //  like "D(C)-R" for a circle of center C and ray R
         virtual tREAL8  InsideNess(const cPt2di & aPt) const = 0;

         /**  If defined, is more sophisticated version of InsideNess, +- equal to signed euclidean distance
              to frontier, can be slower, used in bench (so optional in some way); default error
         */
         virtual tREAL8  EuclidInsideNess(const cPt2di & aPt) const ;

         cPt2di  mP0;
         size_t  mKDir0;
         std::optional<cPt2di> mEndPoint;
         bool                  mClosed;
         cConfig_Freeman_Or mCFO;
         cPt2di  mDir0;
         cPt2di  mP1;
         tREAL8  mIn0;
         tREAL8  mIn1;

         cPt2di mCurPt;
         size_t mDir2Pred;
};

cCurveBySet::cCurveBySet(cPt2di  aPGuess0,size_t aKDir0,bool EightDir,bool Trigo,std::optional<cPt2di> aEndPoint) :
   mP0       (aPGuess0),
   mKDir0    (aKDir0),
   mEndPoint (aEndPoint),
   mClosed   (!mEndPoint.has_value()),
   mCFO      (EightDir,Trigo),
   mDir0     (mCFO.KPt(aKDir0)),
   mP1       (mP0+mDir0)
{
}

void cCurveBySet::Bench(const std::vector<cPt2di> & aVPts,tREAL8 aDMin,tREAL8 aDMax) const
{
    static tREAL8 aMaxD = 0;
    size_t aNbPC = aVPts.size();
    for (size_t aK=0 ; aK<aVPts.size()  ; aK++)
    {
        if (mClosed || (aK+1<aNbPC))
        {
             cPt2di aP0 = aVPts.at(aK);
             cPt2di aP1 = aVPts.at((aK+1)%aNbPC);
             tREAL8 aEI = EuclidInsideNess(aP0);
             UpdateMax(aMaxD,aEI);
             MMVII_INTERNAL_ASSERT_bench(mCFO.NormSphere(aP0-aP1)==1,"Bad consecutive points in cCurveBySet");
             MMVII_INTERNAL_ASSERT_bench(aEI>=aDMin,"Point inside domain in cCurveBySet::Bench");
             // Not sure to be able to establish theorericall bound, put empiricall one ..
             MMVII_INTERNAL_ASSERT_bench(aEI<=aDMax,"Point to far from domain in cCurveBySet::Bench");
        }
    }

    if (mClosed)
       MMVII_INTERNAL_ASSERT_bench((DbleAreaPolygOriented(aVPts)>0)==mCFO.Trigo() ,"Badly oriented");
}

tREAL8  cCurveBySet::EuclidInsideNess(const cPt2di & aPt) const 
{
    return InsideNess(aPt);
}



void cCurveBySet::CorrecSenseDir()
{
     // are we looking in the good direction for searching (i.e if we are inside with In<0 , are we growing , and vice-versa)
     bool aGoodDir = ((mIn0<0) == (mIn1>mIn0)) && (mIn0!=mIn1);

     // if not , invert direction
     if (! aGoodDir)
     {
	mDir0 = -mDir0;
        mKDir0 = mCFO.IndSym(mKDir0);
	mP1 = mP0 + mDir0;
	mIn1 = InsideNess(mP1);
    
        // if does not vary, we dont know what to do ...
        MMVII_INTERNAL_ASSERT_tiny(mIn0!=mIn1,"InsideNess not varying in cCurveBySet");
     }
}

void cCurveBySet::ResearchFrontier()
{
     int aCpt=0;

     // advance untill we cross the frontier
     while ((mIn0>=0) == (mIn1>=0))
     {
           // Curent point is set to next point
	   mP0 = mP1;
	   mIn0 = mIn1;
           // next point is incremented
	   mP1 = mP1 + mDir0;
           mIn1 = InsideNess(mP1);
           // avoid infinite search,  threshold purely empirical
	   aCpt++;
           MMVII_INTERNAL_ASSERT_tiny(aCpt<1e6,"Cannot find frontier in cCurveBySet::Compute");
           MMVII_INTERNAL_ASSERT_tiny(mIn0!=mIn1,"InsideNess not varying in cCurveBySet");
     }
}

void cCurveBySet::InitFirstDir()
{
     // Point must be (conventionnaly) on side >0
     if (mIn0>0)
     {
         mCurPt  = mP0;    
         mDir2Pred = mKDir0;
     }
     else
     {
         mCurPt  = mP1;
         mDir2Pred = mCFO.IndSym(mKDir0); 
     }

     // Dir prec goes for now inside, we turn in the reverse sens, untill we are outside
     size_t aCpt=0;
     while (InsideNess(mCurPt+mCFO.KPt(mDir2Pred)) < 0)
     {
          mDir2Pred = mCFO.IndPred(mDir2Pred);
          aCpt++;
          MMVII_INTERNAL_ASSERT_tiny(aCpt <= mCFO.NbPts(),"Cannot find frontier in cCurveBySet::Compute");
     }
}

std::vector<cPt2di> cCurveBySet::Compute()
{
     mIn0 = InsideNess(mP0);
     mIn1 = InsideNess(mP1);

     CorrecSenseDir();
     ResearchFrontier();

     InitFirstDir();

     std::vector<cPt2di>  aVPts;
     // if end point was not specified -> set it to begining point, its a closed curve
     if (mClosed)
     {
        mEndPoint = mCurPt;
     }
     else  // else add first point, as it wont be added at end
        aVPts.push_back(mCurPt);

     int aCptPts=0;
     bool  goOn  = true;
     while (goOn)
     {
         aCptPts++;
         MMVII_INTERNAL_ASSERT_tiny(aCptPts<1e6,"Cannot stop iteration cCurveBySet::Compute");

         size_t aCptDir=0;
         while (InsideNess(mCurPt+mCFO.KPt(mDir2Pred)) >= 0)
         {
              mDir2Pred = mCFO.IndSucc(mDir2Pred);
              aCptDir++;
              MMVII_INTERNAL_ASSERT_tiny(aCptDir <= mCFO.NbPts(),"Cannot find frontier in cCurveBySet::Compute");
         }
         mDir2Pred = mCFO.IndPred(mDir2Pred);
         mCurPt = mCurPt+mCFO.KPt(mDir2Pred);
	 aVPts.push_back(mCurPt);
	 goOn = (mCurPt != mEndPoint.value());
     }

     return aVPts;
}



/* ********************************************************* */
/*                                                           */
/*                    cCircle_CurveBySetc                    */
/*                                                           */
/* ********************************************************* */

/** Class for tracing a digital circle */

class cCircle_CurveBySet : public cCurveBySet
{
     public :
       cCircle_CurveBySet(const cPt2dr & aC,tREAL8 aRay,bool V8,bool Trigo) :
            cCurveBySet (ToI(aC),0,V8,Trigo,std::optional<cPt2di>()),
            mC          (aC),
            mRay        (aRay),
            mR2         (Square(mRay))
       {
       }

       tREAL8  InsideNess(const cPt2di & aPt) const override 
       {
	        // return Norm2(ToR(aPt)-mC) - mRay;
	        return SqN2(ToR(aPt)-mC) - mR2;
       }
       tREAL8  EuclidInsideNess(const cPt2di & aPt) const override 
       {
	        return Norm2(ToR(aPt)-mC) - mRay;
	        // return SqN2(ToR(aPt)-mC) - mR2;
       }
     private :
       cPt2dr  mC;
       tREAL8  mRay;
       tREAL8  mR2;
};

/* ********************************************************* */
/*                                                           */
/*                    cEllipse_CurveBySetc                   */
/*                                                           */
/* ********************************************************* */

/** Class for tracing a digital ellipse */

class cEllipse_CurveBySetc : public cCurveBySet
{
     public :
       cEllipse_CurveBySetc(const cPt2dr & aC,tREAL8 aRayA,tREAL8 aRayB,double aTeta,bool V8,bool Trigo) :
            cCurveBySet (ToI(aC),0,V8,Trigo,std::optional<cPt2di>()),
            mEllipse    (aC,aTeta,aRayA,aRayB)
       {
       }

       tREAL8  InsideNess(const cPt2di & aPt) const override 
       {
	        return mEllipse.SignedQF_D2(ToR(aPt));
       }
       tREAL8  EuclidInsideNess(const cPt2di & aPt) const override 
       {
	        return mEllipse.SignedEuclidDist(ToR(aPt));
	        // return SqN2(ToR(aPt)-mC) - mR2;
       }
     private :
      cEllipse  mEllipse;
};


/* ********************************************************* */
/*                                                           */
/*                    cLine_CurveBySet                       */
/*                                                           */
/* ********************************************************* */

/** Class for tracing a digital ellipse */


class  cLine_CurveBySet : public cCurveBySet
{
     public :
        cLine_CurveBySet(const cPt2di & aP1,const cPt2di & aP2) :
            cCurveBySet (aP1,0,true,true,aP2),
            mSeg        (ToR(aP1),ToR(aP2))
       {
       }
       std::vector<cPt2di>  Compute() override;

       tREAL8  InsideNess(const cPt2di & aPt) const override 
       {
          return mSeg.ToCoordLoc(ToR(aPt)).y();
       }
     private :
       cSegment2DCompiled<tREAL8>   mSeg;
};


std::vector<cPt2di>  cLine_CurveBySet::Compute() 
{
    // A standard bresenham algorithm
    std::vector<cPt2di> aRes;

    size_t aK1 = mCFO.IndexEnglob(mSeg.V12());
    size_t aK2 = mCFO.IndSucc(aK1);

    cPt2di aDP1 = mCFO.KPt(aK1);
    cPt2di aDP2 = mCFO.KPt(aK2);

    tREAL8 aD1 = Scal(ToR(aDP1),mSeg.Normal());
    tREAL8 aD2 = Scal(ToR(aDP2),mSeg.Normal());

    mCurPt = ToI(mSeg.P1());
    aRes.push_back(mCurPt);
   
    tREAL8 aSignD = 0;

    cPt2di aTarget = ToI(mSeg.P2());


    int aCpt = 1  + round_up(NormInf(mSeg.V12()));

    while (mCurPt != aTarget)
    {
         if (std::abs(aSignD+aD1) < std::abs(aSignD+aD2))
         {
             mCurPt += aDP1;
             aSignD += aD1;
         }
         else
         {
             mCurPt += aDP2;
             aSignD += aD2;
         }
         aRes.push_back(mCurPt);

         MMVII_INTERNAL_ASSERT_tiny(aCpt!=0,"Bresenham infinite loop");
         aCpt--;
    }
    return aRes;
}

/* ********************************************************* */
/*                                                           */
/*                    DilateSetPoints                        */
/*                                                           */
/* ********************************************************* */


class cDilateSetPoints
{
    public :
         cDilateSetPoints(tREAL8 aRay);
         const std::vector<cPt2di> &  Disk0() const {return mDisk0;}
         const std::vector<cPt2di> &  Trans(const cPt2di &) const ;
         void  DilateSet(std::vector<cPt2di> & aRes,const std::vector<cPt2di> & aSet0);
         
         tREAL8 Ray() const { return mRay;}
    private :
         tREAL8                             mRay;
         tREAL8                             mRAR;
         std::vector<cPt2di>                mDisk0;
         mutable std::vector<std::vector<cPt2di>>   mTrans;
         cConfig_Freeman_Or                 mCFO;
};

cDilateSetPoints::cDilateSetPoints(tREAL8 aRay) :
    mRay    (aRay),
    mRAR     (aRay * std::abs(aRay)),
    mTrans  (8),
    mCFO    (true,true)
{
     for (const auto & aPix : cRect2::BoxWindow(round_up(aRay)))
     {
         if (SqN2(aPix) <= mRAR)
            mDisk0.push_back(aPix);
     }
}

const std::vector<cPt2di> &  cDilateSetPoints::Trans(const cPt2di & aDelta) const 
{
    std::vector<cPt2di> & aRes = mTrans.at(mCFO.Pt2Ind(aDelta));

    if (aRes.empty())
    {
         for (const auto & aPix : mDisk0)
         {
             if (SqN2(aPix+aDelta) > mRAR)
                aRes.push_back(aPix+aDelta);
         }
    }

    return aRes;
}

void  cDilateSetPoints::DilateSet(std::vector<cPt2di> &aRes,const std::vector<cPt2di> & aSet0)
{
     aRes.clear();
     if (aSet0.empty())
        return ;

     cPt2di aP0 = aSet0.at(0);
     for (const auto & aPix : mDisk0)
            aRes.push_back(aP0+aPix);

     for (size_t aKP=1 ; aKP<aSet0.size() ; aKP++)
     {
         cPt2di aPPrec = aSet0.at(aKP-1);
         for (const auto & aDelta : Trans( (aSet0.at(aKP)-aPPrec) )    )
             aRes.push_back(aPPrec+aDelta);
     }
}


/* ********************************************************* */
/*                                                           */
/*                    MMVII::                                */
/*                                                           */
/* ********************************************************* */

void ShowCurveBySet(cCurveBySet & aSet,tREAL8 aRay,const std::string & aName)
{
     StdOut() << "=============== ShowCurveBySet : " << aName << " ================================\n";
     auto aVPts = aSet.Compute();
     auto aVDil = aVPts;
     if (aRay>=0)
     {
         cDilateSetPoints aDil(aRay);
         aDil.DilateSet(aVDil,aVPts);
     }
     cBox2di aBox = cBox2di::FromVect(aVDil).Dilate(5);

     cRGBImage aIm(aBox.Sz(),cRGBImage::White);

     for (const auto aPix : aVDil)
        aIm.SetRGBPix(aPix-aBox.P0(),cRGBImage::Red);

     for (const auto aPix : aVPts)
        aIm.SetRGBPix(aPix-aBox.P0(),cRGBImage::Black);

     aIm.ToFile("TEST-DIGICURVE-"+aName);
}

/*
        static tBox FromVect(const std::vector<tPt> & aVecPt,bool AllowEmpty=false);

*/

void BenchCircle_CurveDigit(cPt2dr  aC,double aRayA,double aRayB,double aTeta)
{
     for (bool Trigo : {false,true})
     {
         cCircle_CurveBySet aCCS(aC,aRayA,true,Trigo);
         auto  aVPts = aCCS.Compute();
         aCCS.Bench(aVPts,0,1);

         cEllipse_CurveBySetc aECS(aC,aRayA,aRayB,aTeta,true,Trigo);
         aVPts = aECS.Compute();
         aECS.Bench(aVPts,0,1);
     }
}

void BenchCurveDigit(cParamExeBench & aParam)
{
     if (! aParam.NewBench("CurveDigit")) return;

     cConfig_Freeman_Or(true ,true ).Bench();
     cConfig_Freeman_Or(false,true ).Bench();
     cConfig_Freeman_Or(true ,false).Bench();
     cConfig_Freeman_Or(false,false).Bench();


     if (0)
     {
         cLine_CurveBySet aLCS(cPt2di(-11,12),cPt2di(21,-6));
         ShowCurveBySet(aLCS,1.0,"Line_1.tif");
         ShowCurveBySet(aLCS,2.0,"Line_2.tif");
         ShowCurveBySet(aLCS,5.0,"Line_5.tif");

         cEllipse_CurveBySetc anElCS(cPt2dr(20.5,-10.3),40.0,15.0,0.3,true,true);
         ShowCurveBySet(anElCS,3.0,"El_3.tif");
     }
     for (int aKC=0 ; aKC<200 ; aKC++)
     {
         cPt2di aP1 = ToI(cPt2dr::PRandC() * 100.0);
         cPt2di aP2 = aP1;
         while (aP2==aP1)
               aP2 = ToI(cPt2dr::PRandC() * 100.0);

        cLine_CurveBySet aLCS(aP1,aP2);
        auto aVPts = aLCS.Compute();
        aLCS.Bench(aVPts,-0.5,0.5);
     }

     for (int aKC=0 ; aKC<30 ; aKC++)
     {
	 cPt2dr aC = cPt2dr::PRandC() * 1000.0 ;
         tREAL8 aRayA = RandInInterval(3.0,20.0);
         tREAL8 aRayB = RandInInterval(3.0,20.0);
         tREAL8 aTeta = RandInInterval(-100,100);

	 BenchCircle_CurveDigit(aC,aRayA,aRayB,aTeta);
     }

     aParam.EndBench();
}



/*************************************************/

#if (MMVII_KEEP_LIBRARY_MMV1)
Neighbourhood DiscNeich(const tREAL8 & aRay)
{
    std::vector<Pt2di> aVPts;
    int aNb = round_up(aRay);

    for (int aKx=-aNb ; aKx<=aNb ; aKx++)
    {
        for (int aKy=-aNb ; aKy<=aNb ; aKy++)
        {
            if ((aKx*aKx+aKy*aKy)<= aRay*aRay)
               aVPts.push_back(Pt2di(aKx,aKy));
        }
    }

    return Neighbourhood(&(aVPts[0]),int(aVPts.size()));
}

// typedef std::vector<cPt2di> tResFlux;


void  FluxToV2Points(tResFlux & aRes,Flux_Pts aFlux)
{
//StdOut() << "FluxToV2PointsFluxToV2PointsFluxToV2PointsuuuuuuuuuuuuFluxToV2PointsFluxToV2PointsFluxToV2PointsFluxToV2PointsFluxToV2Points\n";
//getchar();


    aRes.clear();
    Liste_Pts<int,int> aLV1(2);

    ELISE_COPY (aFlux, 1, aLV1);
    Im2D_INT4 aImV1 = aLV1.image();

    int  aNbPts   = aImV1.tx();
    int * aDataX  = aImV1.data()[0];
    int * aDataY  = aImV1.data()[1];

    for (int aKp=0 ; aKp<aNbPts ; aKp++)
    {
         aRes.push_back(cPt2di(aDataX[aKp],aDataY[aKp]));
    }
}

void  GetPts_Circle(tResFlux & aRes,const cPt2dr & aC,double aRay,bool with8Neigh)
{
     FluxToV2Points(aRes,circle(ToMMV1(aC),aRay,with8Neigh));
}

tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh)
{
     tResFlux aRes;
     GetPts_Circle(aRes,aC,aRay,with8Neigh);

     return aRes;
}


void  GetPts_Ellipse(tResFlux & aRes,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh)
{
     Flux_Pts aFlux = ellipse(ToMMV1(aC),aRayA,aRayB,aTeta,with8Neigh);
     FluxToV2Points(aRes,aFlux);
}


void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2,tREAL8 aDil)
{
     Flux_Pts aFlux = line(ToMMV1(ToI(aP1)),ToMMV1(ToI(aP2)));
     if (aDil>0)
        aFlux = dilate(aFlux,DiscNeich(aDil));
     FluxToV2Points(aRes,aFlux);
}

void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2)
{
	GetPts_Line(aRes,aP1,aP2,-1.0);
}
#else //  (MMVII_KEEP_LIBRARY_MMV1)
void  GetPts_Line(tResFlux & aResLine,const cPt2dr & aP1,const cPt2dr &aP2)
{
      cLine_CurveBySet aLCS(ToI(aP1),ToI(aP2));
      aResLine = aLCS.Compute();
}

void  GetPts_Line(tResFlux & aResDilatLine,const cPt2dr & aP1,const cPt2dr &aP2,tREAL8 aRayDilate)
{
    tResFlux aThinLine;
    GetPts_Line(aThinLine,aP1,aP2);

    cDilateSetPoints aDil(aRayDilate);
    aDil.DilateSet(aResDilatLine,aThinLine);
}

void  GetPts_Ellipse(tResFlux & aResEllipse,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh)
{
   cEllipse_CurveBySetc anElCS(aC,aRayA,aRayB,aTeta,with8Neigh,true);
   aResEllipse = anElCS.Compute();
}

tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh)
{
    cCircle_CurveBySet aCCS(aC,aRay,with8Neigh,true);
    return aCCS.Compute();
}

#endif //  (MMVII_KEEP_LIBRARY_MMV1)



};




