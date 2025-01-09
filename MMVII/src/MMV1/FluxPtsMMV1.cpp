#define WITH_MMV1_FUNCTION false

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

      const tIndex  Pt2Ind(const cPt2di & aPt) const
      {
            return  const_cast<cConfig_Freeman_Or*>(this)->AddrP2I(aPt);
      }
/*
      inline INT    succ(INT k) const   { return _succ[k];}
      inline INT    prec(INT k) const   { return _prec[k];}
      inline INT     sym(INT k) const   { return _sym[k];}
      inline INT    nb_pts()    const   { return _nb_pts;}

      inline INT  num_pts(Pt2di p)  const
                  {return compute_freem_code(*_mat_code,p);}
*/

   private :
      tIndex & AddrP2I(const cPt2di & aPt) {return  mPt2Ind.at(aPt.x()+1).at(aPt.y()+1);}

      bool                               mV8;
      bool                               mTrigo;
      size_t                             mNbPts;
      tIndex                             mIndex00;
      const cPt2di *                     mRawPts;
      std::vector<cPt2di>                mVPts;
      std::vector<tIndex>                mSucc;
      std::vector<tIndex>                mPred;
      std::vector<tIndex>                mSym;
      std::vector<std::vector<size_t>>   mPt2Ind;
/*
      INT             * _succ ;
      INT             * _prec ;
      INT             * _sym ;
      MAT_CODE_FREEM  * _mat_code;
*/
};


cConfig_Freeman_Or::cConfig_Freeman_Or(bool v8,bool trigo) :
  mV8      (v8),
  mTrigo   (trigo),
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
    if (mTrigo)
       std::swap(mSucc,mPred);
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
	 std::vector<cPt2di>  Compute();

     private :
         /// Correc Direction if not coherent to reach the frontier
         void  CorrecSenseDir();

         /// Research a pair or point where we reach transition outside/inside
         void ResearchFrontier();

         void InitFirstDir();

	 // convention InsideNess <0 when we are inside the bounded set
	 //  like "D(C)-R" for a circle of center C and ray R
         virtual tREAL8  InsideNess(const cPt2di & aPt) const = 0;

         cPt2di  mP0;
         size_t  mKDir0;
         cConfig_Freeman_Or mCFO;
         cPt2di  mDir0;
         cPt2di  mP1;
         tREAL8  mIn0;
         tREAL8  mIn1;

         cPt2di mCurPt;
         size_t mDirPrec;
};
/*
     cPt2di aDir0 = aCFO.KPt(aKDir0);
     cPt2di aP0 = aPGuess0;
     cPt2di aP1 = aP0 + aDir0;
*/

cCurveBySet::cCurveBySet(cPt2di  aPGuess0,size_t aKDir0,bool EightDir,bool Trigo,std::optional<cPt2di> aEndPoint) :
   mP0       (aPGuess0),
   mKDir0    (aKDir0),
   mCFO      (EightDir,Trigo),
   mDir0     (mCFO.KPt(aKDir0)),
   mP1       (mP0+mDir0)
{
}

/*
INT Flux_By_Contour::k_next_pts(INT *x,INT *y,INT nb,bool & end)
{
    end = false;
    ASSERT_INTERNAL(_init,"INIT probk in Flux_By_Contour");
    if (_first)
    {
         _first = false;
         while (inside(_p_cur+_freem.kth_p(_k_prec)))
           _k_prec = _freem.succ(_k_prec);
    }
   for (INT i=0; i<nb ; i++)
    {
        _nb_max--;
        x[i] = _p_cur.x;
        y[i] = _p_cur.y;
        // count  turn for special cas of 4 neigh (when all 4 neigh are outside)
        INT nb;
        for
        (
            nb =0;
            (nb<_freem.nb_pts()) && (!inside(_p_cur+_freem.kth_p(_k_prec)));
            nb++
        )
            _k_prec = _freem.succ(_k_prec);
        _k_prec =  _freem.prec(_k_prec);
        _p_cur = _p_cur+_freem.kth_p(_k_prec);
        _k_prec =  _freem.sym(_k_prec);
        if ((_p_cur == _p_end) || (! _nb_max))
        {
           end = true;
           return (i+1);
        }
    }
    return nb;
}
*/


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
     if (mIn0>0)
     {
         mCurPt  = mP0;
         mDirPrec = mCFO.IndSym(mKDir0);
     }
     else
     {
         mCurPt  = mP1;
         mDirPrec = mKDir0;
     }

     size_t aCpt=0;
     while (InsideNess(mCurPt+mCFO.KPt(mDirPrec)) < 0)
     {
          mDirPrec = mCFO.IndPred(mDirPrec);
          aCpt++;
          MMVII_INTERNAL_ASSERT_tiny(aCpt < mCFO.NbPts(),"Cannot find frontier in cCurveBySet::Compute");
     }
}

std::vector<cPt2di> cCurveBySet::Compute()
{
     mIn0 = InsideNess(mP0);
     mIn1 = InsideNess(mP1);

     CorrecSenseDir();
     ResearchFrontier();

     InitFirstDir();
#if (0)


     // we follow (conventionnaly) the exterior frontier
     cPt2di aCurPt  = (aIn0>=0) ? aP0 : aP1;
     // if end point was not specified -> set it to begining point
     if (!aEndPoint.has_value())
        aEndPoint = aCurPt;
     // initialize adequate neighbourhood
     const cPt2di* aVNeigh = (EightDir ? &FreemanV8[0] :  &FreemanV4[0] );
     size_t aNbN = EightDir ? 8 : 4;

     int aCpt=0;
     bool  goOn  = true;
     while (goOn)
     {
         aCpt++;
         MMVII_INTERNAL_ASSERT_tiny(aCpt<1e6,"Cannot stop iteration cCurveBySet::Compute");
	 // try to follow the surface with the condition interior (<0) to the left
         std::vector<size_t> aVK;
StdOut() << "CurPt" << aCurPt << " " << InsideNess(aCurPt) << "\n";
         for (size_t aK=0 ; aK<aNbN ; aK++)
         {
             cPt2di aP0 = aCurPt + aVNeigh[aK] ;
             cPt2di aP1 = aCurPt + aVNeigh[(aK+1)%aNbN] ;
StdOut() << "KKK " << aK <<  " I0=" << InsideNess(aP0) << " I1=" << InsideNess(aP1) << "\n";
	     if ( (InsideNess(aP0)>=0) && (InsideNess(aP1)<0) )  
	     {
StdOut() << "*****************\n";
                 aVK.push_back(aK);
	     }
         }
         StdOut() << "=======================================  Vk=" << aVK << "\n";
	 // if we have no point, or multiple point, we have a problem ...
         MMVII_INTERNAL_ASSERT_tiny(aVK.size()== 1,"Topolgical problem in cCurveBySet::Compute");
StdOut() << "KKKKKK " << aVK  << " "<< aVNeigh[aVK.at(0)] << "\n";

	 aVPts.push_back(aCurPt);
	 aCurPt  = aCurPt + aVNeigh[aVK.at(0)];
	 goOn = (aCurPt != aEndPoint);
     }

#endif
     std::vector<cPt2di>  aVPts;
     return aVPts;
}

/* ********************************************************* */
/*                                                           */
/*                    cCircle_CurveBySetc                    */
/*                                                           */
/* ********************************************************* */

/** Base class for computing digital curves as frontiers of set defined by it level */

class cCircle_CurveBySet : public cCurveBySet
{
     public :
       cCircle_CurveBySet(const cPt2dr & aC,tREAL8 aRay) :
            cCurveBySet (ToI(mC),0,true,true,std::optional<cPt2di>()),
            mC          (aC),
            mRay        (aRay),
            mR2         (Square(mRay))
	{
	}

/*
cCurveBySet::cCurveBySet(cPt2di  aPGuess0,size_t aKDir0,bool EightDir,std::optional<cPt2di> aEndPoint)
       std::vector<cPt2di>  Compute(bool EightDir)
       {
            return cCurveBySet::Compute(ToI(mC),0,EightDir,std::optional<cPt2di>());
       }
*/
       tREAL8  InsideNess(const cPt2di & aPt) const override 
       {
	        return Norm2(ToR(aPt)-mC) - mRay;
	        // return SqN2(ToR(aPt)-mC) - mR2;
       }
     private :
       cPt2dr  mC;
       tREAL8  mRay;
       tREAL8  mR2;
};


void BenchCircle_CurveDigit(cPt2dr  aC,double aRay)
{

/*
cPt2di aDec = cPt2di(5,5) +ToI(cPt2dr(aRay,aRay));
aC = aC - ToR(ToI(aC)) + ToR(aDec);
     cCircle_CurveBySet aCCS(aC,aRay);

cIm2D<tU_INT1> aIm(2*aDec);
for (const auto & aPt : aIm.DIm())
{
   aIm.DIm().SetV(aPt,255*(aCCS.InsideNess(aPt)>0));
}
aIm.DIm().ToFile("CurveDigit.tif");

     for (const auto aV8 : {true})
     {
StdOut() << "------------------------------------------------------------------------------------------------\n";
StdOut() << "JJJJJJJJJJ  " << aC << " " << aRay << " " << aV8 << " Dec=" << aDec << "\n";
         aCCS.Compute(aV8);
     }
*/
}

void BenchCurveDigit(cParamExeBench & aParam)
{
     if (! aParam.NewBench("CurveDigit")) return;

     for (int aKC=0 ; aKC<30 ; aKC++)
     {
	 cPt2dr aC = cPt2dr::PRandC() * 1000.0 ;
         tREAL8 aRay = RandInInterval(3.0,20.0);

	 BenchCircle_CurveDigit(aC,aRay);
     }

     aParam.EndBench();
}



/*************************************************/

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



};




