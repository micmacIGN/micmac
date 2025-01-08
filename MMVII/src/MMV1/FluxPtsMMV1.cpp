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
      cConfig_Freeman_Or(bool v8,bool trigo);
/*
      inline Pt2di kth_p(INT k) const   { return _pts[k];}
      inline INT    succ(INT k) const   { return _succ[k];}
      inline INT    prec(INT k) const   { return _prec[k];}
      inline INT     sym(INT k) const   { return _sym[k];}
      inline INT    nb_pts()    const   { return _nb_pts;}

      inline INT  num_pts(Pt2di p)  const
                  {return compute_freem_code(*_mat_code,p);}
*/

   private :
      bool                 mV8;
      bool                 mTrigo;
      size_t               mNbPts;
      const cPt2di *       mRawPts;
      std::vector<cPt2di>  mVPts;
      std::vector<int>     mSucc;
      std::vector<int>     mPred;
/*
      INT             * _succ ;
      INT             * _prec ;
      INT             * _sym ;
      MAT_CODE_FREEM  * _mat_code;
*/
};


cConfig_Freeman_Or::cConfig_Freeman_Or(bool v8,bool trigo) :
  mV8     (v8),
  mTrigo  (trigo),
  mNbPts  (mV8 ? 8 : 4),
  mRawPts (mV8 ? &FreemanV8[0] :  &FreemanV4[0] ),
  mVPts   (mRawPts,mRawPts+mNbPts),
  mSucc   (mNbPts),
  mPred   (mNbPts)
{
     for (size_t aK=0 ; aK<mNbPts ; aK++)
     {
          mSucc[aK] = (aK+1) % mNbPts;
          mPred[(aK+1) % mNbPts ] = aK;
     }
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
         cCurveBySet();
	 std::vector<cPt2di>  Compute(cPt2di  aP0,cPt2di  aDir0,bool EightDir,std::optional<cPt2di>);
     private :
	 // convention InsideNess <0 when we are inside the bounded set
	 //  like "D(C)-R" for a circle of center C and ray R
         virtual tREAL8  InsideNess(const cPt2di & aPt) const = 0;

};

cCurveBySet::cCurveBySet()
{
}

std::vector<cPt2di> cCurveBySet::Compute(cPt2di  aPGuess0,cPt2di  aDir0,bool   EightDir,std::optional<cPt2di> aEndPoint)
{
     std::vector<cPt2di>  aVPts;

     cPt2di aP0 = aPGuess0;
     cPt2di aP1 = aP0 + aDir0;
     tREAL8 aIn0 = InsideNess(aP0);
     tREAL8 aIn1 = InsideNess(aP1);


     // are we looking in the good direction for searching
     bool aGoodDir = ((aIn0<0) == (aIn1>aIn0)) && (aIn0!=aIn1);

     // if not , invert direction
     if (! aGoodDir)
     {
	aDir0 = -aDir0;
	aP1 = aP0 + aDir0;
	aIn1 = InsideNess(aP1);
    
        // if does not vary, we dont know what to do ...
        MMVII_INTERNAL_ASSERT_tiny(aIn0!=aIn1,"InsideNess not varying in cCurveBySet");
     }

     // research for a transition of sign, when we "cross" the frontier
     {
        int aCpt=0;
// StdOut() << "INNN00 " << aIn0 << " "<< aIn1 << " " << aP0 << " " << aP1 << "\n";
        while ((aIn0>=0) == (aIn1>=0))
        {
	   aP0 = aP1;
	   aIn0 = aIn1;

	   aP1 = aP1 + aDir0;
           aIn1 = InsideNess(aP1);
	   aCpt++;
//StdOut() << "INNNKKK " << aIn0 << " "<< aIn1 << " " << aP0 << " " << aP1 << "\n"; getchar();
           MMVII_INTERNAL_ASSERT_tiny(aCpt<1e5,"Cannot find frontier in cCurveBySet::Compute");
        }
     }

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
            cCurveBySet (),
            mC          (aC),
            mRay        (aRay),
            mR2         (Square(mRay))
	{
	}

       std::vector<cPt2di>  Compute(bool EightDir)
       {
            return cCurveBySet::Compute(ToI(mC),cPt2di(0,1),EightDir,std::optional<cPt2di>());
       }
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




