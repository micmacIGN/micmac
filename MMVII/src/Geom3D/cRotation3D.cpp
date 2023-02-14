
#include "MMVII_Tpl_Images.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"

namespace MMVII
{

// template <class Type> cSimilitud3D(cSegment


//template <class TypeElem,class TypeMap> cIsometry3D<Type><Type> FromTriOut(const TypeMap & )

/* ************************************************* */
/*                                                   */
/*               cSimilitud3D<Type>                  */
/*                                                   */
/* ************************************************* */

template <class Type> cSimilitud3D<Type>::cSimilitud3D(const Type& aScale,const tPt& aTr,const cRotation3D<Type> & aRot) :
    mScale (aScale),
    mTr    (aTr),
    mRot   (aRot)
{
}



// tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt)*mScale;}
// mRot.Inverse((aPt-mTr)/mScale)

template <class Type> cSimilitud3D<Type> cSimilitud3D<Type>::operator * (const tTypeMap & aS2) const
{
	// mTr + Sc*R (mTr2 +Sc2*R2*aP)
	return tTypeMap(mScale*aS2.mScale  ,  mTr+ mRot.Value(aS2.mTr)*mScale ,    mRot*aS2.mRot);
}

template <class Type> cSimilitud3D<Type> cSimilitud3D<Type>::MapInverse() const
{
    return tTypeMap
	   (
	          Type(1.0) / mScale,
		 -mRot.Inverse(mTr)/mScale,
		  mRot.MapInverse()
	   );
}

template <class Type> cSimilitud3D<Type> cSimilitud3D<Type>::FromScaleRotAndInOut
                      (const Type& aScale,const cRotation3D<Type> & aRot,const tPt& aPtIn,const tPt& aPtOut )
{
    return tTypeMap
	   (
	          aScale,
		  aPtOut - aRot.Value(aPtIn)*aScale,
		  aRot
	   );
}


template <class Type> cSimilitud3D<Type> cSimilitud3D<Type>::FromTriOut(int aKOut,const tTri  & aTriOut)
{
    tPt aV0 = aTriOut.KVect(aKOut);
    tPt aV1 = aTriOut.KVect((aKOut+1)%3);

    tTypeMap aRes
	   (
		   Norm2(aV0),
	           aTriOut.Pt(aKOut),
                   cRotation3D<Type>::CompleteRON(aV0,aV1)
	   );

    return aRes;
}

template <class Type> cSimilitud3D<Type> 
    cSimilitud3D<Type>::FromTriInAndSeg(const tPt2&aP1,const tPt2&aP2,int aKIn,const tTri  & aTriIn)
{
    // mapping that send Seg(K,K+1)  on (0,0)->(0,1)
    cSimilitud3D<Type> anIs = FromTriOut(aKIn,aTriIn).MapInverse();

    //return anIs;
    cSim2D<Type> aS2D =  cSim2D<Type>::FromMinimalSamples(cPtxd<Type,2>(0,0),cPtxd<Type,2>(1,0),aP1,aP2);

    return aS2D.Ext3D()* anIs;
    /*
    */
}

template <class Type> cSimilitud3D<Type> cSimilitud3D<Type>::FromTriInAndOut
                        (int aKIn,const tTri  & aTriIn,int aKOut,const tTri  & aTriOut)
{
     tTypeMap aRefToOut = FromTriOut(aKOut,aTriOut);
     tTypeMap aInToRef  = FromTriOut(aKIn,aTriIn).MapInverse();

     return aRefToOut * aInToRef;
}



/* ************************************************* */
/*                                                   */
/*               cIsometry3D<Type>                   */
/*                                                   */
/* ************************************************* */

template <class Type> cIsometry3D<Type>::cIsometry3D(const tPt& aTr,const cRotation3D<Type> & aRot) :
    mTr  (aTr),
    mRot (aRot)
{
}

template <class Type> void cIsometry3D<Type>::SetRotation(const cRotation3D<Type> & aRot)
{
    mRot = aRot;
}

template <class Type> cIsometry3D<Type>  cIsometry3D<Type>::Identity()
{
    return cIsometry3D<Type>(cPtxd<Type,3>(0,0,0),cRotation3D<Type>::Identity());
}

//  tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt);}

template <class Type> cIsometry3D<Type>  cIsometry3D<Type>::MapInverse() const
{
    return cIsometry3D<Type>(-mRot.Inverse(mTr),mRot.MapInverse());
}

template <class Type> cIsometry3D<Type> cIsometry3D<Type>::operator * (const tTypeMap & aS2) const
{
	// mTr + R (mTr2 +R2*aP)
	return tTypeMap(mTr+ mRot.Value(aS2.mTr),mRot*aS2.mRot);
}

//        tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt);}

template <class Type> 
         cIsometry3D<Type> cIsometry3D<Type>::FromRotAndInOut
	                    (const cRotation3D<Type> & aRot,const tPt& aPtIn,const tPt& aPtOut )
{
	return cIsometry3D<Type>(aPtOut-aRot.Value(aPtIn),aRot);
}


template <class Type> cIsometry3D<Type> cIsometry3D<Type>::FromTriOut(int aKOut,const tTri  & aTriOut,bool Direct)
{
    int Delta = Direct ? 1 : 2;
    tTypeMap aRes
	     (
	           aTriOut.Pt(aKOut),
                   cRotation3D<Type>::CompleteRON(aTriOut.KVect(aKOut),aTriOut.KVect((aKOut+Delta)%3))
	     );



    return aRes;
}

template <class Type> cTriangle<Type,2> cIsometry3D<Type>::ToPlaneZ0(int aKOut,const tTri  & aTriOut,bool Direct)
{
     cIsometry3D<Type> aIso = FromTriOut(aKOut,aTriOut,Direct);

     tPt aP0 = aIso.Inverse(aTriOut.Pt(aKOut));
     tPt aP1 = aIso.Inverse(aTriOut.Pt((aKOut+1)%3));
     tPt aP2 = aIso.Inverse(aTriOut.Pt((aKOut+2)%3));

      //StdOut() << "ToPlaneZ0 " << aP0.z() << " " << aP1.z() << " " << aP2.z() << "\n";
     // return tTri2d(Proj(aP0),Proj(aP1),Proj(aP2));
     return Proj(tTri(aP0,aP1,aP2));
}




template <class Type> cIsometry3D<Type> cIsometry3D<Type>::FromTriInAndOut
                        (int aKIn,const tTri  & aTriIn,int aKOut,const tTri  & aTriOut)
{
     tTypeMap aRefToOut = FromTriOut(aKOut,aTriOut);
     tTypeMap aInToRef  = FromTriOut(aKIn,aTriIn).MapInverse();

     return aRefToOut * aInToRef;
}

template <class Type> 
   cIsometry3D<Type> cIsometry3D<Type>::RandomIsom3D(const Type & AmplPt)
{
    return  cIsometry3D<Type> (tPt::PRandC()*AmplPt,cRotation3D<Type>::RandomRot());
}



/* ************************************************* */
/*                                                   */
/*               cRotation3D<Type>                   */
/*                                                   */
/* ************************************************* */

template <class Type> cRotation3D<Type>::cRotation3D(const cDenseMatrix<Type> & aMat,bool RefineIt) :
   mMat (aMat)
{
   MMVII_INTERNAL_ASSERT_always((! RefineIt),"Refine to write in Rotation ...");
}


template <class Type> cRotation3D<Type> cRotation3D<Type>::Identity()
{
    return cRotation3D<Type>(cDenseMatrix<Type>(3,eModeInitImage::eMIA_MatrixId),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::MapInverse() const 
{
    return cRotation3D(mMat.Transpose(),false);
}

template <class Type> cRotation3D<Type> cRotation3D<Type>::operator * (const tTypeMap & aS2) const
{
	// mTr + R (mTr2 +R2*aP)
	return tTypeMap(mMat*aS2.mMat,false);
}

template <class Type> cPtxd<Type,3> cRotation3D<Type>::AxeI() const  {return tPt::Col(mMat,0);}
template <class Type> cPtxd<Type,3> cRotation3D<Type>::AxeJ() const  {return tPt::Col(mMat,1);}
template <class Type> cPtxd<Type,3> cRotation3D<Type>::AxeK() const  {return tPt::Col(mMat,2);}


template <class Type> cRotation3D<Type>  cRotation3D<Type>::CompleteRON(const tPt & aP0Init)
{
    cPtxd<Type,3> aP0 = VUnit(aP0Init);
    cPtxd<Type,3> aP1 = VUnit(VOrthog(aP0));
    cPtxd<Type,3> aP2 = aP0 ^ aP1;

    return  cRotation3D<Type>(MatFromCols(aP0,aP1,aP2),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::CompleteRON(const tPt & aP0Init,const tPt & aP1Init)
{
    cPtxd<Type,3> aP0 = VUnit(aP0Init);
    cPtxd<Type,3> aP2 = VUnit(aP0 ^ aP1Init);
    cPtxd<Type,3> aP1 = aP2 ^ aP0;

    return  cRotation3D<Type>(MatFromCols(aP0,aP1,aP2),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromAxiator(const tPt & anAxe)
{
     Type aNorm = Norm2(anAxe);
     if (aNorm<1e-5)
     {
         cDenseMatrix<Type>  aMW =  cDenseMatrix<Type>::Identity(3) + MatProdVect(anAxe);

         return cRotation3D<Type>(aMW.ClosestOrthog(),false);
     }
     return RotFromAxe(anAxe/aNorm,aNorm);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromAxe(const tPt & aP0,Type aTeta)
{
   // Compute a repair with P0 as first axes
   cRotation3D<Type> aRepAxe = CompleteRON(aP0);
   // Extract two other axes
   tPt  aP1 = tPt::Col(aRepAxe.mMat,1);
   tPt  aP2 = tPt::Col(aRepAxe.mMat,2);

   Type aCosT = cos(aTeta); 
   Type aSinT = sin(aTeta);
   // In plane P1,P2 we have the classical formula of 2D rotation
   tPt aQ1 =   aCosT*aP1 + aSinT*aP2;
   tPt aQ2 =  -aSinT*aP1 + aCosT*aP2;
   //  Mat * (aP0,aP1,aP2) = (aP0,aQ1,aQ2)

   return cRotation3D<Type>(MatFromCols(aP0,aQ1,aQ2)* MatFromCols(aP0,aP1,aP2).Transpose(),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RandomRot()
{
   tPt aP0 = tPt::PRandUnit();
   tPt aP1 = tPt::PRandUnit();
   while(Cos(aP0,aP1)>0.99)
       aP1 = tPt::PRandUnit();
   return CompleteRON(aP0,aP1);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RandomRot(const Type & aAmpl)
{
	return RotFromAxiator(cPtxd<Type,3>::PRandC()*aAmpl);
}


template <class Type> void cRotation3D<Type>::ExtractAxe(tPt & anAxe,Type & aTeta)
{
    cDenseVect<Type> aDVAxe =  mMat.EigenVect(1.0);
    anAxe =  cPtxd<Type,3>::FromVect(aDVAxe);

    cRotation3D<Type> aRep = CompleteRON(anAxe);
    cPtxd<Type,3> aP1 = cPtxd<Type,3>::Col(aRep.mMat,1);
    cPtxd<Type,3> aP2 = cPtxd<Type,3>::Col(aRep.mMat,2);

    cPtxd<Type,3> aQ1 = Value(aP1);
    Type aCosT = Cos(aP1,aQ1);
    Type aSinT = Cos(aP2,aQ1);

    cPt2dr  aRhoTeta = ToPolar(cPt2dr(aCosT,aSinT));  // To change with templatized ToPolar when exist

    MMVII_INTERNAL_ASSERT_medium(std::abs(aRhoTeta.x()-1.0)<1e-5,"Axes from rot");
    aTeta = aRhoTeta.y();
}

/*    WPK = Rx(W) Ry(P) Rz(K)  
 *    YPR = Rz(Y) Ry(P) Rx(R)
 *
 *     M =  Rx(W) Ry(P) Rz(K) = 
 *
 *   YPR(y,p,r)   Rz(y) Ry(p) Rx(r)  = t( Rx(-r)  Ry(-p) Rz(-y) ) = t WPK(-r,-p,-y)
 *
 */


template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromYPR(const tPt & aYPR)
{
    return  RotFromWPK(tPt(-aYPR.z(),-aYPR.y(),-aYPR.x())).MapInverse();
}

template <class Type> cPtxd<Type,3>  cRotation3D<Type>::ToYPR() const
{
	tPt aWPK = MapInverse().ToWPK();

	return tPt(-aWPK.z(),-aWPK.y(),-aWPK.x());
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromWPK(const tPt & aWPK)
{
   return 
	    RotFromAxe(cPtxd<Type,3>(1,0,0),aWPK.x())
	  * RotFromAxe(cPtxd<Type,3>(0,1,0),aWPK.y())
	  * RotFromAxe(cPtxd<Type,3>(0,0,1),aWPK.z()) 
   ;
}


template <class Type> cPtxd<Type,3>  cRotation3D<Type>::ToWPK() const
{
    Type aSinPhi = std::min(Type(1.0),std::max(Type(-1.0),mMat(2,0)));
    Type aPhi = ASin(aSinPhi);
    Type aCosPhi = std::cos(aPhi);

    Type aKapa ,aOmega;
    if (std::abs(aCosPhi) < 1e-4)
    {

        // aSinPhi=1  mMat(0,1),mMat(1,1)    =      CosW SinK+SinW CosK  , cosW CosK-sinW sinK =   sin(W+K), cos(W+K)   
        // aSinPhi=-11  mMat(0,1),mMat(1,1)  =      CosW SinK-SinW CosK  , cosW CosK+sinW sinK =   sin(K-W), cos(K-W)   
        Type aCombin = std::atan2(mMat(0,1),mMat(1,1));
        if (aSinPhi>0)
	{
              aKapa = aCombin/2;
	      aOmega = aCombin/2;
	}
	else
	{
              aKapa = aCombin/2;
              aOmega = -aCombin/2;
	}
    }
    else
    {
       aKapa =  std::atan2(-mMat(1,0),mMat(0,0));
       aOmega = std::atan2(-mMat(2,1),mMat(2,2));
    }

    tPt aWPK(aOmega,aPhi,aKapa);

    // aWPK += tPt::PRandC()*Type(0.05);

    // now make some least square optim, do it we finite difference because I am a lazzy guy ;-)
    for (int aK=0 ; aK<1 ; aK++)
    {
        Type aEps = 1e-3;
	std::vector<cDenseMatrix<Type> > aVecDer;

	cLeasSqtAA<Type>  aSys(3);
	for (int aKCoord =0 ; aKCoord<3 ; aKCoord++)
	{
              tPt aWPK_p = aWPK;
              tPt aWPK_m = aWPK;
	      aWPK_p[aKCoord] += aEps;
	      aWPK_m[aKCoord] -= aEps;

	      cDenseMatrix<Type> aMat_p = RotFromWPK(aWPK_p).Mat();
	      cDenseMatrix<Type> aMat_m = RotFromWPK(aWPK_m).Mat();
	      aVecDer.push_back( (aMat_p-aMat_m) * Type(1.0/(2*aEps)));

	}
	aSys.AddObsFixVar(1e-4,0,0.0);  // add some constraint becaus if guimball dont want to have degenerate sys

	for (const auto & aPix : aVecDer[0].DIm())
	{
             cDenseVect<Type> aVCoef(3);
	     for (int aKCoord =0 ; aKCoord<3 ; aKCoord++)
                aVCoef(aKCoord) = aVecDer[aKCoord].GetElem(aPix);
	     aSys.AddObservation(1.0,aVCoef ,mMat.GetElem(aPix));

	}

	cDenseVect<Type> aSol = aSys.Solve();
	aWPK += tPt::FromVect(aSol);
    }

    return aWPK;
}


/*
    U D tV X =0   U0 t.q D(U0) = 0   , Ker => U0 = tV X,    X = V U0
*/


/* ========================== */
/*          ::                */
/* ========================== */


/*
*/
#define MACRO_INSTATIATE_PTXD(TYPE)\
template class  cSimilitud3D<TYPE>;\
template class  cIsometry3D<TYPE>;\
template class  cRotation3D<TYPE>;

/*
template  cRotation3D<TYPE>  cRotation3D<TYPE>::CompleteRON(const tPt & );\
template  cRotation3D<TYPE>  cRotation3D<TYPE>::CompleteRON(const tPt &,const tPt &);\
template  cRotation3D<TYPE>::cRotation3D(const cDenseMatrix<TYPE> & ,bool ) ;\
template  cRotation3D<TYPE>  cRotation3D<TYPE>::RotFromAxe(const tPt & ,TYPE );\
template  cRotation3D<TYPE> cRotation3D<TYPE>::RandomRot();
*/

/*
template <class Type> cRotation3D<Type>  cRotation3D<Type>::CompleteRON(const tPt & aP0Init)
*/


MACRO_INSTATIATE_PTXD(tREAL4)
MACRO_INSTATIATE_PTXD(tREAL8)
MACRO_INSTATIATE_PTXD(tREAL16)



};
