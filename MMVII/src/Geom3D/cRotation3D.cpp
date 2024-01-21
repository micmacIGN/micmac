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

/*  As this method is provide on for serialization we initialize the rotation with null matrix so that
 *  any use drive quickly to absurd result if not error
 */
template <class Type> cIsometry3D<Type>::cIsometry3D() :
	cIsometry3D(tPt(0,0,0),cRotation3D<Type>(cDenseMatrix<Type>(3,3,eModeInitImage::eMIA_Null),false))
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

      //StdOut() << "ToPlaneZ0 " << aP0.z() << " " << aP1.z() << " " << aP2.z() << std::endl;
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

template <class Type> cIsometry3D<tREAL8>  ToReal8(const cIsometry3D<Type>  & anIsom)
{
    return cIsometry3D<tREAL8>(  ToR(anIsom.Tr()) , ToReal8(anIsom.Rot())  );
}

void AddData(const cAuxAr2007 & anAux,tPoseR & aPose)
{
     cPt3dr aC = aPose.Tr();
     cPt3dr aI = aPose.Rot().AxeI();
     cPt3dr aJ = aPose.Rot().AxeJ();
     cPt3dr aK = aPose.Rot().AxeK();
     MMVII::AddData(cAuxAr2007("Center",anAux),aC);

     {
         cAuxAr2007 aAuxRot("RotMatrix",anAux);
         MMVII::AddData(cAuxAr2007("AxeI",aAuxRot),aI);
         MMVII::AddData(cAuxAr2007("AxeJ",aAuxRot),aJ);
         MMVII::AddData(cAuxAr2007("AxeK",aAuxRot),aK);
     }
     if (anAux.Input())
     {
         aPose = tPoseR(aC,cRotation3D<tREAL8>(MatFromCols(aI,aJ,aK),false));
     }
}


/* ************************************************* */
/*                                                   */
/*               cRotation3D<Type>                   */
/*                                                   */
/* ************************************************* */

template <class Type> cRotation3D<Type>::cRotation3D(const cDenseMatrix<Type> & aMat,bool RefineIt) :
   mMat (aMat)
{
   if (RefineIt)
   {
      mMat = mMat.ClosestOrthog();
   }
   // MMVII_INTERNAL_ASSERT_always((! RefineIt),"Refine to write in Rotation ...");
}

template <class Type> cRotation3D<Type>::cRotation3D(const tPt &aI,const tPt & aJ,const tPt & aK,bool RefineIt) :
	cRotation3D<Type>(M3x3FromCol(aI,aJ,aK),RefineIt)
{
}


template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromCanonicalAxes(const std::string& aName)
{
    tPt aVAxes[3];
    size_t aIndChar = 0;
    size_t aIndAxe = 0;

    while ((aIndAxe<3) && (aIndChar<aName.size()))
    {
          aVAxes[aIndAxe++] = tPt::PFromCanonicalName(aName,aIndChar);
    }	    

    // at the end the string must have been consumed exactly
    MMVII_INTERNAL_ASSERT_tiny((aIndAxe==3) && (aIndChar==aName.size()),"Bad format in RotFromCanonicalAxes");
    MMVII_INTERNAL_ASSERT_tiny( NormInf((aVAxes[0]^aVAxes[1])-aVAxes[2]) ==0,"Not a direct repair in RotFromCanonicalAxes");

    return cRotation3D<Type>(aVAxes[0],aVAxes[1],aVAxes[2],false);
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

template <class Type> cDenseMatrix<Type>  cRotation3D<Type>::RotOmega(const tREAL8 & aOmega)
{
   Type aCx = std::cos(aOmega);
   Type aSx = std::sin(aOmega);
   return M3x3FromLines(tPt(1,0,0),tPt(0,aCx,-aSx),tPt(0,aSx,aCx));
}

template <class Type> cDenseMatrix<Type>  cRotation3D<Type>::RotPhi(const tREAL8 & aPhi)
{
   Type aCy = std::cos(aPhi);
   Type aSy = std::sin(aPhi);
   return M3x3FromLines(tPt(aCy,0,aSy),tPt(0,1,0),tPt(-aSy,0,aCy));
}

template <class Type> cDenseMatrix<Type>  cRotation3D<Type>::RotKappa(const tREAL8 & aKappa)
{
   Type aCz = std::cos(aKappa);
   Type aSz = std::sin(aKappa);
   return M3x3FromLines(tPt(aCz,-aSz,0),tPt(aSz,aCz,0),tPt(0,0,1));
}

template <class Type> cDenseMatrix<Type>  cRotation3D<Type>::Rot1WPK(int aK,const tREAL8 & aTeta)
{
    switch (aK)
    {
        case 0 : return RotOmega(aTeta);
        case 1 : return RotPhi(aTeta);
        case 2 : return RotKappa(aTeta);
    }
    MMVII_INTERNAL_ERROR("Bad value of K in Rot1WPK : " + ToStr(aK));
    return  cDenseMatrix<Type>(3);
}


template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromWPK(const tPt & aWPK)
{
   auto aRx = RotOmega(aWPK.x());
   auto aRy = RotPhi(aWPK.y());
   auto aRz = RotKappa(aWPK.z());

   return cRotation3D<Type>(aRx*aRy*aRz,false);
/*
   return 
	    cRotation3D<Type>(aRx,false)
	  * cRotation3D<Type>(aRy,false)
	  * cRotation3D<Type>(aRz,false)
   ;
    Old formulation, slower, but dont destroy can be used again in tuning/testing
   return 
	    RotFromAxe(cPtxd<Type,3>(1,0,0),aWPK.x())
	  * RotFromAxe(cPtxd<Type,3>(0,1,0),aWPK.y())
	  * RotFromAxe(cPtxd<Type,3>(0,0,1),aWPK.z()) 
   ;
   */
}

template <class Type> cRotation3D<tREAL8>  ToReal8(const cRotation3D<Type>  & aRot)
{
    cDenseMatrix<tREAL8>  aM8 =  Convert((tREAL8*)nullptr,aRot.Mat());
    return cRotation3D<tREAL8>(aM8,false);
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

    // now make some least square optim, do it we finite difference because I am a lazzy guy (;-)
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
	     aSys.PublicAddObservation(1.0,aVCoef ,mMat.GetElem(aPix));

	}

	cDenseVect<Type> aSol = aSys.Solve();
	aWPK += tPt::FromVect(aSol);
    }

    return aWPK;
}

/*
    U D tV X =0   U0 t.q D(U0) = 0   , Ker => U0 = tV X,    X = V U0
*/

/* ************************************************* */
/*                                                   */
/*               cSampleQuat                         */
/*                                                   */
/* ************************************************* */

cSampleQuat::cSampleQuat(int aNbStep,bool Is4R) :
   m4R     (Is4R),
   mNbF    (m4R ? 4 :8),
   mNbStep (aNbStep),
   mNbRot  (round_ni(mNbF*std::pow(mNbStep,3)))
{
}

cSampleQuat cSampleQuat::FromNbRot(int aNbRot,bool Is4R)
{
        return cSampleQuat(round_up (std::pow(aNbRot/(Is4R ? 4.0 : 8.0),1/3.0)),Is4R) ;
}

size_t cSampleQuat::NbRot() const { return mNbRot; }
size_t cSampleQuat::NbStep() const { return mNbStep; }

std::vector<cPt4dr>  cSampleQuat::VecAllQuat() const
{
    std::vector<cPt4dr> aRes(mNbRot);
    for (size_t aK=0; aK<mNbRot ; aK++)
            aRes[aK] = KthQuat(aK);

    return aRes;
}

tREAL8 cSampleQuat::Int2Coord(int aK) const
{
    //  the sampling must be regular of step 1/2NbStep and
    //  extrem value 0 & NbStep-1 must but a equal distance of -1 and 1
    //
    //               !! Bad computation if conversion to int is not forced for mNbStep
    return   (aK*2+1-(int)mNbStep) / double(mNbStep);
}

cPt4dr  cSampleQuat::KthQuat(int aK) const
{
   //  NbRot = 8 *  NbStep ^ 3
   cPt4dr  aRes;

   int aIndF = (aK%mNbF);  // index of face   +- ijkt or ijkt in rot mode
   int  aSign =  1 ;  // sign always 1 in rot mode
   if (! m4R)
   {
       aSign =  1 - 2 * (aIndF%2);  // sign ,is it + or -
       aIndF /= 2;  //  between 0 and 3, so ijkt
   }
   aRes[aIndF] = aSign;  // now fix to -1/+1 in the face

   int aIndXYZ  = aK/mNbF;  /// now we code the cube of 3 remaining direction
   aRes[(++aIndF)%4] = Int2Coord(aIndXYZ%mNbStep); // next coord value on "x"
   int aIndYZ  =aIndXYZ/mNbStep;  // now code the 2 remaining direction
   aRes[(++aIndF)%4] = Int2Coord(aIndYZ%mNbStep);  // next coord value on y
   // int aIndZ  =aIndYZ/mNbStep;                     //
   aRes[(++aIndF)%4] = Int2Coord(aIndYZ/mNbStep); // and Z

   return VUnit(aRes);
}

std::vector<tREAL8 >  cSampleQuat::TestVecMinDist(size_t aNbTest)  const
{
    std::vector<cPt4dr > aVPts;
    std::vector<cDenseMatrix<tREAL8> > aVRot;
    std::vector<tREAL8 >          aVDMin;

    // compute the point and the min dist, to limit the calls to KthQuat
    for (size_t aKTest=0 ; aKTest<aNbTest; aKTest++)
    {
        aVPts.push_back(cPt4dr::PRandUnit());
        aVRot.push_back(Quat2MatrRot(aVPts.back()));
        aVDMin.push_back(1e8);
    }

    for (size_t aKRot=0; aKRot < NbRot(); aKRot++) // parse all sampled rot
    {
        // for  a given rot uptade the min distance
        cPt4dr aP1 =  KthQuat(aKRot);  //  quaternion
        cDenseMatrix<tREAL8>   aM1 = Quat2MatrRot(aP1);  // matrix
        for (size_t aKP=0 ; aKP<aNbTest ; aKP++)
        {
            //  distance between rot or quat, depending on "m4R"
            tREAL8 aD = m4R ? Square(aM1.L2Dist(aVRot[aKP]))  : SqN2(aP1-aVPts[aKP]);
            UpdateMin(aVDMin[aKP],aD);
        }
    }
    return aVDMin;
}

cStdStatRes  cSampleQuat::TestStatMinDist(size_t aNbTest) const
{
    std::vector<tREAL8 >  aVDMin = TestVecMinDist(aNbTest);
    cStdStatRes aRes;
    for (const auto & aDMin : aVDMin)
        aRes.Add(std::sqrt(aDMin));

    return aRes;
}

tREAL8 cSampleQuat::TestMinDistPairQuat() const
{
    tREAL8 aD12Min=1e8;
    std::vector<cPt4dr > aVPRot = VecAllQuat();
    for (size_t aKRot1=0; aKRot1 < NbRot(); aKRot1++)
    {
        for (size_t aKRot2=aKRot1+1; aKRot2 < NbRot(); aKRot2++)
        {
             tREAL8 aD = SqN2(aVPRot[aKRot1]-aVPRot[aKRot2]);
             UpdateMin(aD12Min,aD);
        }
    }
    return std::sqrt(aD12Min);
}

/*  Test of sampling of quaternion ; as it is difficult to make stritc test, by defauklt it is inactivated,
 *  if need activate in the if and check the values
 *
 */
void BenchSampleQuat()
{
    // test ratio of distance between quaternion and their rotation
    // show experimentally that
    //    - this ratio is constant for small difference, value ~ 1.06 (why not 1 or another majic number ??)
    //    - this ratio is not bounded because the mapping is not bijective, several quat 
    //      can correspond to the same rotation

/*  In fact this ratio is perfectly normal if we take into account quat => mat & micmac convention for matrix dist
 *  (wich is averag quad and no a sum)
 *
 * Consider a small quat  Q(d) = (1,0,0,d) / sqrt(1+d2)  we have ||Q(d)-1|| ~ d
 *
 *
 *         (1-d^2  -2d     0    )
 * M(d) =  (2d     1-d^2   0    )  / (1+d^2) 
 *         (0      0       1+d^2)
 *
 * and  || M(d) -Id|| ~  d sqrt(8/9)     and "sqrt(9/8) = 1.06066 "
 *
 *
 */
   if (0)
   {
        tREAL8 aThR = std::sqrt(9.0/8.0);  // theoreticall ratio
        StdOut() <<  "1.06 ?  = " << aThR << std::endl;
        for (tREAL8 aEps : {1e-3,1e-2,1e-1})  // for different value of what means close
        {
             cStdStatRes aStatC1;  // stat for close points
             cStdStatRes aStat2;   // stat for random pairs
             for (int aK=0 ;aK<100000 ; aK++)
             {
                 cPt4dr aP1 = cPtxd<tREAL8,4>::PRandUnit() ;  // random point
                 cPt4dr aP2 = cPtxd<tREAL8,4>::PRandUnit() ;  // random point
                 cPt4dr aPC1 = VUnit(aP1+cPt4dr::PRandUnit() *aEps);;  // random point "close to P1"
                 tREAL8 aDP12 = Norm2(aP1-aP2);  // distance between random pair of quaternion
                 tREAL8 aDP1C = Norm2(aP1-aPC1);  // distance  between closed points

                 cDenseMatrix<tREAL8> aM1 = Quat2MatrRot(aP1);    // Mat rotation corresponding to quat P1
                 cDenseMatrix<tREAL8> aM2 = Quat2MatrRot(aP2);    // Mat rotation corresponding to quat P2
                 cDenseMatrix<tREAL8> aMC1 = Quat2MatrRot(aPC1);  // Mat rotation corresponding to quat PC1


                 tREAL8 aDM12 = aM1.L2Dist(aM2); // distance between matrix for random pair
                 tREAL8 aDM1 = aM1.L2Dist(aMC1); // distance between matrix for closed points

                 aStatC1.Add(aDP1C/aDM1);   // add ratio Quat/mat of closed pair 
                 aStat2.Add (aDP12/aDM12);  // add ratio Quat/mat of random pair 

             }

             StdOut() << " ============ "  << aEps  << " ============ " << std::endl;
             StdOut() << "R12 = " << aStat2.Min()  << " " << aStat2.Max()  << std::endl;
             StdOut() << "R1 -sqrt(9/8)  = " << std::abs(aStatC1.Min()-aThR)  << " " << std::abs(aStatC1.Max()-aThR)  << std::endl;
        }
   }

   /*  check that for a given quaternion there exist one, and only one, other quaternion that have the same
    *  rotation (its opposite, by the way)
    */
   if (0)
   {
       int aNbStep = 6;
       cSampleQuat aSQ (aNbStep,false);
       tREAL8 aThreshold=1e-3;

       size_t aKQ = 0;  // index of a quaternion
       cPt4dr aPQ = aSQ.KthQuat(aKQ);  // the quaternion of the index
       cDenseMatrix<tREAL8> aMQ = Quat2MatrRot(aPQ);  // the matrix  of the quaternion

       cWhichMin<size_t,tREAL8> aWMin;  // save index closest for rotation
       std::vector<size_t> aVecClose;  // store all the index close

       for (size_t aKQ2= 0 ; aKQ2 <aSQ.NbRot() ; aKQ2++)  // parse all quat
       {
           if (aKQ2 != aKQ) // avoid the quat itself that would be obviously close
           {
               cDenseMatrix<tREAL8> aMQ2 = Quat2MatrRot(aSQ.KthQuat(aKQ2)); // compute matrix
               tREAL8 aD = aMQ2.L2Dist(aMQ);  // compute distance of matrix
               aWMin.Add(aKQ2,aD); // update if dist inf
               if (aD<aThreshold)  
               {
                  aVecClose.push_back(aKQ2);  // memorize if close
               }
           }
       }

       StdOut()  << "PQ=" << aPQ << " NBC" << aVecClose << std::endl;
       StdOut()  << "D=" << aWMin.ValExtre()  << std::endl;
       StdOut()  << "Q2=" << aSQ.KthQuat(aWMin.IndexExtre())  << std::endl;
   }

    if (0)
    {
        int aNbRot =    50000;
        int aNbTest =   10000;

        cSampleQuat aSQ = cSampleQuat::FromNbRot(aNbRot,false);
        StdOut() << "DMAX=" << aSQ.TestStatMinDist(aNbTest).Max() << std::endl;

        cSampleQuat aSQR = cSampleQuat::FromNbRot(aNbRot,true);
        StdOut() << "DMAXR=" << aSQR.TestStatMinDist(aNbTest).Max() << std::endl;

        StdOut() << "D12Min=" << aSQ.TestMinDistPairQuat() << std::endl;
    }
}





/* ========================== */
/*          ::                */
/* ========================== */


/*
*/
#define MACRO_INSTATIATE_PTXD(TYPE)\
template  cRotation3D<tREAL8>  ToReal8(const cRotation3D<TYPE>  & aRot);\
template  cIsometry3D<tREAL8>  ToReal8(const cIsometry3D<TYPE>  & anIsom);\
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
