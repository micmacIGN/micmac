#include "MMVII_Matrix.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

/*  *********************************************************** */
/*                                                              */
/*                  tSeg3dr                                     */
/*                                                              */
/*  *********************************************************** */

/*
 *   "ONR"= Orthognal Normalised Repair 
 *
 *    Algo 4 Bundle inter, for each seg Sk,  let 
 *
 *       *  uk be a normal vector, and Pk a point on SK
 *
 *    I we complete the base with vk,wk such that (uk,vk,wk) is a ONR, we can write the least square equations "Q in Sk " as :
 *
 *       (Pk-Q) . vk=0   ; (Pk-Q).wk = 0
 *    
 *    So the square resiual R2 is
 *
 *    R2  =   Sum  ( (Pk-Q) . vk ^2 + (Pk-Q).wk )
 *        =   Sum  ( (Pk-Q) (tvk vk + twk wk) (Pk-Q) )
 *
 *    But as its a ONR  we have  tukuk +  tvk vk + twk wk = Id  , let call :
 *
 *      Ok = Id- tukuk = tvk vk + twk wk 
 *
 *   R2 = Sum(  tQ Ok Q  -2 Ok Pk + Cte)
 *
 *
 *
 */

	/*
cPt3dr  BundleInters(const std::vector<tSeg3dr> & aVSeg,const std::vector<tREAL8> * aVWeight = nullptr)
{
     cDenseMatrix<tREAL8>  aDM(3,eModeInitImage::eMIA_Null);
     cPt3dr aRHS(0,0,0);

     int aNbWNN = 0;

     for (size_t aKSeg=0 ; aKSeg<aVSeg.size() ; aKSeg++)
     {
         tREAL8 aW = aVWeight ?  aVWeight->at(aKSeg)  : 1.0 ;

	 cPt3dr aP1  =  aVSeg[aK].P1();
	 cPt3dr aP2  =  aVSeg[aK].P2();

	 cPt3dr aUk = VUnit(aP2-aP1);
	 tREAL8 aXu = aUk.x();
	 tREAL8 aYu = aUk.y();
	 tREAL8 aZu = aUk.z();

	 tREAL8 a00 = 1 - Square(aUk.x());

         if (aW>0) aNbWNN++;
     }
     MMVII_INTERNAL_ASSERT_bench(aNbWNN>=2,"Not enough segs in BundleInters");

     aDM.SelfSymetrizeBottom();
     return SolveCol(aDM,aRHS);
}
*/


/*  *********************************************************** */
/*                                                              */
/*                  cPlan3D                                     */
/*                                                              */
/*  *********************************************************** */


cPlane3D::cPlane3D(const cPt3dr & aP0,const cPt3dr& aAxeI , const cPt3dr& aAxeJ) :
     mP0(aP0)
{
    cRotation3D<tREAL8> aRot = cRotation3D<tREAL8>::CompleteRON(aAxeI,aAxeJ);

    mAxeI = aRot.AxeI();
    mAxeJ = aRot.AxeJ();
    mAxeK = aRot.AxeK();
}


cPlane3D cPlane3D::FromPtAndNormal(const cPt3dr & aP0,const cPt3dr& aAxeK)
{
   cRotation3D<tREAL8> aRep = cRotation3D<tREAL8>::CompleteRON(aAxeK);

   return cPlane3D(aP0,aRep.AxeJ(),aRep.AxeK());
}

const cPt3dr& cPlane3D::AxeI() const {return mAxeI;}
const cPt3dr& cPlane3D::AxeJ() const {return mAxeJ;}
const cPt3dr& cPlane3D::AxeK() const {return mAxeK;}

cPt3dr  cPlane3D::ToLocCoord(const cPt3dr & aPGlob) const
{
     cPt3dr aVect = aPGlob-mP0;
     return cPt3dr (Scal(mAxeI,aVect), Scal(mAxeJ,aVect), Scal(mAxeK,aVect));
}

cPt3dr  cPlane3D::FromCoordLoc(const cPt3dr & aP) const
{
    return mP0 + mAxeI*aP.x() + mAxeJ*aP.y() + mAxeK*aP.z();
}

cPt3dr  cPlane3D::Inter(const cPt3dr&aP0,const cPt3dr&aP1) const
{
     cPt3dr aVect = aP1-aP0;
     tREAL8 aS1  = Scal(mAxeK,aP1-mP0);
     tREAL8 aS01 = Scal(mAxeK,aVect);
    
     //  Scal(mAxeK,aP1+t*aVect -mP0) = 0
     //  t = - Scal(aP1-mP0,aK)  / Scal (aVect,aK)

     return  aP1 -  aVect*(aS1/aS01);
}
cPt3dr  cPlane3D::Inter(const tSeg3dr&aSeg) const {return Inter(aSeg.P1(),aSeg.P2());}


std::vector<cPt3dr>  cPlane3D::RandParam()
{
    cPt3dr aP0 = cPt3dr::PRandC() * 100.0;

    cPt3dr  aI =  cPt3dr::PRandUnit() ;
    cPt3dr  aJ =  cPt3dr::PRandUnitDiff(aI) ;

    return std::vector<cPt3dr>{aP0,aI*RandInInterval(0.1,2.0),aJ*RandInInterval(0.1,2.0)};
}

void BenchPlane3D()
{
    for  (int aK=0 ;aK<100 ;aK++)
    {
         std::vector<cPt3dr>  aVP = cPlane3D::RandParam();
         cPlane3D aPlane(aVP[0],aVP[1],aVP[2]);
	 MMVII_INTERNAL_ASSERT_bench(Norm2(aPlane.ToLocCoord(aVP[0])) < 1e-9,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(aPlane.ToLocCoord(aVP[0]+aVP[1]).z())<1e-5,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(aPlane.ToLocCoord(aVP[0]+aVP[2]).z())<1e-5,"BenchPlane3D");

         cPt3dr aP0 = cPt3dr::PRandC() * 100.0;
         cPt3dr aP1 = aP0 +  aPlane.AxeI() * RandUnif_C() + aPlane.AxeJ() * RandUnif_C() +  aPlane.AxeK()  * RandUnif_C_NotNull(0.1);

	 cPt3dr aPI = aPlane.Inter(aP0,aP1);
	 cPt3dr aPI2 = aPlane.Inter(tSeg3dr(aP0,aP1));
	 MMVII_INTERNAL_ASSERT_bench(Norm2(aPI-aPI2)<1e-9,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(std::abs(aPlane.ToLocCoord(aPI).z())<1e-5,"BenchPlane3D");

	 cSegmentCompiled<tREAL8,3> aSeg(aP0,aP1);
	 MMVII_INTERNAL_ASSERT_bench(aSeg.Dist(aPI)<1e-5,"BenchPlane3D");

	 MMVII_INTERNAL_ASSERT_bench(Norm2(aP0 -aPlane.ToLocCoord(aPlane.FromCoordLoc(aP0)))<1e-5,"BenchPlane3D");
	 MMVII_INTERNAL_ASSERT_bench(Norm2(aP0 -aPlane.FromCoordLoc(aPlane.ToLocCoord(aP0)))<1e-5,"BenchPlane3D");
    }
}

/*  *********************************************************** */
/*                                                              */
/*  *********************************************************** */

template<class T> cPtxd<T,3>  PFromNumAxe(int aNum)
{
   static const cDenseMatrix<T> anId3x3(3,3,eModeInitImage::eMIA_MatrixId);
   return cPtxd<T,3>::Col(anId3x3,aNum);
}

template<class T> cDenseMatrix<T> MatFromCols(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2)
{
   cDenseMatrix<T> aRes(3,3);

   SetCol(aRes,0,aP0);
   SetCol(aRes,1,aP1);
   SetCol(aRes,2,aP2);

   return aRes;
}

template<class T> cDenseMatrix<T> MatFromLines(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2)
{
   cDenseMatrix<T> aRes(3,3);

   SetLine(0,aRes,aP0);
   SetLine(1,aRes,aP1);
   SetLine(2,aRes,aP2);

   return aRes;
}

/*
    (X1)   (X2)      Y1*Z2 - Z1*Y2     ( 0   -Z1    Y1)   (X2) 
    (Y1) ^ (Y2) =    Z1*X2 - X1*Z2  =  ( Z1    0   -X1) * (Y2)
    (Z1)   (Z2)      X1*Y2 - Y1*X2     (-Y1    X1    0)   (Z2)
 
*/

template<class T> cDenseMatrix<T> MatProdVect(const cPtxd<T,3>& W)
{
	return MatFromLines<T>
               (
	          cPtxd<T,3>(  0    , -W.z() ,  W.y() ),
	          cPtxd<T,3>( W.z() ,   0    , -W.x() ),
	          cPtxd<T,3>(-W.y() ,  W.x() ,   0    )
	       );
}


/*
template <class T>  cPtxd<T,3> operator ^ (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2)
{
   return cPtxd<T,3>
          (
               aP1.y() * aP2.z() -aP1.z()*aP2.y(),
               aP1.z() * aP2.x() -aP1.x()*aP2.z(),
               aP1.x() * aP2.y() -aP1.y()*aP2.x()
          );
}
*/

template<class T> cPtxd<T,3>  VOrthog(const cPtxd<T,3> & aP)
{
   // we make a vect product with any vector, just avoid one too colinear  to P
   // test I and J, as P cannot be colinear to both, its sufficient 
   // (i.e : we are sur to maintain the biggest of x, y and z)
   if (std::abs(aP.x()) > std::abs(aP.y()))
      return cPtxd<T,3>( aP.z(), 0, -aP.x());

  return cPtxd<T,3>(0,aP.z(),-aP.y());
}

template <class T>  T  TetraReg (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const T& FactEps)
{
     cPtxd<T,3> aCDG = (aP1+aP2+aP3) /static_cast<T>(3.0);

     T aSqDist = (SqN2(aCDG) + SqN2(aCDG-aP1) + SqN2(aCDG-aP2) + SqN2(aCDG-aP3)) / static_cast<T>(4.0);
     T aCoeffNorm = std::pow(aSqDist,3.0/2.0);  // 1/2 for D2  3->volume

     return Determinant(aP1,aP2,aP3) / (aCoeffNorm +  std::numeric_limits<T>::epsilon()*FactEps);
}

template <class T>  T  TetraReg (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const cPtxd<T,3> & aP4,const T& FactEps)
{
    return TetraReg(aP2-aP1,aP3-aP1,aP4-aP1,FactEps);
}

template <class T>  T  Determinant (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3)
{
	return Scal(aP1,aP2^aP3);
}

template<class Type>  cTriangle<Type,3> RandomTriang(Type aAmpl)
{
      return cTriangle<Type,3>(cPtxd<Type,3>::PRandC()*aAmpl,cPtxd<Type,3>::PRandC()*aAmpl,cPtxd<Type,3>::PRandC()*aAmpl);
}

template<class Type>  cTriangle<Type,3> RandomTriangRegul(Type aRegulMin,Type aAmpl)
{
    for (;;)
    {
        cTriangle<Type,3> aT = RandomTriang(aAmpl);
	if (aT.Regularity()> aRegulMin)
           return aT;
    }
    return RandomTriang(static_cast<Type>(0.0)); // Not sur its mandatory to have a return here
}

template<class Type>  cTriangle<Type,3> RandomTetraTriangRegul(Type aRegulMin,Type aAmpl)
{
    for (;;)
    {
        cTriangle<Type,3> aT = RandomTriang(aAmpl);
	if (TetraReg(aT.Pt(0),aT.Pt(1),aT.Pt(2)) > aRegulMin)
           return aT;
    }
    return RandomTriang(static_cast<Type>(0.0)); // Not sur its mandatory to have a return here
}




/* ========================== */
/*          ::                */
/* ========================== */

//template cPtxd<int,3>  operator ^ (const cPtxd<int,3> & aP1,const cPtxd<int,3> & aP2);
//template cPtxd<TYPE,3>  operator ^ (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2);

#define MACRO_INSTATIATE_PTXD(TYPE)\
template  cTriangle<TYPE,3> RandomTriang(TYPE aRegulMin);\
template  cTriangle<TYPE,3> RandomTriangRegul(TYPE aRegulMin,TYPE aAmpl);\
template  cTriangle<TYPE,3> RandomTetraTriangRegul(TYPE aRegulMin,TYPE aAmpl);\
template TYPE  Determinant (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3);\
template TYPE  TetraReg (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3,const TYPE&);\
template TYPE  TetraReg (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3,const cPtxd<TYPE,3> & aP4,const TYPE&);\
template cDenseMatrix<TYPE> MatProdVect(const cPtxd<TYPE,3>& W);\
template cDenseMatrix<TYPE> MatFromCols(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cDenseMatrix<TYPE> MatFromLines(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cPtxd<TYPE,3>  PFromNumAxe(int aNum);\
template cPtxd<TYPE,3>  VOrthog(const cPtxd<TYPE,3> & aP);


MACRO_INSTATIATE_PTXD(tREAL4)
MACRO_INSTATIATE_PTXD(tREAL8)
MACRO_INSTATIATE_PTXD(tREAL16)



};
