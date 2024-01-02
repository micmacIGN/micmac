#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Tpl_Images.h"

namespace MMVII
{

/**  This class is just a copy of Eigen indexe, this allow to use eigen sparse
      matrix w/o having to include eigen headers
*/




/*
   General note on this file :

      use of static fonction in first version to allow use of function
      on various vector (REAL4,8,16) as  the method could not be template & virtural
   
*/

/* ========================== */
/*          cSParseVect        */
/* ========================== */

template <class Type> cSparseVect<Type>::cSparseVect(int aSzReserve) :
   mIV (new tCont)
{
  if (aSzReserve>0) 
     IV().reserve(aSzReserve);
}

template <class Type> cSparseVect<Type> cSparseVect<Type>::Dup() const
{
    tSV aRes(size());

   *(aRes.mIV) = *mIV;
 
   return aRes;
}

template <class Type> cSparseVect<Type>::cSparseVect(const cDenseVect<Type> & aDV) :
    cSparseVect<Type>  (aDV.Sz())
{
    for (int aK=0 ; aK<aDV.Sz() ; aK++)
       if (aDV(aK)!=0)
           AddIV(aK,aDV(aK));
}

template <class Type>  bool cSparseVect<Type>::IsInside(int aNb) const
{
    for (const auto & aP : *(mIV.get()))
    {
        if ((aP.mInd<0) || (aP.mInd>= aNb))
           return false;
    }
    return true;
}

template <class Type> void cSparseVect<Type>::Reset()
{
    mIV->clear();
}

template <class Type>  void cSparseVect<Type>::AddIV(const int & anInd,const Type & aV)
{
   AddIV(tCplIV(anInd,aV));
}
template <class Type>  void cSparseVect<Type>::AddIV(const tCplIV & aCpl) 
{ 
   IV().push_back(aCpl); 
}


template <class Type> void cSparseVect<Type>::CumulIV(const tCplIV & aCpl)
{
   tCplIV * aPairExist = Find(aCpl.mInd);
   if (aPairExist != nullptr)
   {
       aPairExist->mVal += aCpl.mVal;
   }
   else
   {
        AddIV(aCpl);
   }
}


template <class Type> cSparseVect<Type>  cSparseVect<Type>::RanGenerate(int aNbVar,double aProba,tREAL8 aMinVal,int aMinSize)
{
    cSparseVect<Type>  aRes;

    while (aRes.size()<aMinSize)
    {
        for (int aK=0 ; aK<aNbVar ; aK++)
        {
            if(RandUnif_0_1() < aProba)
	    {
                aRes.AddIV(aK,RandUnif_C_NotNull(aMinVal));
	    }
        }
    }

    return aRes;
}

template <class Type> const typename cSparseVect<Type>::tCplIV *  cSparseVect<Type>::Find(int anInd) const
{
    for (const auto & aPair : *mIV)
        if (aPair.mInd==anInd)
           return & aPair;
    return nullptr;
}
template <class Type> typename cSparseVect<Type>::tCplIV *  cSparseVect<Type>::Find(int anInd) 
{
    for (auto & aPair : *mIV)
        if (aPair.mInd==anInd)
           return & aPair;
    return nullptr;
}

template <class Type>  int cSparseVect<Type>::MaxIndex(int aDef) const
{
    for (const auto & aPair : *mIV)
        UpdateMax(aDef,aPair.mInd);

    MMVII_INTERNAL_ASSERT_tiny(aDef>=-1,"No def value for empty vect in cSparseVect<Type>::MaxIndex");
    return aDef;
}

template <class Type>  void cSparseVect<Type>::EraseIndex(int anInd)
{
    erase_if(*mIV,[anInd](const auto & aPair){return aPair.mInd==anInd;});
}
/*
*/



/* ========================== */
/*          cDenseVect        */
/* ========================== */

template <class Type> cDenseVect<Type>::cDenseVect(tIM anIm) :
   mIm  (anIm) 
{
}

template <class Type> cDenseVect<Type>  cDenseVect<Type>::RanGenerate(int aSz)
{
    return cDenseVect<Type>(aSz,eModeInitImage::eMIA_RandCenter);
}


template <class Type> cDenseVect<Type>::cDenseVect(const std::vector<Type> & aVect) :
   cDenseVect<Type> (tIM(aVect))
{
}

template <class Type> cDenseVect<Type>::cDenseVect(int aSz,eModeInitImage aModeInit) :
   cDenseVect<Type> (tIM  (aSz,nullptr,aModeInit))
{
}

template <class Type> cDenseVect<Type>::cDenseVect(const tSpV & aSpV,int aSz) :
       cDenseVect<Type>(std::max(aSz,1+aSpV.MaxIndex()) ,eModeInitImage::eMIA_Null)
{
     for (const auto & aPair : aSpV)
          mIm.DIm().SetV(aPair.mInd,aPair.mVal);
}


template <class Type> cDenseVect<Type> cDenseVect<Type>::Dup() const
{
    return cDenseVect<Type>(mIm.Dup());
}

template <class Type>  void cDenseVect<Type>::ResizeAndCropIn(const int & aX0,const int & aX1,const cDenseVect<Type> & aV2)
{
    tDIM & aDIm = mIm.DIm();

    aDIm.Resize(aX1-aX0);
    aDIm.CropIn(aX0,aV2.DIm());
}

template <class Type>  void cDenseVect<Type>::Resize(const int & aSz)
{
     mIm.DIm().Resize(aSz);
}

template <class Type> cDenseVect<Type>   cDenseVect<Type>::Cste(int aSz,const Type & aVal)
{
    cDenseVect<Type> aRes(aSz);

    Type * aRD = aRes.RawData();
    for (int aK=0 ; aK<aSz ; aK++)
        aRD[aK] = aVal;

    return aRes;
}

template <class Type> cDenseVect<Type>   cDenseVect<Type>::SubVect(int aK0,int aK1) const
{
      MMVII_INTERNAL_ASSERT_tiny
      (
         (aK0>=0) && (aK0<aK1) && (aK1<=Sz()),
	 "Bad size in SubVect"
      );
      cDenseVect<Type> aRes(aK1-aK0);

      for (int aK=aK0 ; aK<aK1 ; aK++)
          aRes(aK-aK0) = (*this)(aK);

      return aRes;
}

template <class Type> std::vector<Type>   cDenseVect<Type>::ToStdVect() const
{
    return std::vector<Type>(RawData(),RawData()+Sz());
}


/*
template <class Type> cDenseVect<Type>::cDenseVect(int aSz,eModeInitImage aModeInit) :
   mIm  (aSz,nullptr,aModeInit)
{
}
*/

template <class Type> double cDenseVect<Type>::L1Dist(const cDenseVect<Type> & aV,bool isAvg) const
{
   return mIm.DIm().L1Dist(aV.mIm.DIm(),isAvg);
}
template <class Type> double cDenseVect<Type>::L2Dist(const cDenseVect<Type> & aV,bool isAvg) const
{
   return mIm.DIm().L2Dist(aV.mIm.DIm(),isAvg);
}

template <class Type> double cDenseVect<Type>::L1Norm(bool isAvg) const
{
   return mIm.DIm().L1Norm(isAvg);
}
template <class Type> double cDenseVect<Type>::L2Norm(bool isAvg) const
{
   return mIm.DIm().L2Norm(isAvg);
}
template <class Type> double cDenseVect<Type>::SqL2Norm(bool isAvg) const
{
   return mIm.DIm().SqL2Norm(isAvg);
}
template <class Type> double cDenseVect<Type>::LInfNorm() const
{
   return mIm.DIm().LInfNorm();
}

template <class Type> cDenseVect<Type> cDenseVect<Type>::VecUnit() const
{
   return  Type(SafeDiv(1.0,L2Norm()))  * (*this);
}

// double L1Norm() const;   ///< Norm som abs double L2Norm() const;   ///< Norm square double LInfNorm() const; ///< Nomr max





template <class Type> double cDenseVect<Type>::DotProduct(const cDenseVect<Type> & aV) const
{
   return MMVII::DotProduct(DIm(),aV.DIm());
}
// double L2Dist(const cDenseVect<Type> & aV) const;

template <class Type> Type*       cDenseVect<Type>::RawData()       {return DIm().RawDataLin();}
template <class Type> const Type* cDenseVect<Type>::RawData() const {return DIm().RawDataLin();}

// const Type * RawData() const;

template <class Type> void  cDenseVect<Type>::WeightedAddIn(Type aW,const tSpV & aVect)
{
    Type * aD =  RawData();

    for (const auto & aP : aVect)
       aD[aP.mInd] += aW * aP.mVal;
}


template <class Type> std::ostream & operator << (std::ostream & OS,const cDenseVect<Type> &aV)
{
   OS << "[";
   for (int aK=0 ; aK<aV.DIm().Sz() ; aK++)
   {
         if (aK!=0) OS << " ";
         OS << aV(aK);
   }
   OS << "]";
   return OS;
}


template <class Type> Type  cDenseVect<Type>::ProdElem() const
{
   Type aRes = 1.0;
   for (int aK=0 ; aK<Sz() ; aK++)
        aRes *= (*this)(aK);

   return aRes;
}

template <class Type> Type  cDenseVect<Type>::SumElem() const
{
   Type aRes = 0.0;
   for (int aK=0 ; aK<Sz() ; aK++)
        aRes += (*this)(aK);

   return aRes;
}

template <class Type> Type  cDenseVect<Type>::AvgElem() const
{
    return SumElem() / Type(Sz());
}

template <class Type> void  cDenseVect<Type>::SetAvg(const Type & aTargAvg)
{
   Type  aMul = SafeDiv (aTargAvg,AvgElem());
   for (int aK=0 ; aK<Sz() ; aK++)
        (*this)(aK) *= aMul;
}

template <class Type>  cDenseVect<Type> cDenseVect<Type>::GramSchmidtCompletion(const std::vector<tDV> & aVV) const
{
     cDenseVect<Type> aRes = *this;
 
     for (const auto & aV : aVV)
     {
         aRes = aRes -  Type(aV.DotProduct(*this)/aV.SqL2Norm()) * aV;
     }

     return aRes;
}

template <class Type>  std::vector<cDenseVect<Type>>  cDenseVect<Type>::GramSchmidtOrthogonalization(const std::vector<tDV> & aVV) 
{
     std::vector<cDenseVect<Type>> aRes;

     for (const auto  & aV : aVV)
         aRes.push_back(aV.GramSchmidtCompletion(aRes));

     return aRes;
}

template <class Type> cDenseVect<Type>   cDenseVect<Type>::VecComplem(const std::vector<tDV> & aVV,Type aDMin) 
{
     size_t aDim = aVV.at(0).Sz();
     cDenseVect<Type>  aTest(aDim,eModeInitImage::eMIA_Null);

     cWhichMax<int,Type>  aWMax(-1,-1.0);

     for (size_t aK=0 ; aK<=aDim ; aK++)
     {
         // forced end of loop, we select the "less bad vector"
         if (aK==aDim)
         {
             aDMin = -1;
             aK = aWMax.IndexExtre();
         }
         aTest(aK) = 1;
         
         cDenseVect<Type> aRes = aTest-aTest.ProjOnSubspace(aVV);
         Type aNorm = aRes.L2Norm();
         if (aNorm>aDMin)
            return Type(1.0/aNorm) * aRes;
         aWMax.Add(aK,aNorm);
         aTest(aK) = 0;
     }

     MMVII_INTERNAL_ASSERT_tiny(false,"VecComplem : should not be here !!");
     return aTest;
}

// Can go much faster by selecting all the  result inside VecCompl and selecting the K Best
template <class Type> std::vector<cDenseVect<Type>>  cDenseVect<Type>::BaseComplem(const std::vector<tDV> & aVV,bool WithInit,Type aDMin) 
{
     int aDim = aVV.at(0).Sz();
     std::vector<tDV> aRes = aVV;

     for (int aK= aVV.size() ; aK<aDim ; aK++)
         aRes.push_back(VecComplem(aRes,aDMin));

     if (WithInit)
        return aRes;

    return std::vector<cDenseVect<Type>>(aRes.begin()+aVV.size(),aRes.end());
}

template <class Type> Type cDenseVect<Type>::ASymApproxDistBetweenSubspace(const std::vector<tDV>  & aVV1,const std::vector<tDV>  & aVV2)
{
   Type aRes=0.0;

   for (const auto & aV1 : aVV1)
       UpdateMax(aRes,aV1.DistToSubspace(aVV2));

   return aRes;
}

template <class Type> Type cDenseVect<Type>::ApproxDistBetweenSubspace(const std::vector<tDV>  & aVV1,const std::vector<tDV>  & aVV2)
{
	return std::max(ASymApproxDistBetweenSubspace(aVV1,aVV2),ASymApproxDistBetweenSubspace(aVV2,aVV1));
}

/*
template <class Type> void AddData(const  cAuxAr2007 & anAux, cDenseVect<Type> & aDV)
{
    int aNbEl = aDV.Sz();
    AddData(cAuxAr2007("NbEl",anAux),aNbEl);
    if (anAux.Input())
       aDV = cDenseVect<Type>(aNbEl);

    TplAddRawData(anAux,aDV.RawData(),aNbEl);
}
*/

/* ========================== */
/*          cMatrix       */
/* ========================== */
template <class Type> cMatrix<Type>::cMatrix(int aX,int aY) :
   cRect2(cPt2di(0,0),cPt2di(aX,aY))
{
}


template <class Type> double cMatrix<Type>::TriangSupicity() const   ///< How close to triangular sup
{
     double aNb=0;
     double aSom =0.0;
     for (const auto & aP : *this)
     {
         if (aP.x() < aP.y())
         {
            aNb++;
            aSom += Square(V_GetElem(aP.x(),aP.y()));
         }
     }
     aSom /= std::max(1.0,aNb);
     return std::sqrt(aSom);
}
template <class Type> void cMatrix<Type>::SelfSymetrizeBottom()
{
   cMatrix<Type>::CheckSquare(*this);
   int aNb = Sz().x();
   for (int aX=0 ; aX<aNb ; aX++)
   {
       for (int aY=aX+1 ; aY<aNb ; aY++)
       {
            V_SetElem(aX,aY,V_GetElem(aY,aX));
       }
   }
}


         // ============  Mul Col ========================

template <class Type> static Type TplMulColElem(int aY,const cMatrix<Type> & aMat,const cDenseVect<Type> & aVIn)
{
    aMat.TplCheckSizeX(aVIn);

    Type aRes = 0.0;
    for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aRes += aVIn(aX) * aMat.V_GetElem(aX,aY);

    return aRes;
}


template <class Type> static void TplMulCol(cDenseVect<Type> & aVOut,const cMatrix<Type> & aMat,const cDenseVect<Type> & aVIn) 
{
   aMat.TplCheckSizeYandX(aVOut,aVIn);

   if (&aVOut==&aVIn) // Will see later if we handle this case
   {
       MMVII_INTERNAL_ASSERT_strong(false,"Aliasing in TplMulCol")
   }

   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
   {
       aVOut(aY) = TplMulColElem(aY,aMat,aVIn);
   }
}

template <class Type> static cDenseVect<Type> TplMulCol(const cMatrix<Type> & aMat,const cDenseVect<Type> & aVIn) 
{
    cDenseVect<Type> aRes(aMat.Sz().y());
    TplMulCol(aRes,aMat,aVIn);

    return aRes;
}

template <class Type> static void TplReadColInPlace(const cMatrix<Type> & aMat,int aX,cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeY(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aV(aY) = aMat.V_GetElem(aX,aY);
}

template <class Type> static void TplWriteCol(cMatrix<Type> & aMat,int aX,const cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeY(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aMat.V_SetElem(aX,aY,aV(aY)) ;
}


template <class Type> static void TplAdd_tAB(cMatrix<Type> & aMat,const cDenseVect<Type> & aCol,const cDenseVect<Type> & aLine)
{
    aMat.TplCheckSizeY(aCol);
    aMat.TplCheckSizeX(aLine);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        {
           aMat.V_SetElem(aX,aY,aMat.V_GetElem(aX,aY) + aCol(aY) * aLine(aX));
        }
    }
}

template <class Type> static void Weighted_TplAdd_tAA(Type aW,cMatrix<Type> & aMat,const cDenseVect<Type> & aV,bool OnlySup)
{
    aMat.TplCheckSizeY(aV);
    aMat.TplCheckSizeX(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        Type aVy = aV(aY) * aW;
        for (int aX= OnlySup ? aY : 0 ; aX<aMat.Sz().x() ; aX++)
        {
           aMat.V_SetElem(aX,aY,aMat.V_GetElem(aX,aY) + aVy * aV(aX));
        }
    }
}

template <class Type> static void TplAdd_tAA(cMatrix<Type> & aMat,const cDenseVect<Type> & aV,bool OnlySup)
{
     Weighted_TplAdd_tAA(Type(1.0),aMat,aV,OnlySup);
/*
    aMat.TplCheckSizeY(aV);
    aMat.TplCheckSizeX(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        for (int aX= OnlySup ? aY : 0 ; aX<aMat.Sz().x() ; aX++)
        {
           aMat.V_SetElem(aX,aY,aMat.V_GetElem(aX,aY) + aV(aY) * aV(aX));
        }
    }
*/
}
template <class Type> static void TplSub_tAA(cMatrix<Type> & aMat,const cDenseVect<Type> & aV,bool OnlySup)
{
    Weighted_TplAdd_tAA(Type(-1.0),aMat,aV,OnlySup);
/*
    aMat.TplCheckSizeY(aV);
    aMat.TplCheckSizeX(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        for (int aX= OnlySup ? aY : 0 ; aX<aMat.Sz().x() ; aX++)
        {
           aMat.V_SetElem(aX,aY,aMat.V_GetElem(aX,aY) - aV(aY) * aV(aX));
        }
    }
*/
}


         // ============  Mul Line ========================

template <class Type> static Type TplMulLineElem(int aX,const cMatrix<Type> & aMat,const cDenseVect<Type> & aVIn)
{
    aMat.TplCheckSizeY(aVIn);
    Type aRes = 0.0;
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aRes += aVIn(aY) * aMat.V_GetElem(aX,aY);

    return aRes;
}


template <class Type> static void TplMulLine(cDenseVect<Type> & aVOut,const cMatrix<Type> & aMat,const cDenseVect<Type> & aVIn)  
{
   aMat.TplCheckSizeYandX(aVIn,aVOut);

   if (&aVOut==&aVIn) // Will see later if we handle this case
   {
       MMVII_INTERNAL_ASSERT_strong(false,"Aliasing in TplMulCol")
   }

   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
   {
       aVOut(aX) = TplMulLineElem(aX,aMat,aVIn);
   }
}

template <class Type> static cDenseVect<Type> TplMulLine(const cMatrix<Type> & aMat,const cDenseVect<Type> & aVIn) 
{
    cDenseVect<Type> aRes(aMat.Sz().x());
    TplMulLine(aRes,aMat,aVIn);

    return aRes;
}

template <class Type> static void TplReadLineInPlace(const cMatrix<Type> & aMat,int aY,cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeX(aV);
    for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aV(aX) = aMat.V_GetElem(aX,aY);
}



template <class Type> static void TplWriteLine(cMatrix<Type> & aMat,int aY,const cDenseVect<Type>& aV)
{
    aMat.TplCheckSizeX(aV);
    for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aMat.V_SetElem(aX,aY,aV(aX)) ;
}



     // Virtuals tREAL4

template <class Type> void  cMatrix<Type>::Add_tAB(const tDV & aCol,const tDV & aLine) { TplAdd_tAB(*this,aCol,aLine); }
template <class Type> void  cMatrix<Type>::Add_tAA(const tDV & aV,bool OnlySup) {TplAdd_tAA(*this,aV,OnlySup);}
template <class Type> void  cMatrix<Type>::Sub_tAA(const tDV & aV,bool OnlySup) {TplSub_tAA(*this,aV,OnlySup);}
template <class Type> void  cMatrix<Type>::Weighted_Add_tAA(Type aW,const tDV & aV,bool OnlySup) 
{
   Weighted_TplAdd_tAA(aW,*this,aV,OnlySup);
}



template <class Type> void cMatrix<Type>::MulColInPlace(tDV & aOut,const tDV & aIn) const
{
    TplMulCol(aOut,*this,aIn);
}
template <class Type> cDenseVect<Type> cMatrix<Type>::MulCol(const tDV & aIn) const
{
    return TplMulCol(*this,aIn);
}

template <class Type> Type cMatrix<Type>::MulColElem(int aY,const tDV & aIn) const
{
    return TplMulColElem(aY,*this,aIn);
}
template <class Type> void cMatrix<Type>::MulLineInPlace(tDV & aOut,const tDV & aIn) const
{
    TplMulLine(aOut,*this,aIn);
}
template <class Type> cDenseVect<Type> cMatrix<Type>::MulLine(const tDV & aIn) const
{
    return TplMulLine(*this,aIn);
}
template <class Type> Type cMatrix<Type>::MulLineElem(int aX,const tDV & aIn) const
{
    return TplMulLineElem(aX,*this,aIn);
}

template <class Type> void cMatrix<Type>::ReadColInPlace(int aX,tDV & aV)  const {TplReadColInPlace(*this,aX,aV);}
template <class Type> void cMatrix<Type>::WriteCol(int aX,const tDV  &aV)        {TplWriteCol(*this,aX,aV);}
template <class Type> void cMatrix<Type>::ReadLineInPlace(int aY,tDV & aV) const {TplReadLineInPlace(*this,aY,aV);}
template <class Type> void cMatrix<Type>::WriteLine(int aY,const tDV  &aV)       {TplWriteLine(*this,aY,aV);}

template <class Type> cDenseVect<Type>  cMatrix<Type>::ReadCol(int aX) const
{
     cDenseVect<Type> aRes(Sz().y());
     ReadColInPlace(aX,aRes);

     return aRes;
}

template <class Type> cDenseVect<Type>  cMatrix<Type>::ReadLine(int aY) const
{
     cDenseVect<Type> aRes(Sz().x());
     ReadLineInPlace(aY,aRes);

     return aRes;
}

template <class Type> std::vector<cDenseVect<Type>>  cMatrix<Type>::MakeLines() const
{
    std::vector<cDenseVect<Type>> aRes;
    for (int aY=0 ; aY<Sz().y() ; aY++)
        aRes.push_back(ReadLine(aY));

    return aRes;
}


     //    ===   MulMat ====


template <class Type> void cMatrix<Type>::MatMulInPlace(const tMat & aM1,const tMat & aM2)
{
   CheckSizeMulInPlace(aM1,aM2);

   for (int aY= 0 ; aY< this->Sz().y() ; aY++)
   {
       for (int aX= 0 ; aX< this->Sz().x() ; aX++)
       {
           tREAL16 aVal = 0.0;
           for (int aK=0 ; aK<aM1.Sz().x() ; aK++)
              aVal += aM1.V_GetElem(aK,aY) * aM2.V_GetElem(aX,aK);
           V_SetElem(aX,aY,aVal);
       }
   }
}
#if (0)
#endif

     /* ========== Methods with sparse vectors ============ */


template <class Type>  void  cMatrix<Type>::Weighted_Add_tAA(Type aWeight,const tSpV & aSparseV,bool OnlySup)
{
   CheckSquare(*this);
   TplCheckX(aSparseV);
   const typename cSparseVect<Type>::tCont & aIV =  aSparseV.IV();
   int aNb  = aIV.size();

   if (OnlySup)
   {
       for (int aKY=0 ; aKY<aNb ; aKY++)
       {
          Type aVy = aIV[aKY].mVal * aWeight;
          int  aY = aIV[aKY].mInd;
          for (int aKX=  0 ; aKX<aNb ; aKX++)
          {
              int aX = aIV[aKX].mInd;
              if (aX>=aY)
                 V_SetElem(aX,aY,V_GetElem(aX,aY) + aVy * aIV[aKX].mVal);
          }
       }
   }
   else
   {
       for (int aKY=0 ; aKY<aNb ; aKY++)
       {
          Type aVy = aIV[aKY].mVal * aWeight;
          int  aY = aIV[aKY].mInd;
          for (int aKX= 0 ; aKX<aNb ; aKX++)
          {
              int aX = aIV[aKX].mInd;
              V_SetElem(aX,aY,V_GetElem(aX,aY) + aVy * aIV[aKX].mVal);
          }
       }
   }
}



     /* ========== operator  ============ */

template <class Type> cDenseVect<Type> operator * (const cDenseVect<Type> & aLine,const cMatrix<Type>& aMat)
{
   return aMat.MulLine(aLine);
}

template <class Type> cDenseVect<Type> operator * (const cMatrix<Type> & aMat,const cDenseVect<Type> & aCol)
{
   return aMat.MulCol(aCol);
}

template <class Type> std::ostream & operator << (std::ostream & OS,const cMatrix<Type> &aMat)
{
   OS << "[\n";
   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
   {
      cDenseVect<Type> aV(aMat.Sz().x());
      aMat.ReadLineInPlace(aY,aV);
      OS << " " << aV << "\n";
         // if (aK!=0) OS << " ";
         // OS << aV(aK);
   }
   OS << "]\n";
   return OS;
}


template <class Type>  cDenseVect<Type>   cDenseVect<Type>::ProjOnSubspace(const std::vector<tDV>  & aVV) const
{
     cDenseVect<Type>  aRes(Sz(),eModeInitImage::eMIA_Null);

     int aNbVec = aVV.size();

     if (aNbVec)
     {
         cLeasSqtAA<Type>   aSys(aNbVec);

         for (int aKCoord=0 ; aKCoord<Sz()  ; aKCoord++)
         {
              cDenseVect<Type> anEq(aNbVec);
              for (int aKV=0 ; aKV<aNbVec ; aKV++)
                  anEq(aKV) = aVV.at(aKV)(aKCoord);

             aSys.PublicAddObservation(1.0,anEq,(*this)(aKCoord));
         }
         cDenseVect<Type>  aSol = aSys.Solve();

         for (int aKV = 0 ; aKV<aNbVec ; aKV++)
             aRes +=  aSol(aKV) * aVV.at(aKV);

      }
      return aRes;
}

template <>  cDenseVect<tINT4>   cDenseVect<tINT4>::ProjOnSubspace(const std::vector<tDV>  & aVV) const
{
    return cDenseVect<tINT4>(1);
}

template <class Type>  Type   cDenseVect<Type>::DistToSubspace(const std::vector<tDV>  & aVV) const
{
    cDenseVect<Type> aProj = ProjOnSubspace(aVV);

    return  this->L2Dist(aProj);
}


// template void AddData(const  cAuxAr2007 & anAux, cDenseVect<Type> &);

/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_DENSE_VECT(Type)\
template  class  cDenseVect<Type>;\
template  std::ostream & operator << (std::ostream & OS,const cDenseVect<Type> &aV);\

#define INSTANTIATE_BASE_MATRICES(Type)\
INSTANTIATE_DENSE_VECT(Type)\
template  class  cSparseVect<Type>;\
template  class  cMatrix<Type>;\
template  cDenseVect<Type> operator * (const cDenseVect<Type> & aLine,const cMatrix<Type> & aMat);\
template  cDenseVect<Type> operator * (const cMatrix<Type> & aMat ,const cDenseVect<Type> & aCol);\
template  std::ostream & operator << (std::ostream & OS,const cMatrix<Type> &aMat);\




INSTANTIATE_BASE_MATRICES(tREAL4)
INSTANTIATE_BASE_MATRICES(tREAL8)
INSTANTIATE_BASE_MATRICES(tREAL16)

// INSTANTIATE_DENSE_VECT(tU_INT1)
INSTANTIATE_DENSE_VECT(tINT4)



};


/* ========================== */
/*          cMatrix           */
/* ========================== */

