#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

// https://liris.cnrs.fr/page-membre/remi-ratajczak

namespace MMVII
{

/* ========================== */
/*       cLeasSqtAA           */
/* ========================== */



template<class Type>  cLeasSqtAA<Type>::cLeasSqtAA(int aNbVar):
   cLeasSq<Type>   (aNbVar),
   mtAA            (aNbVar,aNbVar,eModeInitImage::eMIA_Null),
   mtARhs          (aNbVar,eModeInitImage::eMIA_Null)
{
}

template<class Type> void  cLeasSqtAA<Type>::AddObservation
                           (
                               const Type& aWeight,
                               const cDenseVect<Type> & aCoeff,
                               const Type &  aRHS
                           ) 
{
    mtAA.Weighted_Add_tAA(aWeight,aCoeff,true);
    WeightedAddIn(mtARhs.DIm(),aWeight*aRHS,aCoeff.DIm());
}

template<class Type> void  cLeasSqtAA<Type>::AddObservation
                           (
                               const Type& aWeight,
                               const cSparseVect<Type> & aCoeff,
                               const Type &  aRHS
                           ) 
{
    mtAA.Weighted_Add_tAA(aWeight,aCoeff,true);
    mtARhs.WeightedAddIn(aWeight*aRHS,aCoeff);
}



template<class Type> void  cLeasSqtAA<Type>::Reset()
{
   mtAA.DIm().InitNull();
   mtARhs.DIm().InitNull();
}


template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::Solve()
{
   mtAA.SelfSymetrizeBottom();
   return mtAA.Solve(mtARhs,eTyEigenDec::eTED_LLDT);
}

template<class Type> const cDenseMatrix<Type> & cLeasSqtAA<Type>::tAA () const {return mtAA;}
template<class Type> const cDenseVect<Type>   & cLeasSqtAA<Type>::tARhs () const {return mtARhs;}

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::SparseSolve()
{
   const  cDataIm2D<Type> & aDIm = mtAA.DIm();
   std::vector<cEigenTriplet<Type> > aVCoeff;            // list of non-zeros coefficients
   for (const auto & aPix : aDIm)
   {
       const Type & aVal = aDIm.GetV(aPix);
       if ((aVal != 0.0)  && (aPix.x()>=aPix.y()))
       {
           cEigenTriplet<Type>  aTri(aPix.x(),aPix.y(),aVal);
           aVCoeff.push_back(aTri);
       }
   }

   return EigenSolveCholeskyarseFromV3(aVCoeff,mtARhs);
}


/* ========================== */
/*       cSparseLeasSqtAA     */
/* ========================== */

template<class Type>  class cSparseLeasSqtAA : public cLeasSq<Type>
{
      public :
	 typedef cCplIV<Type>             tCplIV;
	 typedef std::vector<tCplIV>      tLine;
	 typedef std::pair<Type,tLine>    tSWVect;  // sparse weighted vector
         typedef std::pair<Type,tSWVect*> tSCyWVect; //  tSWVect + coeff Y used for merge

         cSparseLeasSqtAA(int  aNbVar,double  aPerEmpty=4 );

       /// Here genereate an error
         void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
       /// Here memorize the obs
         void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;

          void Reset() override;
          cDenseVect<Type>  Solve() override;


	 /// Put bufferd line in matrixs, used at end or during filling to liberate memorry
	 void EmptyBuff() ;
      private :

	 std::list<tSWVect>   mBufInput;
	 double               mPerEmpty;
	 std::vector<tLine>   mtAA;    /// Som(W tA A)
         cDenseVect<Type>     mtARhs;  /// Som(W tA Rhs)
	 double               mNbInBuff;

};



template<class Type> 
    cSparseLeasSqtAA<Type>::cSparseLeasSqtAA(int aNbVar,double aPerEmpty) :
          cLeasSq<Type>       (aNbVar),
	  mPerEmpty           (aPerEmpty),
	  mtAA                (this->mNbVar),
          mtARhs              (this->mNbVar,eModeInitImage::eMIA_Null),
	  mNbInBuff           (0)
{
}

template<class Type> void cSparseLeasSqtAA<Type>::Reset()
{
    mBufInput.clear();
    for (auto & aLine : mtAA)
        aLine.clear();
    mtARhs.DIm().InitNull();
    mNbInBuff = 0;
}

template<class Type> cDenseVect<Type> cSparseLeasSqtAA<Type>::Solve()
{
   std::vector<cEigenTriplet<Type> > aVCoeff;            // list of non-zeros coefficients
   EmptyBuff();
   for (int aKy=0 ; aKy<int(mtAA.size()) ;aKy++)
   {
       for (const auto & aPair : mtAA.at(aKy))
       {
	       aVCoeff.push_back(cEigenTriplet<Type>(aPair.mInd,aKy,aPair.mVal));
       }
   }
   return EigenSolveCholeskyarseFromV3(aVCoeff,mtARhs);
}


template<class Type> void cSparseLeasSqtAA<Type>::AddObservation(const Type& ,const cDenseVect<Type> & ,const Type &  ) 
{
	MMVII_INTERNAL_ERROR("Used dense vector onn cSparseLeasSqtAA");
}

template<class Type> void  cSparseLeasSqtAA<Type>::AddObservation
                           (
                               const Type& aWeight,
			       const cSparseVect<Type> & aCoeff,
			       const Type &  aRHS
                           )
{
    mtARhs.WeightedAddIn(aWeight*aRHS,aCoeff);
    mBufInput.push_back(tSWVect(aWeight,aCoeff.IV()));
    mNbInBuff+= double(aCoeff.size());

    if (mNbInBuff >= (this->mNbVar*mPerEmpty))
       EmptyBuff();
}

template<class Type> void  cSparseLeasSqtAA<Type>::EmptyBuff() 
{

   // For each line, all equation it belongs to  [Cy tSWVect*=[W [ [i1 c1]  [i2 c2] ...]]]
   std::vector<std::list<tSCyWVect > >  aVListSW(this->NbVar());

   for (auto &  aSW : mBufInput)
   {
       for (const auto & aPair : aSW.second)
       {
	    aVListSW.at(aPair.mInd).push_back(tSCyWVect(aPair.mVal,&aSW));
       }
   }

   std::vector<bool>   aVOccupied(this->NbVar(),false);
   std::vector<Type>   aVSom(this->NbVar(),0.0);
   std::vector<int>    aUsedInd;

   for(int aKy=0 ; aKy<this->NbVar() ; aKy++)
   {
       tLine & aLine=  mtAA.at(aKy);    /// Som(W tA A)
       // Put the existing line in the buff struct
       for (const auto & anEl : aLine)
       {
           const int & anInd = anEl.mInd;
           aVOccupied.at(anInd) = true;
           aVSom.at(anInd) = anEl.mVal;
	   aUsedInd.push_back(anInd);
       }
       aLine.clear();

       //  transfer the equation  in the matrix
       for (const auto & aPtrVect : aVListSW.at(aKy))
       {
	       //  Weight * coeffiient of Y
            Type aMul = aPtrVect.first *aPtrVect.second->first;
	    for (const auto & aPair : aPtrVect.second->second)
	    {
                  const int & anInd = aPair.mInd;
		  if (anInd>=aKy)  // Only compute triangluar superior part
		  {
                      if (!aVOccupied.at(anInd) )
		      {
                          aVOccupied.at(anInd) = true;
		          aUsedInd.push_back(anInd);
		      }
                      aVSom.at(anInd) +=  aMul *  aPair.mVal;
		  }
	    }
       }

       // save the  buff in the matrix 
       for (const auto & anInd : aUsedInd)
       {
           aLine.push_back(tCplIV(anInd,aVSom[anInd]));
           aVOccupied.at(anInd) = false;
           aVSom.at(anInd) = 0.0;
       }
       aUsedInd.clear();
   }

   mNbInBuff =0;
   mBufInput.clear();
}

/* ======================= */
/*       cLeasSq           */
/* ======================= */


template<class Type>  cLeasSq<Type>::cLeasSq(int aNbVar):
     cSysSurResolu<Type> (aNbVar)
{
}

template<class Type> Type  cLeasSq<Type>::Residual
                             (
                                 const cDenseVect<Type> & aVect,
                                 const Type& aWeight,
                                 const cDenseVect<Type> & aCoeff,
                                 const Type &  aRHS
                             ) const
{
   return aWeight * Square(aVect.DotProduct(aCoeff)-aRHS);
}

template<class Type> cLeasSq<Type> * cLeasSq<Type>::AllocSparseLstSq(int aNbVar,double aPerEmptyBuf)
{
	return new cSparseLeasSqtAA<Type>(aNbVar,aPerEmptyBuf);
}

template<class Type> cLeasSq<Type> * cLeasSq<Type>::AllocDenseLstSq(int aNbVar)
{
	return new cLeasSqtAA<Type>(aNbVar);
}

/* ========================== */
/*       cSysSurResolu        */
/* ========================== */

template<class Type> cSysSurResolu<Type>::cSysSurResolu(int aNbVar) :
   mNbVar (aNbVar)
{
}

template<class Type> cSysSurResolu<Type>::~cSysSurResolu()
{
}

template<class Type> int cSysSurResolu<Type>::NbVar() const
{
   return mNbVar;
}

template<class Type> void cSysSurResolu<Type>::AddObsFixVar(const Type& aWeight,int aIndVal,const Type & aVal)
{
   cSparseVect<Type> aSpV;
   aSpV.AddIV(aIndVal,1.0);
   // static cIndSV<Type> & aIV = aSpV.IV()[0];
   // aIV.mInd  = aIndVal;
   // aIV.mVal  = 1.0;
   
   AddObservation(aWeight,aSpV,aVal);
}

template<class Type> void cSysSurResolu<Type>::AddObsFixVar(const Type& aWeight,const cSparseVect<Type> & aVVarVals)
{
    for (const auto & aP : aVVarVals)
        AddObsFixVar(aWeight,aP.mInd,aP.mVal);
}

template<class Type> void cSysSurResolu<Type>::AddObsFixVar (const Type& aWeight,const cDenseVect<Type>  &  aVRHS)
{
    MMVII_INTERNAL_ASSERT_medium(aVRHS.Sz() == mNbVar,"cSysSurResolu<Type>::AddObsFixVar");
    for (int aK=0 ; aK<mNbVar ; aK++)
        AddObsFixVar(aWeight,aK,aVRHS(aK));
}

template<class Type> cDenseVect<Type> cSysSurResolu<Type>::SparseSolve()
{
     return Solve();
}

template<class Type> cSysSurResolu<Type> * cSysSurResolu<Type>::AllocSSR(eModeSSR aMode,int aNbVar)
{
     switch (aMode)
     {
	     case eModeSSR::eSSR_LsqDense  :  return cLeasSq<Type>::AllocDenseLstSq (aNbVar);
	     case eModeSSR::eSSR_LsqSparse :  return cLeasSq<Type>::AllocSparseLstSq(aNbVar);
             
             default :;
     }

     MMVII_INTERNAL_ERROR("Bad enumerated valure for AllocSSR");
     return nullptr;
}



/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_LEASTSQ_TAA(Type)\
template  class  cSparseLeasSqtAA<Type>;\
template  class  cLeasSqtAA<Type>;\
template  class  cLeasSq<Type>;\
template  class  cSysSurResolu<Type>;


INSTANTIATE_LEASTSQ_TAA(tREAL4)
INSTANTIATE_LEASTSQ_TAA(tREAL8)
INSTANTIATE_LEASTSQ_TAA(tREAL16)


};


/* ========================== */
/*          cMatrix           */
/* ========================== */

