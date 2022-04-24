#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{


/* *********************************** */
/*                                     */
/*            cSparseLeasSq            */
/*                                     */
/* *********************************** */

/**  Mother class for sparse least square systems */

template<class Type>  class cSparseLeasSq : public cLeasSq<Type>
{
      public :
       /// Here genereate an error, no need to handle dense vector in sparse systems
         void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
         cSparseLeasSq(int  aNbVar);
};

template<class Type> void cSparseLeasSq<Type>::AddObservation(const Type& ,const cDenseVect<Type> & ,const Type &  ) 
{
	MMVII_INTERNAL_ERROR("Used dense vector onn cSparseLeasSqtAA");
}

template<class Type> 
    cSparseLeasSq<Type>::cSparseLeasSq(int aNbVar) :
          cLeasSq<Type> (aNbVar)
{
}

/* *********************************** */
/*                                     */
/*            cSparseLeasSqtAA         */
/*                                     */
/* *********************************** */

/**   Class for sparse least square usign normal equation
 
      A sparse normal matrix is computed.  This construction requires some kind of uncompression,
      as it is time consuming, this is not done at each equation. The equations are memorized
      in a buffer, and periodically the buffer is emptied in the normal matrix.
 */

template<class Type>  class cSparseLeasSqtAA : public cSparseLeasSq<Type>
{
      public :
	 typedef cCplIV<Type>             tCplIV;
	 typedef std::vector<tCplIV>      tLine;
	 typedef std::pair<Type,tLine>    tSWVect;  // sparse weighted vector
         typedef std::pair<Type,tSWVect*> tSCyWVect; //  tSWVect + coeff Y used for merge

         cSparseLeasSqtAA(int  aNbVar,double  aPerEmpty=4 );

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
          cSparseLeasSq<Type> (aNbVar),
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

/* *********************************** */
/*                                     */
/*            cSparseLeasSqGC          */
/*                                     */
/* *********************************** */

/**  Class for sparse least square w/o computing the normal matrix, the system is solved
     by minimization using conjugate gradient.  Basically the observation are converted
     in triplet (eigen requirement) and memorized, at the end a call is done to eigen
     library using the vector of triplets.
*/

template<class Type>  class cSparseLeasSqGC : public cSparseLeasSq<Type>
{
      public :
	 typedef cEigenTriplet<Type>   tTri;

         cSparseLeasSqGC(int  aNbVar );

       /// Here memorize the obs
         void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;

         void Reset() override;
         cDenseVect<Type>  Solve() override;


	 /// Here the temporay are in fact processed like standards equation, they decoded an memorized
	 void AddObsWithTmpK(const cSetIORSNL_SameTmp<Type>&) override;


      private :
	 int                 mNbVarTmp;
	 std::vector<tTri>   mVTri;
	 std::vector<Type>   mVRhs;

};

template<class Type>  
   cSparseLeasSqGC<Type>::cSparseLeasSqGC(int aNbVar) :
	cSparseLeasSq<Type> (aNbVar),
	mNbVarTmp           (0)
{
}

template<class Type>  void cSparseLeasSqGC<Type>::AddObservation
                           (
                               const Type& aWeight,
                               const cSparseVect<Type> & aCoeff,
                               const Type &  aRHS
			    ) 
{

    Type aSW = Sqrt(aWeight);
    for (const auto & aCpl : aCoeff.IV())
    {
        tTri aTri(mVRhs.size(),aCpl.mInd,aCpl.mVal*aSW);
        mVTri.push_back(aTri);
    }
    mVRhs.push_back(aSW*aRHS);
}

template<class Type>  void cSparseLeasSqGC<Type>::Reset() 
{
    mVTri.clear();
    mVRhs.clear();
    mNbVarTmp = 0;
}

template<class Type>  cDenseVect<Type>  cSparseLeasSqGC<Type>::Solve()
{
	return EigenSolveLsqGC(mVTri,mVRhs,this->mNbVar);
}

template<class Type>  void  cSparseLeasSqGC<Type>::AddObsWithTmpK(const cSetIORSNL_SameTmp<Type>& aSetSetEq) 
{
    aSetSetEq.AssertOk();

     size_t aNbTmp = aSetSetEq.AllEq()[0].mTmpUK.size();
    // For example parse all the camera seening a point
    for (const auto & aSetEq : aSetSetEq.AllEq())
    {
         size_t aNbUk = aSetEq.mVInd.size();
         // For example parse the two equation on i,j residual
         for (size_t aKEq=0 ; aKEq<aSetEq.mVals.size() ; aKEq++)
	 {
		 const std::vector<Type> & aVDer = aSetEq.mDers.at(aKEq);
		 Type aSW = Sqrt(aSetEq.WeightOfKthResisual(aKEq));

                 // For example parse the intrinsic & extrinsic parameters
		 for (size_t aKUk=0 ; aKUk<aNbUk ; aKUk++)
		 {
                     tTri aTri(mVRhs.size(),aSetEq.mVInd[aKUk],aVDer[aKUk]*aSW);
                     mVTri.push_back(aTri);
		 }
                 // For example parse the 3 unknown x,y,z of "temporary" point
		 for (size_t aKTmp=0 ; aKTmp<aNbTmp ; aKTmp++)
		 {
                     tTri aTri(mVRhs.size(),aKTmp+mNbVarTmp,aVDer[aNbUk+aKTmp]*aSW);
                     mVTri.push_back(aTri);
		 }
                 mVRhs.push_back(aSetEq.mVals.at(aKEq)*aSW);
	 }
    }
    mNbVarTmp += aNbTmp;
}


/* *********************************** */
/*                                     */
/*            cLeasSq                  */
/*                                     */
/* *********************************** */

template<class Type> cLeasSq<Type> * cLeasSq<Type>::AllocSparseGCLstSq(int aNbVar)
{
	return new cSparseLeasSqGC<Type>(aNbVar);
}

template<class Type> cLeasSq<Type> * cLeasSq<Type>::AllocSparseNormalLstSq(int aNbVar,double aPerEmptyBuf)
{
	return new cSparseLeasSqtAA<Type>(aNbVar,aPerEmptyBuf);
}



/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_LEASTSQ_TAA(Type)\
template  class  cSparseLeasSqGC<Type>;\
template  class  cSparseLeasSq<Type>;\
template  class  cLeasSq<Type>;


INSTANTIATE_LEASTSQ_TAA(tREAL4)
INSTANTIATE_LEASTSQ_TAA(tREAL8)
INSTANTIATE_LEASTSQ_TAA(tREAL16)


};


/* ========================== */
/*          cMatrix           */
/* ========================== */

