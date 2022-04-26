#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{

template <class Type> class  cSMLineTransf
{
	public :
	    typedef cCplIV<Type>             tCplIV;
	    typedef std::vector<tCplIV>      tLine;

	    cSMLineTransf(size_t aNb);
	    void InputSparseLine(const tLine & aLine);
            void SaveInSparseLine_And_Clear(tLine & aLine);
            void AddToCurLine(size_t anInd,const Type & aVal);
	private :
	    std::vector<Type>    mCumulLine;
            cSetIntDyn           mSet;
};

template <class Type> cSMLineTransf<Type>::cSMLineTransf(size_t aNb) :
     mCumulLine (aNb,0.0),
     mSet  (aNb)
{
}

template <class Type> void cSMLineTransf<Type>::InputSparseLine(const tLine & aLine)
{
    for (const auto & anEl : aLine)
    {
        const int & anInd = anEl.mInd;
        mSet.mOccupied.at(anInd) = true;
        mCumulLine.at(anInd) = anEl.mVal;
        mSet.mVIndOcc.push_back(anInd);
    }
}

template <class Type> void cSMLineTransf<Type>::SaveInSparseLine_And_Clear(tLine & aLine)
{
       aLine.clear();
       // save the  buff in the matrix 
       for (const auto & anInd : mSet.mVIndOcc)
       {
           aLine.push_back(tCplIV(anInd,mCumulLine[anInd]));
           mSet.mOccupied.at(anInd) = false;
           mCumulLine.at(anInd) = 0.0;
       }
       mSet.mVIndOcc.clear();
}

template <class Type> void cSMLineTransf<Type>::AddToCurLine(size_t anInd,const Type & aVal)
{
    if (!mSet.mOccupied.at(anInd) )
    {
        mSet.mOccupied.at(anInd) = true;
        mSet.mVIndOcc.push_back(anInd);
    }
    mCumulLine.at(anInd) +=  aVal;
}

template  class  cSMLineTransf<tREAL16>;

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


template<class Type>  class cSparseWeigtedVect
{
     public :
        cSparseWeigtedVect(const Type & aWeight,const std::vector<cCplIV<Type> > & aVect) :
		mWeight (aWeight),
		mVect   (aVect)
	{
	}

	Type                         mWeight;
	std::vector<cCplIV<Type> >   mVect;
};

template<class Type>  class cValAndSWVPtr
{
     public :
        cValAndSWVPtr(const Type & aVal,cSparseWeigtedVect<Type> *  aPtrV) :
		mVal  (aVal),
		mPtrV (aPtrV)
	{
	}

	const Type & Weight() const {return mPtrV->mWeight;}
	const std::vector<cCplIV<Type> > &   Vect() const {return mPtrV->mVect;}

	Type                         mVal;
	cSparseWeigtedVect<Type> *   mPtrV;
};

/**   Class for sparse least square usign normal equation
 
      A sparse normal matrix is computed.  This construction requires some kind of uncompression,
      as it is time consuming, this is not done at each equation. The equations are memorized
      in a buffer, and periodically the buffer is emptied in the normal matrix.

      The normal matric is full vector of sparse vector :

           mtAA[y] =   (x1,C1) (x2,C2) ....   where 
 */

template<class Type>  class cSparseLeasSqtAA : public cSparseLeasSq<Type>
{
      public :
         typedef cSparseWeigtedVect<Type> tWeigtedVect;
         typedef cValAndSWVPtr<Type>      tVal_WV;

	 typedef cCplIV<Type>             tCplIV;
	 typedef std::vector<tCplIV>      tLine;  


         cSparseLeasSqtAA(int  aNbVar,double  aPerEmpty=4 );

       /// Here memorize the obs
         void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;

          void Reset() override;
          cDenseVect<Type>  Solve() override;


	 /// Put bufferd line in matrixs, used at end or during filling to liberate memorry
	 void PutBufererEqInNormalMatrix() ;
      private :

	 std::list<tWeigtedVect>   mBufInput;
	 double                    mPerEmpty;
	 std::vector<tLine>        mtAA;    /// Som(W tA A)
         cDenseVect<Type>          mtARhs;  /// Som(W tA Rhs)
	 double                    mNbInBuff;

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
   PutBufererEqInNormalMatrix();
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
    mBufInput.push_back(tWeigtedVect(aWeight,aCoeff.IV()));
    mNbInBuff+= double(aCoeff.size());

    if (mNbInBuff >= (this->mNbVar*mPerEmpty))
       PutBufererEqInNormalMatrix();
}

template<class Type> void  cSparseLeasSqtAA<Type>::PutBufererEqInNormalMatrix() 
{

   // For each line y,  aVListSW[y]  will store all equation t belongs to it  
   std::vector<std::list<tVal_WV > >  aVListSW(this->NbVar());

   for (auto &  aSW : mBufInput)
   {
       for (const auto & aPair : aSW.mVect)
       {
	    aVListSW.at(aPair.mInd).push_back(tVal_WV(aPair.mVal,&aSW));
       }
   }

   cSMLineTransf<Type>  aLineTransf(this->NbVar());
   for(int aKy=0 ; aKy<this->NbVar() ; aKy++)
   {
       const auto & aListEQ = aVListSW.at(aKy);
       if (!aListEQ.empty())
       {
           tLine & aLine=  mtAA.at(aKy);    /// Som(W tA A)

           // Put the existing line in the buff struct
           aLineTransf.InputSparseLine(aLine);
      
           //  transfer all the equations  in the matrix
           for (const auto & aPtrVect : aListEQ)
           {
	       //  Weight * coeffiient of Y
                Type aMul = aPtrVect.mVal * aPtrVect.Weight();
	        for (const auto & aPair : aPtrVect.Vect())
	        {
                      const int & anInd = aPair.mInd;
		      if (anInd>=aKy)  // Only compute triangluar superior part
		      {
                          aLineTransf.AddToCurLine(anInd,aMul *  aPair.mVal);
		      }
	        }
           }

           // save the  buff in the matrix 
           aLineTransf.SaveInSparseLine_And_Clear(aLine);
       }
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
	 void AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&) override;


      private :
	 int                 mNbTmpVar;
	 std::vector<tTri>   mVTri;
	 std::vector<Type>   mVRhs;

};

template<class Type>  
   cSparseLeasSqGC<Type>::cSparseLeasSqGC(int aNbVar) :
	cSparseLeasSq<Type> (aNbVar),
	mNbTmpVar           (0)
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
    mNbTmpVar = 0;
}

template<class Type>  cDenseVect<Type>  cSparseLeasSqGC<Type>::Solve()
{
     cDenseVect<Type> aRes = EigenSolveLsqGC(mVTri,mVRhs,this->mNbVar+mNbTmpVar);
     // supress the temporary variables
     return aRes.SubVect(0,this->mNbVar);
}

template<class Type>  void  cSparseLeasSqGC<Type>::AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>& aSetSetEq) 
{
    aSetSetEq.AssertOk();

    size_t aNbTmp = aSetSetEq.AllEq().at(0).mTmpUK.size();
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
                     tTri aTri(mVRhs.size(),aSetEq.mVInd.at(aKUk),aVDer.at(aKUk)*aSW);
                     mVTri.push_back(aTri);
		 }
                 // For example parse the 3 unknown x,y,z of "temporary" point
		 for (size_t aKTmp=0 ; aKTmp<aNbTmp ; aKTmp++)
		 {
                     tTri aTri(mVRhs.size(),this->mNbVar+mNbTmpVar+aKTmp,aVDer.at(aNbUk+aKTmp)*aSW);
                     mVTri.push_back(aTri);
		 }
		 // Note the minus sign because we have a taylor expansion we need to annulate
                 mVRhs.push_back(-aSetEq.mVals.at(aKEq)*aSW);
	 }
    }
    mNbTmpVar += aNbTmp;
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

