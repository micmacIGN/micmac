#include "MMVII_SysSurR.h"
#include "MMVII_TplHeap.h"
#include "MMVII_Sys.h"


namespace MMVII
{
template <class Type> class  cSMLineTransf;        // for fast compress/uncompress of compressed line
template<class Type>  class cSparseWeigtedVect;    //  Sparse line +a weight
template<class Type>  class cValAndSWVPtr;         //  Pointer to sparse line + value(of the column)
template <class Type> class  cLineSparseLeasSqtAA; //  storage of sparse normal matrix
template<class Type>  class cSparseLeasSqtAA ;  // class for sparse least sq usign normal equation

/* *********************************** */
/*                                     */
/*            cSMLineTransf            */
/*                                     */
/* *********************************** */


/** Class used for uncompressing  the line  sparse matrix of  cSparseLeasSqtAA
    when they are modified
 
  */
template <class Type> class  cSMLineTransf : public cMemCheck
{
	public :
	    typedef cCplIV<Type>             tCplIV;
	    typedef std::vector<tCplIV>      tLine;

	    /// constructor, reserve plave in mSet and mCumulIne
	    cSMLineTransf(size_t aNb);

	    /// Input a compressed line in the uncompressed
	    void InputSparseLine(const tLine & aLine);
	    /// Modify the current line
            inline void AddToCurLine(size_t anInd,const Type & aVal);
	    /// Save the un-compressed line in compressed one
            void SaveInSparseLine_And_Clear(tLine & aLine);
	    /// Make an empty line of it
            void Clear();
	    /// Transfert the value in triplet format
            void TransfertInTriplet(std::vector<cEigenTriplet<Type>>&aV3,int anY);

	    /// number of elem !=0
            int NbElemNN() {return mSetNot0.NbElem();}
	private :
	    cSMLineTransf(const cSMLineTransf<Type>&) = delete;
	    std::vector<Type>    mCumulLine;   ///< uncompressed line, easily modifiable
            cSetIntDyn           mSetNot0;     ///< set of non null indexed
};


template <class Type> cSMLineTransf<Type>::cSMLineTransf(size_t aNb) :
     mCumulLine   (aNb,0.0),
     mSetNot0     (aNb )
{
}

template <class Type> void cSMLineTransf<Type>::InputSparseLine(const tLine & aLine)
{
    for (const auto & anEl : aLine)
    {
        const int & anInd = anEl.mInd;
        mSetNot0.mOccupied.at(anInd) = true;
        mCumulLine.at(anInd) = anEl.mVal;
        mSetNot0.mVIndOcc.push_back(anInd);
    }
}

template <class Type> void cSMLineTransf<Type>::SaveInSparseLine_And_Clear(tLine & aLine)
{
       aLine.clear();
       // save the  buff in the matrix 
       for (const auto & anInd : mSetNot0.mVIndOcc)
       {
           aLine.push_back(tCplIV(anInd,mCumulLine[anInd]));
           mSetNot0.mOccupied.at(anInd) = false;
           mCumulLine.at(anInd) = 0.0;
       }
       mSetNot0.mVIndOcc.clear();
}

template <class Type> void cSMLineTransf<Type>::Clear()
{
   for (const auto & anInd : mSetNot0.mVIndOcc)
   {
       mSetNot0.mOccupied.at(anInd) = false;
       mCumulLine.at(anInd) = 0.0;
   }
   mSetNot0.mVIndOcc.clear();
}

template <class Type> void cSMLineTransf<Type>::AddToCurLine(size_t anInd,const Type & aVal)
{
    mSetNot0.AddIndFixe(anInd);
    mCumulLine.at(anInd) +=  aVal;
}

template <class Type> void cSMLineTransf<Type>::TransfertInTriplet
                           (
	                        std::vector<cEigenTriplet<Type> > & aV3,
	                        int anY
                           )
{
       for (const auto & anInd : mSetNot0.mVIndOcc)
       {
	   if (mCumulLine[anInd]!=0) 
	       aV3.push_back(cEigenTriplet<Type>(anInd,anY,mCumulLine[anInd]));
       }
}


// template  class  cSMLineTransf<tREAL16>;

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

template<class Type> void cSparseLeasSq<Type>::AddObservation
                      (const Type& aW ,const cDenseVect<Type> & aDV ,const Type & aVal ) 
{
   // call to virtual  method, dont know why, compiler dont agre w/o cast 
    static_cast<cLinearOverCstrSys<Type> *>(this)-> AddObservation(aW,cSparseVect(aDV),aVal);
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

    /* ******************* */
    /*  cSparseWeigtedVect */
    /* ******************* */

/** Auxilary class for memorazing a weighted sparse vector in Buffer Input of cSparseLeasSqtAA */

template<class Type>  class cSparseWeigtedVect : public cMemCheck
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

    /* ******************* */
    /*    cValAndSWVPtr    */
    /* ******************* */

/** Auxilary class for  efficient indexation of Buffer input while uncompressing Buffer Input
    contain  a value and a weighted vector
  */
template<class Type>  class cValAndSWVPtr : public cMemCheck
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

/** Enum to store the state of a line */
enum class eLineSLSqtAA
{
      eLS_AlwaysDense,   ///< line that will be always dense (as probably internal parameters)
      eLS_TempoDense,    ///< line temporary dense
      eLS_TempoSparse    ///< line temporary sparse
};

    /* ************************** */
    /*    cLineSparseLeasSqtAA    */
    /* ************************** */

/**  class for storing the line of a sparse normal matrix (in fact any matrix sym). The representation is
     hybrid to try to have a good compromise between runtime and memory efficience. It can be in three state :

         *  dense, in this case, just a dense vector is memorized, this can be used for unknown
	   that are in relation with many other, typically the intrinsic parameters of a camera
	   (in case we dont have one/param by cams)

         * sparse, in fact temporary sparse, presented by a sparse vector of [i,val], will be 
	   uncompressed when it is used

	  * temporay dense,  the system maintain uncompressed a certain number of lines to avoid
	    repetitive compression/uncompression

 */
template <class Type> class  cLineSparseLeasSqtAA : public cMemCheck
{
      public :
	 typedef cCplIV<Type>             tCplIV;
	 typedef std::vector<tCplIV>      tSparseLine;  
	 Type *                           tDenseLine;
	 typedef cSMLineTransf<Type>      tTempoDenseLine;
         typedef cValAndSWVPtr<Type>      tVal_WV;
	 typedef cSparseLeasSqtAA<Type>   tMat;


         cLineSparseLeasSqtAA();
         cLineSparseLeasSqtAA(size_t aY,size_t  aNb,bool isAlwaysDense);
         ~cLineSparseLeasSqtAA();
         int NbElemNN() {return mTempoDenseLine->NbElemNN();}

	 void AddEquations(tMat & aMat,const std::list<tVal_WV > &);

	 /// save the dense reprensation in sparse mode en return the dense rep that can used by others
	 tTempoDenseLine *  MakeTempoSparse();

	 /// transfert to a dense representation
	 void  MakeTempoDense(tTempoDenseLine *);

         void AddLineMatr(size_t aYRed,const cDenseMatrix<Type> & atAAS,const std::vector<size_t> & );

         int & HeapIndex() {return mHeapIndex;}   ///< accessor
         eLineSLSqtAA State() const {return  mState;}   ///< accessor
	 bool CanRecycle() const {return mCanRecycle;}  ///< accessor
	 void SetCanRecycle(bool aCR) {mCanRecycle = aCR;}  ///< Modifier
	 /// Make equivalent to  a line with  only 0
	 void Clear(int aNbY);

         void TransfertInTriplet(std::vector<cEigenTriplet<Type> > &,int aNbVar);

      private :
          cLineSparseLeasSqtAA (const cLineSparseLeasSqtAA<Type> &) = delete;
	  Type & DLine(int anX) {return mDenseLine[anX];}
	  // Type & DLine(int anX) {return mDenseLine.at(anX-mY);}

	  size_t             mY;
	  bool               mCanRecycle;       ///< if in cur pack of buf, it must not be recycled
	  int                mHeapIndex;      ///< place for heap indexation
          eLineSLSqtAA       mState;          ///< state of use, dense for ever or temporary dense/sparse
          Type *             mDenseLine;      ///< "classical" vector of val, case  permanent dense line
	  // std::vector<Type>  mDenseLine;      ///< "classical" vector of val, case  permanent dense line
          tSparseLine        mSparseLine;     ///< vector of sparse representation [i,val], case tempo sparse
	  tTempoDenseLine *  mTempoDenseLine; ///<  pointer of shared dense rep (vec+indices), case tempo dense
};

    /* ************************** */
    /*        cCmpSLMPtr          */
    /*        cIndexSLMPtr        */
    /* ************************** */

template <class Type> class cCmpSLMPtr
{
    public : 
        typedef cLineSparseLeasSqtAA<Type> * tPtrLine;
        bool operator ()(const tPtrLine & aL1,const tPtrLine & aL2)  const 
        {
             if  (aL1->CanRecycle() ==  aL2->CanRecycle())
                return aL1->NbElemNN()   < aL2->NbElemNN();
             // else one is true, one is false ; if L1 can be recyled is has lower priority ...
	     return aL1->CanRecycle() ;
        }

};

template <class Type> class cIndexSLMPtr
{
    public : 
        typedef cLineSparseLeasSqtAA<Type> * tPtrLine;
	static void SetIndex(const tPtrLine & aPtr,tINT4 i) {aPtr->HeapIndex()=i;} 
        static int  GetIndex(const tPtrLine & aPtr) { return aPtr->HeapIndex(); }

};

// cSMLineTransf

/**   Class for sparse least square usign normal equation
 
      A sparse normal matrix is computed.  This construction requires some kind of uncompression,
      as it is time consuming, this is not done at each equation. The equations are memorized
      in a buffer, and periodically the buffer is emptied in the normal matrix.

      The normal matric is standard vector of sparse vector :

           mtAA[y] =   (x1,C1) (x2,C2) ....   where 
 */


template<class Type>  class cSparseLeasSqtAA : public cSparseLeasSq<Type>
{
      public :
         typedef cSparseWeigtedVect<Type>         tWeigtedVect;
         typedef cValAndSWVPtr<Type>              tVal_WV;
	 typedef cCplIV<Type>                     tCplIV;
	 typedef cLineSparseLeasSqtAA<Type>       tLine;
	 typedef cCmpSLMPtr<Type>                 tCmpLine;
	 typedef cIndexSLMPtr<Type>               tHeapInd;
	 typedef cIndexedHeap<tLine *,tCmpLine,tHeapInd>   tHeap;
	 typedef cSMLineTransf<Type>           tTempoDenseLine;

         cSparseLeasSqtAA(int  aNbVar,const cParamSparseNormalLstSq & aParam);
         ~cSparseLeasSqtAA();

       /// Here memorize the obs
         void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;

          void Reset() override;
          cDenseVect<Type>  Solve() override;


         void  AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>& aSetSetEq)  override;
	 /// Put bufferd line in matrixs, used at end or during filling to liberate memorry
	 void PutBufererEqInNormalMatrix() ;

	 ///  Set a temporary line, can be allocated new or recycled from an existing line 
         void  SetTempoDenseLine(tLine & aLine);
	 ///  Update the position in the heap
         void  HeapUpdate(tLine & aLine);


      private :

	 std::list<tWeigtedVect>   mBufInput;
	 double                    mPerEmpty;
	 std::vector<tLine *>      mtAA;    /// Som(W tA A)
         cDenseVect<Type>          mtARhs;  /// Som(W tA Rhs)
	 double                    mNbInBuff;
	 int                       mNbDLTempo; 
	 tCmpLine                  mCmpLine;
	 tHeap                     mHeapDL;
	 cBufSchurrSubst<Type>     mBufSchurr;
};

/* ******************************************** */
/*                                              */
/*               cLineSparseLeasSqtAA           */
/*                                              */
/* ******************************************** */


template <class Type> cLineSparseLeasSqtAA<Type>::cLineSparseLeasSqtAA(size_t aY,size_t  aNb,bool isAlwaysDense) :
	mY                (aY),
	mCanRecycle       (true),
	mHeapIndex        (HEAP_NO_INDEX),
	mState            (isAlwaysDense ? eLineSLSqtAA::eLS_AlwaysDense   : eLineSLSqtAA::eLS_TempoSparse),
	mDenseLine        (isAlwaysDense ? (cMemManager::Alloc<Type>(aY,aNb))   : nullptr),
	// mDenseLine        (isAlwaysDense ? ((aNb-aY))   : 0),
	mSparseLine       (),
	mTempoDenseLine   (nullptr)
{
     if (isAlwaysDense)
     {
	for (size_t aX= mY ; aX<aNb ; aX++)
            DLine(aX) = 0.0;
     }
}
template <class Type> cLineSparseLeasSqtAA<Type>::~cLineSparseLeasSqtAA()
{
    delete mTempoDenseLine;
    if (mState == eLineSLSqtAA::eLS_AlwaysDense )
    {
        cMemManager::Free(mDenseLine+mY);
    }
}

template <class Type>  cSMLineTransf<Type> * cLineSparseLeasSqtAA<Type>::MakeTempoSparse()
{
    mTempoDenseLine->SaveInSparseLine_And_Clear(mSparseLine);
    cSMLineTransf<Type> * aRes = mTempoDenseLine;
    mTempoDenseLine = nullptr;
    mState   =  eLineSLSqtAA::eLS_TempoSparse;

    return aRes;
}

template <class Type>  void cLineSparseLeasSqtAA<Type>::MakeTempoDense(tTempoDenseLine * aTDL)
{
     mTempoDenseLine = aTDL;
     mTempoDenseLine->InputSparseLine(mSparseLine);
     mSparseLine.clear();
     mState   =  eLineSLSqtAA::eLS_TempoDense;

}    

template <class Type> 
    void cLineSparseLeasSqtAA<Type>::TransfertInTriplet
         (
	      std::vector<cEigenTriplet<Type> > & aV3,
	      int aNbVar
         )
{
     if (mState==eLineSLSqtAA::eLS_AlwaysDense)
     {
        for (int aX=mY ;aX< aNbVar ; aX++)
            if (DLine(aX))
	    {
	       aV3.push_back(cEigenTriplet<Type>(aX,mY,DLine(aX)));
	    }
     }
     else if (mState==eLineSLSqtAA::eLS_TempoSparse)
     {
          for (const auto & aPair : mSparseLine)
          {
               if (aPair.mVal!=0)
	          aV3.push_back(cEigenTriplet<Type>(aPair.mInd,mY,aPair.mVal));
          }
     }
     else if  (mState==eLineSLSqtAA::eLS_TempoDense)
     {
           mTempoDenseLine->TransfertInTriplet(aV3,mY);
     }
}

template <class Type> void cLineSparseLeasSqtAA<Type>::Clear(int aNbVar)
{
     if (mState==eLineSLSqtAA::eLS_AlwaysDense)
     {
        for (int aX=mY ;aX< aNbVar ; aX++)
            DLine(aX) = 0.0;
     }
     else if (mState==eLineSLSqtAA::eLS_TempoSparse)
     {
          mSparseLine.clear();
     }
     else if  (mState==eLineSLSqtAA::eLS_TempoDense)
     {
           mTempoDenseLine->Clear();
     }
}
template <class Type> 
   void cLineSparseLeasSqtAA<Type>::AddLineMatr(size_t aYRed,const cDenseMatrix<Type> & atAAS,const std::vector<size_t> & aVInd)
{
     if (mState==eLineSLSqtAA::eLS_AlwaysDense)
     {
         for (size_t aXRed = 0 ; aXRed<aVInd.size() ; aXRed++)
         {
             size_t anX = aVInd.at(aXRed);
             if (anX>=mY)  // Only compute triangluar superior part
             {
                 DLine(anX) += atAAS.GetElem(aXRed,aYRed);
             }
         }
     }
     else if (mState==eLineSLSqtAA::eLS_TempoDense)
     {
         for (size_t aXRed = 0 ; aXRed<aVInd.size() ; aXRed++)
         {
             size_t anX = aVInd.at(aXRed);
             if (anX>=mY)  // Only compute triangluar superior part
             {
                mTempoDenseLine->AddToCurLine(anX,atAAS.GetElem(aXRed,aYRed));
             }
         }
     }
     else
     {
     }
}

template <class Type> 
   void cLineSparseLeasSqtAA<Type>::AddEquations
        (
	     tMat & aMat,
             const std::list<tVal_WV > & aLEq
        )
{
     if (mState==eLineSLSqtAA::eLS_AlwaysDense)
     {
          for (const auto & aPtrVect : aLEq)
          {
	       //  Weight * coeffiient of Y
                Type aMul = aPtrVect.mVal * aPtrVect.Weight();
	        for (const auto & aPair : aPtrVect.Vect())
	        {
                      const int & anInd = aPair.mInd;
		      if (size_t(anInd)>=mY)  // Only compute triangluar superior part
		      {
                          DLine(anInd) += aMul *  aPair.mVal;
		      }
	        }
          }
     }
     else
     {
         if (mState==eLineSLSqtAA::eLS_TempoSparse)
	 {
            aMat.SetTempoDenseLine(*this);
	 }
         for (const auto & aPtrVect : aLEq)
         {
	     //  Weight * coeffiient of Y
             Type aMul = aPtrVect.mVal * aPtrVect.Weight();
	     for (const auto & aPair : aPtrVect.Vect())
	     {
                  const int & anInd = aPair.mInd;
                  if (size_t(anInd)>=mY)  // Only compute triangluar superior part
                  {
                      mTempoDenseLine->AddToCurLine(anInd,aMul *  aPair.mVal);
                  }
	     }
         }
	 // The number of equation !=0 may have change, so need uddate in heap
	 aMat.HeapUpdate(*this);
     }
}

/* ******************************************** */
/*                                              */
/*               cParamSparseNormalLstSq        */
/*                                              */
/* ******************************************** */

cParamSparseNormalLstSq::cParamSparseNormalLstSq
(
      double aPerEmptyBuf,
      size_t aNbMaxRangeDense,
      size_t aNbBufDense
)  :
	mPerEmptyBuf      (aPerEmptyBuf),
	mIndMaxRangeDense (aNbMaxRangeDense),
	mNbBufDense       (std::max(size_t(1),aNbBufDense))  // need at least one buf dense ...
{
}
cParamSparseNormalLstSq::cParamSparseNormalLstSq () :
      cParamSparseNormalLstSq (4.0,0,13)
{
}

/* ******************************************** */
/*                                              */
/*               cSparseLeasSqtAA               */
/*                                              */
/* ******************************************** */

template<class Type> 
    cSparseLeasSqtAA<Type>::cSparseLeasSqtAA
    (
                int aNbVar,
                const cParamSparseNormalLstSq & aParam
     ) :
          cSparseLeasSq<Type> (aNbVar),
	  mPerEmpty           (aParam.mPerEmptyBuf),
          mtARhs              (this->mNbVar,eModeInitImage::eMIA_Null),
	  mNbInBuff           (0),
	  mNbDLTempo          (aParam.mNbBufDense),
	  mCmpLine            (),
	  mHeapDL             (mCmpLine),
	  mBufSchurr          (aNbVar)
{
    mtAA.reserve(this->mNbVar);
    cSetIntDyn aSetDense(this->mNbVar,aParam.mVecIndDense);

    for (size_t aY=0 ; aY<size_t(this->mNbVar) ; aY++)
    {
        mtAA.push_back(new tLine(aY,this->mNbVar,(aY<aParam.mIndMaxRangeDense) || aSetDense.mOccupied.at(aY)));
    }
}

template<class Type> cSparseLeasSqtAA<Type>::~cSparseLeasSqtAA()
{
      DeleteAllAndClear(mtAA);
}


template<class Type> void  cSparseLeasSqtAA<Type>::SetTempoDenseLine(tLine & aL2Dense)
{
   tTempoDenseLine *  aTDL = nullptr;
   tLine ** aLow = mHeapDL.Lowest();
   // We reuse the current dense line if heap is not empty and lowest value is not in current use and heap has reach limit
   if ((aLow!=nullptr) && (*aLow)->CanRecycle() &&  (mHeapDL.Sz() >= mNbDLTempo))
   {
         // !!  two following line dont commute, as aLow points to Heap[0], if pop, this alterate Low ...
	 aTDL = (*aLow)->MakeTempoSparse();   // save its value in sparse representation and return the dense freeed
         mHeapDL.Pop();
   }
   else
   {
         aTDL = new tTempoDenseLine(this->mNbVar); // else allocate new dense vect
   }

   aL2Dense.MakeTempoDense(aTDL); // Put sparse rep in dense
   mHeapDL.Push(&aL2Dense);  // Put it in the heap for possible recycling
}

template<class Type> void  cSparseLeasSqtAA<Type>::HeapUpdate(tLine & aDenseL)
{
   mHeapDL.UpDate(&aDenseL);
}

template<class Type> void cSparseLeasSqtAA<Type>::Reset()
{
    mBufInput.clear();
    for (auto & aLine : mtAA)
        aLine->Clear(this->mNbVar);
    mtARhs.DIm().InitNull();
    mNbInBuff = 0;
    // Nothing to do for the heap as all dense line have 0 elemt, the heap struct is still OK

}

template<class Type> cDenseVect<Type> cSparseLeasSqtAA<Type>::Solve()
{
   std::vector<cEigenTriplet<Type> > aVCoeff;            // list of non-zeros coefficients
   PutBufererEqInNormalMatrix();
   for (auto & aLine : mtAA)
   {
       aLine->TransfertInTriplet(aVCoeff,this->mNbVar);
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
    // modify the vector RHS
    mtARhs.WeightedAddIn(aWeight*aRHS,aCoeff);
    // memorize the equation in the list mBufInput
    mBufInput.push_back(tWeigtedVect(aWeight,aCoeff.IV()));
    mNbInBuff+= double(aCoeff.size());

    // if too many equation memorized, save them in normal matrix
    if (mNbInBuff >= (this->mNbVar*mPerEmpty))
       PutBufererEqInNormalMatrix();
}

template<class Type> void  cSparseLeasSqtAA<Type>::PutBufererEqInNormalMatrix() 
{
   // make an indexation of all buffered equations
   // For each line y,  aVListSW[y]  will store all equation that belongs to it  
   std::vector<std::list<tVal_WV > >  aVListSW(this->NbVar());

   for (auto &  aSW : mBufInput)
   {
       for (const auto & aPair : aSW.mVect)
       {
	    aVListSW.at(aPair.mInd).push_back(tVal_WV(aPair.mVal,&aSW));
       }
   }

   for(int aKy=0 ; aKy<this->NbVar() ; aKy++)
   {
       const auto & aListEQ = aVListSW.at(aKy);
       if (!aListEQ.empty())
       {
            mtAA.at(aKy)->AddEquations(*this,aListEQ);
       }
   }

   mNbInBuff =0;
   mBufInput.clear();
}


template<class Type>  void  cSparseLeasSqtAA<Type>::AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>& aSetSetEq) 
{
// StdOut() << "cSparseLeasSqtAA<Type>::AddObsWithTmpUK \n";

    //  1 - Compute the reduce schurr matrix
    mBufSchurr.CompileSubst(aSetSetEq);
    const std::vector<size_t> & aVInd = mBufSchurr.VIndexUsed();


        // 1.2 is mNbDLTempo was over dimensionned, make it grow,  no harm ...
    mNbDLTempo = std::max(mNbDLTempo,(int)mBufSchurr.VIndexUsed().size());

    // 2 - Uncompress the line used in this substitution, try to mininize work, often they would be already uncompress from 
    // previous jobs

        // 2-1 SetCanRecycle to  all non always dense, so they have higher priority (so update heap), memorize sparse line
    std::vector<tLine *> aVSparseL;
    for (const auto & anInd : aVInd)
    {
        tLine * aLine = mtAA.at(anInd);
	if (aLine->State() != eLineSLSqtAA::eLS_AlwaysDense)  // Nothing to dow for always dense lines
	{
	    aLine->SetCanRecycle(false);   // The line used cannot be recycled as longs as this step was not processed
	    if (aLine->State() == eLineSLSqtAA::eLS_TempoDense)
	    {
                 mHeapDL.UpDate(aLine);
	    }
	    else
            {
                 aVSparseL.push_back(aLine);
            }
	}
    }

        // 2-2 Put the sparse line in dense representation
    for (const auto & aPtrSL :aVSparseL)
    {
        SetTempoDenseLine(*aPtrSL);
    }

    // 3 - Update the heap, 
    //     const std::vector<size_t> &  aVI = mBSC->VIndexUsed();

    const cDenseVect<Type> &   atARhsS = mBufSchurr.tARhsSubst() ;
    const cDenseMatrix<Type> & atAAS =   mBufSchurr.tAASubst() ;

    for (size_t aIndRed=0 ;  aIndRed<aVInd.size() ; aIndRed++)
    {
        size_t anInd = aVInd.at(aIndRed);
        mtARhs(anInd) += atARhsS(aIndRed);
        tLine * aLine = mtAA.at(anInd);
	aLine->AddLineMatr(aIndRed,atAAS,aVInd);
	FakeUseIt(aLine);
    }


    // 4 - Update the heap, 
    for (const auto & anInd : mBufSchurr.VIndexUsed())
    {
        tLine * aLine = mtAA.at(anInd);
	if (aLine->State() != eLineSLSqtAA::eLS_AlwaysDense)  // Nothing to dow for always dense lines
	{
	    aLine->SetCanRecycle(true);   // The line can be recycled
            mHeapDL.UpDate(aLine);        // take into account in heap-priority
	}
    }
	// MMVII_INTERNAL_ERROR("SparseLeasSqtAA<Type>::AddObsWithTmpUK");
}
/*
enum class eLineSLSqtAA
{
      eLS_AlwaysDense,   ///< line that will be always dense (as probably internal parameters)
      eLS_TempoDense,    ///< line temporary dense
      eLS_TempoSparse    ///< line temporary sparse
};
*/

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

    // For example parse all the camera seening a point
    for (const auto & aSetEq : aSetSetEq.AllEq())
    {
         // For example parse the two equation on i,j residual
         for (size_t aKEq=0 ; aKEq<aSetEq.mVals.size() ; aKEq++)
	 {
		 const std::vector<Type> & aVDer = aSetEq.mDers.at(aKEq);
		 Type aSW = Sqrt(aSetEq.WeightOfKthResisual(aKEq));

                 size_t  aIndTmp =  this->mNbVar+mNbTmpVar;
		 for (size_t aKGlob=0 ; aKGlob<aSetEq.mGlobVInd.size() ; aKGlob++)
                 {
                     int aInd = aSetEq.mGlobVInd[aKGlob];
                     if (cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd))
                        aInd = aIndTmp + cSetIORSNL_SameTmp<Type>::ToIndTmp(aInd);
                     tTri aTri(mVRhs.size(),aInd,aVDer.at(aKGlob)*aSW);
                     mVTri.push_back(aTri);
                 }
		 // Note the minus sign because we have a taylor expansion we need to annulate
                 mVRhs.push_back(-aSetEq.mVals.at(aKEq)*aSW);
	 }
    }
    // size_t aNbTmp = aSetSetEq.AllEq().at(0).mVTmpUK.size();
    mNbTmpVar += aSetSetEq.NbTmpUk();
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

template<class Type> cLeasSq<Type> * cLeasSq<Type>::AllocSparseNormalLstSq(int aNbVar,const cParamSparseNormalLstSq & aParam)
{
	return new cSparseLeasSqtAA<Type>(aNbVar,aParam);
}



/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_LEASTSQ_TAA(Type)\
template  class  cSparseLeasSqtAA<Type>;\
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

