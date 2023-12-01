#ifndef _COMMON_HEADER_SYMBDER_H_
#define _COMMON_HEADER_SYMBDER_H_

/** 
   \brief contain functionnality that are required for code gen and need micma lib
*/

#include "MMVII_Ptxd.h"
#include "SymbDer/SymbolicDerivatives.h"
#include <typeinfo>       // operator typeid

using namespace NS_SymbolicDerivative;


namespace MMVII
{
/// required so that we can define points on formula ...

template <> class tNumTrait<cFormula <tREAL8> >
{
    public :
        // For these type rounding mean something
        // static bool IsInt() {return true;}
        typedef cFormula<tREAL8>  tBase;
        typedef cFormula<tREAL8>  tBig;
        typedef cFormula<tREAL8>  tFloatAssoc;
        static void AssertValueOk(const cFormula<double> & ) {}
};

template <class Type> class cMatF
{
    private :
         template <class TypeFunc>  cMatF<Type>  OpBin(const cMatF<Type> &  aM2,const TypeFunc & Oper) const
         {
              MMVII_INTERNAL_ASSERT_always(mSz==aM2.mSz,"Sz diff in op bin");
              
	      cMatF<Type>  aRes(mSz.x(),mSz.y(),mC0);

              for (int aKy= 0 ; aKy<mSz.y() ; aKy++)
                   for (int aKx= 0 ; aKx<mSz.x() ; aKx++)
			   aRes(aKx,aKy) =  Oper((*this)(aKx,aKy),aM2(aKx,aKy));

	      return aRes;
	 }


    public :

      typedef std::vector<Type> tLine;

      cMatF(size_t aSzX,size_t aSzY,const Type & aVal) :
         mSz   (aSzX,aSzY),
	 mMatr (mSz.y(),tLine(mSz.x(),aVal)),
         mC0   (CreateCste(0.0,aVal)) 
      {
      }
      const Type & operator () (size_t anX,size_t anY) const {return mMatr.at(anY).at(anX);}
      Type & operator () (size_t anX,size_t anY) {return mMatr.at(anY).at(anX);}

      void  PushInVect(std::vector<Type> & aRes) const 
      {
          for (int aKy= 0 ; aKy<mSz.y() ; aKy++)
              for (int aKx= 0 ; aKx<mSz.x() ; aKx++)
		  aRes.push_back((*this)(aKx,aKy));
      }
      std::vector<Type> ToVect() const
      {
          std::vector<Type> aRes;
          PushInVect(aRes);
          return aRes;
      }

      cMatF(size_t aSzX,size_t aSzY, const std::vector<Type> & aVal,size_t anInd0) :
           cMatF(aSzX,aSzY,aVal.at(anInd0))
      {
          for (int aKy= 0 ; aKy<mSz.y() ; aKy++)
	  {
              for (int aKx= 0 ; aKx<mSz.x() ; aKx++)
	      {
		      (*this)(aKx,aKy) = aVal.at(anInd0++);
	      }
	  }
      }

      cMatF<Type> Transpose() const
      {
          cMatF<Type> aRes(mSz.y(),mSz.x(),mC0);
          for (int aKy= 0 ; aKy<mSz.y() ; aKy++)
              for (int aKx= 0 ; aKx<mSz.x() ; aKx++)
                  aRes(aKy,aKx) = (*this)(aKx,aKy);

	  return aRes;
      }

      static cMatF<Type>  MatAxiator(const cPtxd<Type,3> & aW)
      {
          //  cPtxd<T,3>(  0    , -W.z() ,  W.y() ),
          //  cPtxd<T,3>( W.z() ,   0    , -W.x() ),
          //  cPtxd<T,3>(-W.y() ,  W.x() ,   0    )
          cMatF<Type> aMat (3,3,CreateCste(1.0,aW.x()));

          aMat(1,0) = -aW.z();
          aMat(2,0) =  aW.y();
          aMat(2,1) = -aW.x();

          aMat(0,1) =  -aMat(1,0);
          aMat(0,2) =  -aMat(2,0);
          aMat(1,2) =  -aMat(2,1);

          return aMat;
      }

      template <const int  aDim> cPtxd<Type,aDim> operator * (const cPtxd<Type,aDim> & aPt) const
      {
           //      - - -      #      -
           //    y # # #      #   -> #
           //      - - -      #      -
           cPtxd<Type,aDim> aRes  = cPtxd<Type,aDim>::PCste(mC0);
	   MMVII_INTERNAL_ASSERT_always((aDim==mSz.x()) && (aDim==mSz.y()),"Sz in operator  cMatF * aPt");

	   for (int aY = 0 ; aY<mSz.y() ; aY++)
	   {
	        for (int aX = 0 ; aX<mSz.x() ; aX++)
		{
                     aRes[aY]  = aRes[aY]  +  aPt[aX] * (*this)(aX,aY);
		}
	   }

	   return aRes;
      }

      cMatF<Type> operator * (const cMatF<Type> & aMatOper) const
      {
           //                  x
           //      - - -     - #        - -
           //    y # # #     - #    =   - #
           //      - - -     - #        - -
           //      - - -                - -
	   const cMatF<Type> & aM1 = *this;  // to symetrize notation
	   const cMatF<Type> & aM2 = aMatOper;
	  
	   MMVII_INTERNAL_ASSERT_always(aM1.mSz.x()==aM2.mSz.y(),"Bas size in matrix operator * ");
	   cMatF<Type> aRes(aM2.mSz.x(),aM1.mSz.y(),mC0);

           for (int aX=0 ; aX<aRes.mSz.x() ; aX++)
               for (int aY=0 ; aY<aRes.mSz.y() ; aY++)
	       {
                   Type & aElem = aRes(aX,aY);
                   for (int aK=0 ; aK<aM2.mSz.y() ; aK++)
	           {
		       aElem = aElem + aM1(aK,aY) * aM2(aX,aK);
	           }
	       }

	   return aRes;

      }

      cMatF<Type> operator - (const cMatF<Type> & aM2) const 
      {
	       return this->OpBin(aM2,[](const Type & A,const Type &B) {return A-B;});
      }

      cMatF<Type> operator + (const cMatF<Type> & aM2) const 
      {
	       return this->OpBin(aM2,[](const Type & A,const Type &B) {return A+B;});
      }

      cPt2di             mSz;
      std::vector<tLine> mMatr;     
      Type               mC0;
};




template <class Type> Type SqNormL2V2(const Type & aX,const Type & aY)
{
    return Square(aX) + Square(aY);
}
template <class Type> Type SqNormL2V3(const Type & aX,const Type & aY,const Type & aZ)
{
    return Square(aX) + Square(aY) + Square(aZ);
}


template <class Type> Type NormL2V2(const Type & aX,const Type & aY)
{
    return sqrt(SqNormL2V2(aX,aY));
}
template <class Type> Type NormL2V3(const Type & aX,const Type & aY,const Type & aZ)
{
    return sqrt(SqNormL2V3(aX,aY,aZ));
}

template <class Type> Type NormL2Vec2(const std::vector<Type> & aVec)
{
    return NormL2V2(aVec.at(0),aVec.at(1));
}



template  <typename tScal> std::vector<tScal> ToVect(const cPtxd<tScal,3> & aPt)
{
     return  {aPt.x(),aPt.y(),aPt.z()};
}
template  <typename tScal> std::vector<tScal> ToVect(const cPtxd<tScal,2> & aPt)
{
     return {aPt.x(),aPt.y()};
}



template  <typename tScal> tScal PScal(const cPtxd<tScal,3> & aP1,const cPtxd<tScal,3> & aP2)
{
         return aP1.x()*aP2.x() + aP1.y() *aP2.y() + aP1.z() * aP2.z();
}
template  <typename tScal> cPtxd<tScal,3>  VtoP3(const  std::vector<tScal> & aV,size_t aInd=0)
{
        return cPtxd<tScal,3>(aV.at(aInd),aV.at(aInd+1),aV.at(aInd+2));
}
template  <typename tScal> cPtxd<tScal,2>  VtoP2(const  std::vector<tScal> & aV,size_t aInd=0)
{
        return cPtxd<tScal,2>(aV.at(aInd),aV.at(aInd+1));
}

template  <typename tScal> cPtxd<tScal,3>   MulMat(const std::vector<tScal> & aV,size_t aInd,const  cPtxd<tScal,3> & aP)
{
     cPtxd<tScal,3> aL1 =  VtoP3(aV,aInd);
     cPtxd<tScal,3> aL2 =  VtoP3(aV,aInd+3);
     cPtxd<tScal,3> aL3 =  VtoP3(aV,aInd+6);

     return cPtxd<tScal,3>(PScal(aP,aL1),PScal(aP,aL2),PScal(aP,aL3));
}

template  <typename tScal> tScal   MatVal(const std::vector<tScal> & aV,size_t aInd,size_t aX,size_t aY,size_t aSzMat=3)
{
   return aV.at(aInd+aX+aY*aSzMat);
}
template  <typename tScal> tScal & MatVal(std::vector<tScal> & aV,size_t aInd,size_t aX,size_t aY,size_t aSzMat=3)
{
   return aV.at(aInd+aX+aY*aSzMat);
}

template  <typename tScal> 
     std::vector<tScal>  MulMat(const std::vector<tScal> & aV1,size_t aInd1,const std::vector<tScal> & aV2,size_t aInd2,size_t aSzMat=3)
{
    std::vector<tScal>  aRes(aSzMat*aSzMat);

    //                  x
    //      - - -     - # -
    //    y # # #     - # -
    //      - - -     - # -

    for (size_t aKX=0 ; aKX<aSzMat ; aKX++)
    {
         for (size_t aKY=0 ; aKY<aSzMat ; aKY++)
         {
             tScal aSom = 0.0;
             for (size_t aK=0 ; aK<aSzMat ; aK++)
	     {
                 tScal aM1 = MatVal(aV1,aInd1,aK,aKY,aSzMat);
                 tScal aM2 = MatVal(aV2,aInd2,aKX,aK,aSzMat);
		 aSom = aSom + aM1*aM2;
	     }
             MatVal(aRes,0,aKX,aKY,aSzMat) = aSom;
         }
    }

    return aRes;
}

template  <typename tScal> 
     std::vector<tScal>  Transpose(const std::vector<tScal> & aVect,size_t aInd,size_t aSzMat=3)
{
    std::vector<tScal>  aRes(aSzMat*aSzMat);
    for (size_t aKX=0 ; aKX<aSzMat ; aKX++)
    {
         for (size_t aKY=0 ; aKY<aSzMat ; aKY++)
         {
             MatVal(aRes,0,aKX,aKY,aSzMat) =  MatVal(aVect,aInd,aKY,aKX,aSzMat);
         }
    }

    return aRes;
}


/** this class represent a Pose on  forumla (or real if necessary)
 *    It contains a Center and the rotation Matrix IJK
 */
template <class Type> class cPoseF
{
     public :

        cPoseF(const cPtxd<Type,3> & aCenter,const cMatF<Type> & aMat) :
             mCenter  (aCenter),
             mIJK     (aMat)
        {
        }


        cPoseF(const std::vector<Type> &  aVecUk,size_t aK0Uk,const std::vector<Type> &  aVecObs,size_t aK0Obs,bool WithAxiator) :
            cPoseF<Type>
            (
                 VtoP3(aVecUk,aK0Uk),
             //  The matrix is Current matrix *  Axiator(-W) , the "-" in omega comming from initial convention
             //  See  cPoseWithUK::OnUpdate()  &&  cEqColinearityCamPPC::formula
                 (WithAxiator                                                                              ?  
                      cMatF<Type>(3,3,aVecObs,aK0Obs)  *  cMatF<Type>::MatAxiator(-VtoP3(aVecUk,aK0Uk+3))  :
                      cMatF<Type>(3,3,aVecObs,aK0Obs)
                 )
            )
        {
        }


        /// A pose being considered a the, isometric, mapinc X->Tr+R*X, return pose corresponding to inverse mapping
        cPoseF<Type> Inverse() const
        {
             //  PA-1 =  {-tRA CA ; tRA}
             cMatF<Type> aMatInv = mIJK.Transpose();
             return cPoseF<Type>(- (aMatInv* mCenter),aMatInv);
        }

        //   MatA = M0A * WA                   MatB = M0B * WB
        //   MatA-1 =  tWA * tM0A
        //   trA-1  =  - tWA * tM0A *CA
        //
        //    MatAB =  tWA * tM0A * M0B * WB
        //    CAB   =   - tWA * tM0A *CA +   tWA * tM0A * CB = tWA tM0A * (CB-CA)


        /// A pose being considered as mapping, return their composition
        cPoseF<Type> operator * (const cPoseF<Type> & aP2) const
        {
            const cPoseF<Type> & aP1 = *this;

             //   {CA;RA}* {CB;RB} = {CA+RA*CB ; RA*RB}
            return cPoseF<Type>
                   (
                        aP1.mCenter + aP1.mIJK*aP2.mCenter,
                        aP1.mIJK * aP2.mIJK
                   );
        }

        cPoseF<Type>  PoseRel(const cPoseF<Type> & aP2) const
        {
            //  PA~PB = PA-1 * PB
            return Inverse() * aP2;
        }

        cPtxd<Type,3>  mCenter;
        cMatF<Type>    mIJK;
};




};//  namespace MMVII

#endif // _COMMON_HEADER_SYMBDER_H_
