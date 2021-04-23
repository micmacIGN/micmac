#include "include/MMVII_all.h"

namespace MMVII
{

/* ============================================= */
/*      cDataMapping<Type>                       */
/* ============================================= */


template <class Type> class cSimil2D : public cDataMapping<Type,2,2>
{
     public :
        typedef cDataMapping<Type,2,2> tSuper;
        typedef cPtxd<Type,2>          tPt;
        typedef std::vector<tPt>       tVPt;

        cSimil2D(const tPt &aTr,const tPt &aSc);
        inline tPt  Direct(const tPt & aPt) const override { return mTr + mSc * aPt;}
        const tVPt &  Direct(const std::vector<tPt> & aPt) const override ;
     private :
        tPt mTr;
        tPt mSc;
};

template <class Type>  
   cSimil2D<Type>::cSimil2D(const tPt &aTr,const tPt &aSc) :
      
      mTr (aTr),
      mSc (aSc)
{
}


template <class Type>  
    const std::vector<cPtxd<Type,2>> &  cSimil2D<Type>::Direct(const tVPt & aVPt) const
{
   typename tSuper::tVecOut&  aBuf = tSuper::BufOut();
   for (const auto & aPt : aVPt)
       aBuf.push_back(cSimil2D<Type>::Direct(aPt));
   return aBuf;
}


template class cSimil2D<double>;



};
