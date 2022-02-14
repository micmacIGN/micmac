#ifndef  _MMVII_SET_I_TPL_H_
#define  _MMVII_SET_I_TPL_H_

namespace MMVII
{

/** \file MMVII_SetITpl.h
    \brief classses fot sets of integers


*/


template  <typename tSet> size_t SetI_CardByNbMax(const tSet & aSet);
template  <typename tSet> std::vector<int> SetI_ToVectNbMax(const tSet & aSet);
template  <typename tSet> tSet  SetI_Interv(int aK1, int aK2);
template  <typename tSet> std::vector<tSet>  SubKAmongN(int aK, int aN);

template <typename tVal> class cSetISingleFixed;

/*  ********************************** */
/*                                     */
/*            cSetISingleFixed         */
/*                                     */
/*  ********************************** */

template <typename tVal> class cSetISingleFixed
{
    public :
         typedef  cSetISingleFixed<tVal>  tSet;
         typedef  tU_INT4                 tElem;

         static size_t  NbMax() {return NbBits;}


         static tSet EmptySet() {return tSet(0);}
         static tSet FullSet()  {return tSet(tElemNumTrait<tVal>::MaxVal());}
         static tSet Singleton(const tElem & anElem) {return tSet(FlagSingle(anElem));}


         bool  IsInside(const tElem & anElem) const {return (mFlagBits & FlagSingle(anElem)) != 0;}
         size_t  Cardinality() const {return SetI_CardByNbMax(*this);}
         std::vector<int> ToVect() const {return SetI_ToVectNbMax(*this);}

         void AddElem(const tElem & anElem) {mFlagBits |= FlagSingle(anElem);}

    private :
         static constexpr  size_t NbBits = 8 *sizeof(tVal);
         cSetISingleFixed(const tVal & aFlagBit) : mFlagBits (aFlagBit) {}
         static tVal  FlagSingle(const tElem & anElem) {return (1<<anElem);}

         tVal    mFlagBits;
};


// Compute Cardinality, Usable when NbMax (number max of elem in type) is defined
template  <typename tSet> size_t SetI_CardByNbMax(const tSet & aSet)
{
    size_t aRes =0;
    for (size_t aK=0 ; aK<tSet::NbMax() ; aK++)
        if (aSet.IsInside(aK))
           aRes++;

    return aRes;
}
// To Vector as SetI_CardByNbMax
template  <typename tSet> std::vector<int> SetI_ToVectNbMax(const tSet & aSet)
{
    std::vector<int> aRes ;
    for (size_t aK=0 ; aK<tSet::NbMax() ; aK++)
        if (aSet.IsInside(aK))
           aRes.push_back(aK);

    return aRes;
}

//  Compute a set in the given interval  
template  <typename tSet> tSet  SetI_Interv(int aK1, int aK2)
{
    tSet aResult = tSet::EmptySet();

    for (int aK =aK1 ; aK<aK2 ; aK++)
    {
        aResult.AddElem(aK);
    }
    return aResult;
}

// Subset of K element among N, using "Pascal Triangle"  C(K,N) = C(K-1,N-1) + C(K,N-1)
template  <typename tSet> std::vector<tSet>  SubKAmongN(int aK, int aN)
{
   std::vector<tSet> aRes;
   // Two terminal case
   if (aK==0)
   {
       aRes.push_back(tSet::EmptySet());
       return aRes;
   }
   if (aK==aN)
   {
       aRes.push_back(SetI_Interv<tSet>(0,aN));
       return aRes;
   }

   // Recursive formula
   aRes = SubKAmongN<tSet>(aK,aN-1);  // C(K,N-1)
   for (auto & aSub : SubKAmongN<tSet>(aK-1,aN-1))
   {
       aSub.AddElem(aN-1);
       aRes.push_back(aSub);
   }

   return aRes;
}

template  <typename tSet> void  ShowSetByCardMax(std::ostream & OS,const tSet & aSet)
{
    OS << "{";
    bool First = true;
    for (size_t aK=0 ; aK<tSet::NbMax() ; aK++)
    {
        if (aSet.IsInside(aK))
        {
            if (!First)
               OS << " ";
            OS << aK;
            First = false;
        }
    }
    OS << "}";
}

template <typename tVal> std::ostream &  operator << ( std::ostream & OS,const cSetISingleFixed<tVal> & aSet)
{
    ShowSetByCardMax(OS,aSet);
    return OS;
}


};

#endif  //  _MMVII_SET_I_TPL_H_
