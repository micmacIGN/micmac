#include "include/MMVII_all.h"

namespace MMVII
{

/* ========================== */
/*          cSemA2007         */
/* ========================== */

cSemA2007::cSemA2007(eTA2007 aType,const std::string & anAux) :
   mType (aType),
   mAux  (anAux)
{
}

cSemA2007::cSemA2007(eTA2007 aType) :
   cSemA2007(aType,"")
{
}

eTA2007  cSemA2007::Type()            const {return mType;}
const std::string &  cSemA2007::Aux() const {return mAux;}



/* ========================== */
/*          cSpecOneArg2007   */
/* ========================== */

const std::vector<cSemA2007>   cSpecOneArg2007::TheEmptySem;

cSpecOneArg2007::cSpecOneArg2007(const std::string & aName,const std::string & aCom,const tVSem & aVSem) :
   mName (aName),
   mCom  (aCom),
   mVSem (aVSem),
   mNbMatch (0)
{
}

void cSpecOneArg2007::IncrNbMatch()
{
   mNbMatch++;
}

int  cSpecOneArg2007::NbMatch () const
{
   return mNbMatch;
}

const cSpecOneArg2007::tVSem & cSpecOneArg2007::VSem() const
{
   return mVSem;
}

bool cSpecOneArg2007::HasType(const eTA2007 & aType,std::string * aValue) const
{
    for (const auto & aSem : mVSem)
    {
       if (aSem.Type() == aType)
       {
          if (aValue) 
             *aValue =  aSem.Aux();
          return true;
       }
   }

    return false;
}

const std::string  & cSpecOneArg2007::Name() const
{
   return mName;
}

const std::string  & cSpecOneArg2007::Com() const
{
   return mCom;
}



/* ============================ */
/*          cCollecSpecArg2007  */
/* ============================ */


cCollecSpecArg2007 & cCollecSpecArg2007::operator << (tPtrArg2007 aVal)
{
    mV.push_back(aVal);
    return *this; 
}

cCollecSpecArg2007::cCollecSpecArg2007()
{
}

size_t cCollecSpecArg2007::size() const
{
   return mV.size();
}

tPtrArg2007 cCollecSpecArg2007::operator [] (int aK) const
{
   return mV.at(aK);
}

void cCollecSpecArg2007::clear()
{
   mV.clear();
}

tVecArg2007 & cCollecSpecArg2007::Vec()
{
   return mV;
}



/* ============================================== */
/*                                                */
/*       cInstReadOneArgCL2007                    */
/*                                                */
/* ============================================== */

template <class Type> class cInstReadOneArgCL2007 : public cSpecOneArg2007
{
    public :
        void InitParam(const std::string & aStr) override
        {
            mVal = cStrIO<Type>::FromStr(aStr);
        }
        cInstReadOneArgCL2007 (Type & aVal,const std::string & aName,const std::string & aCom,const tVSem & aVSem) :
              cSpecOneArg2007(aName,aCom,aVSem),
              mVal         (aVal)
        {
        }
        const std::string & NameType() const override 
        {
            return  cStrIO<Type>::msNameType;
        }
        void * AdrParam() override {return &mVal;}

    private :
        Type &          mVal;
};


template <class Type> tPtrArg2007 Arg2007(Type & aVal, const std::string & aCom,const cSpecOneArg2007::tVSem & aVSem )
{
   return tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,"",aCom,aVSem));
}




template <class Type> tPtrArg2007 AOpt2007(Type & aVal,const std::string & aName, const std::string &aCom,const cSpecOneArg2007::tVSem & aVSem)
{
   return  tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,aName,aCom,aVSem));
}

#define MACRO_INSTANTIATE_ARG2007(Type)\
template tPtrArg2007 Arg2007<Type>(Type &, const std::string & aCom,const cSpecOneArg2007::tVSem & aVSem);\
template tPtrArg2007 AOpt2007<Type>(Type &,const std::string & aName, const std::string & aCom,const cSpecOneArg2007::tVSem & aVSem);

MACRO_INSTANTIATE_ARG2007(int)
MACRO_INSTANTIATE_ARG2007(double)
MACRO_INSTANTIATE_ARG2007(bool)
MACRO_INSTANTIATE_ARG2007(std::string)


/*
template <> tPtrArg2007 AOpt2007<int>(int &,const std::string & aName, const std::string & aCom);

template <> tPtrArg2007 Arg2007<double>(double &, const std::string & aCom);
template <> tPtrArg2007 AOpt2007<double>(double &,const std::string & aName, const std::string & aCom);

template <> tPtrArg2007 Arg2007<std::string>(std::string &, const std::string & aCom);
template <> tPtrArg2007 AOpt2007<std::string>(std::string &,const std::string & aName, const std::string & aCom);

template <> tPtrArg2007 Arg2007<bool>(bool &, const std::string & aCom);
template <> tPtrArg2007 AOpt2007<bool>(bool &,const std::string & aName, const std::string & aCom);
*/




};
