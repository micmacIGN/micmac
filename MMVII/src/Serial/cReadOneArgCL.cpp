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



/* ======================== */
/*          cOneArgCL2007   */
/* ======================== */

const std::vector<cSemA2007>   cOneArgCL2007::TheEmptySem;

cOneArgCL2007::cOneArgCL2007(const std::string & aName,const std::string & aCom,const tVSem & aVSem) :
   mName (aName),
   mCom  (aCom),
   mVSem (aVSem),
   mNbMatch (0)
{
}

void cOneArgCL2007::IncrNbMatch()
{
   mNbMatch++;
}

int  cOneArgCL2007::NbMatch () const
{
   return mNbMatch;
}

const cOneArgCL2007::tVSem & cOneArgCL2007::VSem() const
{
   return mVSem;
}

bool cOneArgCL2007::HasType(const eTA2007 & aType) const
{
    for (auto aSem : mVSem)
       if (aSem.Type() == aType)
          return true;

    return false;
}

const std::string  & cOneArgCL2007::Name() const
{
   return mName;
}

const std::string  & cOneArgCL2007::Com() const
{
   return mCom;
}



/* ======================== */
/*          cCollecArg2007  */
/* ======================== */


cCollecArg2007 & cCollecArg2007::operator << (tPtrArg2007 aVal)
{
    mV.push_back(aVal);
    return *this; 
}

cCollecArg2007::cCollecArg2007()
{
}

size_t cCollecArg2007::size() const
{
   return mV.size();
}

tPtrArg2007 cCollecArg2007::operator [] (int aK) const
{
   return mV.at(aK);
}

void cCollecArg2007::clear()
{
   mV.clear();
}

tVecArg2007 & cCollecArg2007::Vec()
{
   return mV;
}



/* ============================================== */
/*                                                */
/*       cInstReadOneArgCL2007                    */
/*                                                */
/* ============================================== */

template <class Type> class cInstReadOneArgCL2007 : public cOneArgCL2007
{
    public :
        void InitParam(const std::string & aStr) override
        {
            mVal = cStrIO<Type>::FromStr(aStr);
        }
        cInstReadOneArgCL2007 (Type & aVal,const std::string & aName,const std::string & aCom,const tVSem & aVSem) :
              cOneArgCL2007(aName,aCom,aVSem),
              mVal         (aVal)
        {
        }
        const std::string & NameType() const override 
        {
            return  cStrIO<Type>::msNameType;
        }

    private :
        Type &          mVal;
};


template <class Type> tPtrArg2007 Arg2007(Type & aVal, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem )
{
   return tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,"",aCom,aVSem));
}




template <class Type> tPtrArg2007 AOpt2007(Type & aVal,const std::string & aName, const std::string &aCom,const cOneArgCL2007::tVSem & aVSem)
{
   return  tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,aName,aCom,aVSem));
}


template tPtrArg2007 Arg2007<int>(int &, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 AOpt2007<int>(int &,const std::string & aName, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 Arg2007<double>(double &, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 AOpt2007<double>(double &,const std::string & aName, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 Arg2007<std::string>(std::string &, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 AOpt2007<std::string>(std::string &,const std::string & aName, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 Arg2007<bool>(bool &, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
template tPtrArg2007 AOpt2007<bool>(bool &,const std::string & aName, const std::string & aCom,const cOneArgCL2007::tVSem & aVSem);
/*
*/

/*
*/

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
