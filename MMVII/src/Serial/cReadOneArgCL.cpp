#include "include/MMVII_all.h"

namespace MMVII
{


/*          cOneArgCL2007   */
cOneArgCL2007::cOneArgCL2007(const std::string & aName,const std::string & aCom) :
   mName (aName),
   mCom  (aCom)
{
}

/*          cCollecArg2007  */

tVecArg2007  & cCollecArg2007::V() {return mV;}

cCollecArg2007 & operator << (cCollecArg2007 & aV ,tPtrArg2007 aVal)
{
    aV.V().push_back(aVal);
    return aV; 
}



template <class Type> class cInstReadOneArgCL2007 : public cOneArgCL2007
{
    public :
        void InitParam(const std::string & aStr) override
        {
            mVal = cStrIO<Type>::FromStr(aStr);
        }
        cInstReadOneArgCL2007 (Type & aVal,const std::string & aName,const std::string & aCom) :
              cOneArgCL2007(aName,aCom),
              mVal         (aVal)
        {
        }
    private :
        Type &          mVal;
};


template <class Type> tPtrArg2007 Arg2007(Type & aVal, const std::string & aCom)
{
   return tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,"",aCom));
}




template <class Type> tPtrArg2007 AOpt2007(Type & aVal,const std::string & aName, const std::string &aCom)
{
   return  tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,aName,aCom));
}


template tPtrArg2007 Arg2007<int>(int &, const std::string & aCom);
template tPtrArg2007 AOpt2007<int>(int &,const std::string & aName, const std::string & aCom);
template tPtrArg2007 Arg2007<double>(double &, const std::string & aCom);
template tPtrArg2007 AOpt2007<double>(double &,const std::string & aName, const std::string & aCom);
template tPtrArg2007 Arg2007<std::string>(std::string &, const std::string & aCom);
template tPtrArg2007 AOpt2007<std::string>(std::string &,const std::string & aName, const std::string & aCom);

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
