
#include "MMVII_Stringifier.h"

namespace MMVII
{

/* ========================== */
/*          cEnumAttr         */
/* ========================== */

template <class TypeEnum> cEnumAttr<TypeEnum>::cEnumAttr(TypeEnum aType,const std::string & anAux) :
   mType (aType),
   mAux  (anAux)
{
}
template <class TypeEnum> cEnumAttr<TypeEnum>::cEnumAttr(TypeEnum aType) :
   cEnumAttr<TypeEnum>(aType,"")
{
}
template <class TypeEnum> TypeEnum            cEnumAttr<TypeEnum>::Type() const {return mType;}
template <class TypeEnum> const std::string & cEnumAttr<TypeEnum>::Aux()  const {return mAux;}


/* ========================== */
/*    cES_PropertyList        */
/* ========================== */

template <class TypeEnum> cES_PropertyList<TypeEnum>::cES_PropertyList(const tAllPairs & aAllPairs) :
   mAllPairs (aAllPairs)
{
}

template <class TypeEnum> const typename  cES_PropertyList<TypeEnum>::tAllPairs & cES_PropertyList<TypeEnum>::AllPairs() const
{
   return mAllPairs;
}


template class cEnumAttr<eTA2007>;
template class cES_PropertyList<eTA2007>;


/* ========================== */
/*             ::             */
/* ========================== */


std::string  Name4Help(const tSemA2007 & aSem)
{
   if (int(aSem.Type()) < int(eTA2007::AddCom))
   {
      return E2Str(aSem.Type()) + aSem.Aux();
   }

   return "";
}



/* ========================== */
/*          cSpecOneArg2007   */
/* ========================== */

const std::vector<tSemA2007>   cSpecOneArg2007::TheEmptySem;

cSpecOneArg2007::cSpecOneArg2007(const std::string & aName,const std::string & aCom,const tAllSemPL & aVPL) :
   mName  (aName),
   mCom   (aCom),
   mSemPL (aVPL)
{
    ReInit();
}

cSpecOneArg2007::~cSpecOneArg2007()
{
}

void cSpecOneArg2007::ReInit()
{
   mNbMatch = 0;
}


std::string  cSpecOneArg2007::Name4Help() const
{
   std::string aRes;
   int aNb=0;
   for (const auto & aSem : SemPL())
   {
      std::string aStr = MMVII::Name4Help(aSem);
      if (aStr!="")
      {
         if (aNb==0)
            aRes = " [";
         else
            aRes = aRes + ",";
         aRes = aRes + aStr;
         aNb++;
      }
   }
   if (aNb!=0)
      aRes += "]";
   return aRes;
}

std::list<std::string>  cSpecOneArg2007::AddComs() const
{
   std::list<std::string> aRes;
   for (const auto & aSem : SemPL())
   {
      if (aSem.Type()== eTA2007::AddCom)
         aRes.push_back(aSem.Aux());
   }
   return aRes;
}


void cSpecOneArg2007::IncrNbMatch()
{
   mNbMatch++;
}

int  cSpecOneArg2007::NbMatch () const
{
   return mNbMatch;
}

const cSpecOneArg2007::tAllSemPL & cSpecOneArg2007::SemPL() const
{
   return mSemPL.AllPairs();
}

bool cSpecOneArg2007::HasType(const eTA2007 & aType,std::string * aValue) const
{
    for (const auto & aSem : SemPL())
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

const std::string  & cSpecOneArg2007::Value() const
{
   return mValue;
}

const std::string  & cSpecOneArg2007::Com() const
{
   return mCom;
}

void  cSpecOneArg2007::InitParam(const std::string & aStr) 
{
   mValue = aStr;
   V_InitParam(aStr);
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

template <class Type> void  GlobCheckSize(const Type & ,const std::string & anArg) 
{
    MMVII_INTERNAL_ASSERT_always(false,"Check size vect for non vect arg");
}

template <class Type> void  GlobCheckSize(const std::vector<Type> & aVal,const std::string & anArg) 
{
    cPt2di aSz = cStrIO<cPt2di>::FromStr(anArg);
    if ((int(aVal.size()) < aSz.x()) || ((int(aVal.size()) > aSz.y()))) 
    {
       MMVII_UsersErrror(eTyUEr::eBadSize4Vect,"IntervalOk=" + anArg + " Got=" + ToStr(int(aVal.size())));
    }
}


template <class Type> class cInstReadOneArgCL2007 : public cSpecOneArg2007
{
    public :

       void  CheckSize(const std::string & anArg) const override 
       {
               GlobCheckSize(mVal,anArg);
       }


        void V_InitParam(const std::string & aStr) override
        {
            mVal = cStrIO<Type>::FromStr(aStr);
        }
        cInstReadOneArgCL2007 (Type & aVal,const std::string & aName,const std::string & aCom,const tAllSemPL & aVSem) :
              cSpecOneArg2007(aName,aCom,aVSem),
              mVal         (aVal)
        {
        }
        const std::string & NameType() const override 
        {
            return  cStrIO<Type>::msNameType;
        }
        void * AdrParam() override {return &mVal;}
        std::string NameValue() const override {return ToStr(mVal);}

    private :
        Type &          mVal;
};


template <class Type> tPtrArg2007 Arg2007(Type & aVal, const std::string & aCom,const cSpecOneArg2007::tAllSemPL & aVSem )
{
   return tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,"",aCom,aVSem));
}




template <class Type> tPtrArg2007 AOpt2007(Type & aVal,const std::string & aName, const std::string &aCom,const cSpecOneArg2007::tAllSemPL & aVSem)
{
   return  tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,aName,aCom,aVSem));
}

#define MACRO_INSTANTIATE_ARG2007(Type)\
template tPtrArg2007 Arg2007<Type>(Type &, const std::string & aCom,const cSpecOneArg2007::tAllSemPL & aVSem);\
template tPtrArg2007 AOpt2007<Type>(Type &,const std::string & aName, const std::string & aCom,const cSpecOneArg2007::tAllSemPL & aVSem);

MACRO_INSTANTIATE_ARG2007(int)
MACRO_INSTANTIATE_ARG2007(double)
MACRO_INSTANTIATE_ARG2007(bool)
MACRO_INSTANTIATE_ARG2007(std::string)
MACRO_INSTANTIATE_ARG2007(std::vector<std::string>)
MACRO_INSTANTIATE_ARG2007(std::vector<int>)
MACRO_INSTANTIATE_ARG2007(std::vector<double>)
MACRO_INSTANTIATE_ARG2007(cPt2di)
MACRO_INSTANTIATE_ARG2007(cPt2dr)
MACRO_INSTANTIATE_ARG2007(cPt3di)
MACRO_INSTANTIATE_ARG2007(cPt3dr)
MACRO_INSTANTIATE_ARG2007(eProjPC)
MACRO_INSTANTIATE_ARG2007(eOpAff)
MACRO_INSTANTIATE_ARG2007(eModeEpipMatch)
MACRO_INSTANTIATE_ARG2007(eModeTestPropCov)
MACRO_INSTANTIATE_ARG2007(eModePaddingEpip)
MACRO_INSTANTIATE_ARG2007(eModeCaracMatch)
MACRO_INSTANTIATE_ARG2007(eDCTFilters)

MACRO_INSTANTIATE_ARG2007(cBox2di)
MACRO_INSTANTIATE_ARG2007(cBox2dr)
MACRO_INSTANTIATE_ARG2007(cBox3di)
MACRO_INSTANTIATE_ARG2007(cBox3dr)

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
