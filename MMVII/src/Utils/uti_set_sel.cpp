#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include <regex>
#include <unordered_set>
#include <unordered_map>


/** \file uti_set_sel.cpp
    \brief Implementation of selector & sets

   This file contains implementation of a set of class that allow to
  construct selectors, elemenatary one, or by boolean construction

*/


/* ======================================== */
/*                                          */
/*     cOrderedPair<Type>                   */
/*                                          */
/* ======================================== */

/// Cle de hashage pour pouvoir faire des unordered set
namespace std
{
template<>
class hash<MMVII::tNamePair> {
public:
    size_t operator()(const MMVII::tNamePair &s) const 
    {
        return std::hash<std::string>()(s.V1()) ^ std::hash<std::string>()(s.V2());
    }
};
}



namespace MMVII
{

static bool TheCmp4Op=false;

template <class Type> const Type & cOrderedPair<Type>::Min4OP(const Type & aV1,const Type & aV2)
{
    TheCmp4Op = (aV1<aV2);
    return TheCmp4Op ? aV1 : aV2; 
}

template <class Type>  cOrderedPair<Type>::cOrderedPair(const Type & aV1,const Type & aV2) :
   mV1 (Min4OP(aV1,aV2)),
   mV2 (TheCmp4Op?aV2:aV1)
{
}

template <class Type>  cOrderedPair<Type>::cOrderedPair() 
{
}

template <class Type>  bool cOrderedPair<Type>::operator < (const cOrderedPair<Type> & aP2) const
{
   if (mV1 < aP2.mV1) return true;
   if (mV1 > aP2.mV1) return false;
   if (mV2 < aP2.mV2) return true;
   return false;
}
template <class Type> bool cOrderedPair<Type>::operator == (const cOrderedPair<Type> & aP2) const
{
  return (mV1 == aP2.mV1) && (mV2 == aP2.mV2);
}

template <class Type> const Type & cOrderedPair<Type>::V1() const {return  mV1;}
template <class Type> const Type & cOrderedPair<Type>::V2() const {return  mV2;}
template <class Type> Type & cOrderedPair<Type>::V1() {return  mV1;}
template <class Type> Type & cOrderedPair<Type>::V2() {return  mV2;}

template  class cOrderedPair<std::string>;

/* =============   cDataSelector<Type>   ========= */

// Base class for Selector
/**  a selector is just a boolean predicate,
     the heriting class will implement override Match
     Class is accessible via cSelector

     As the implentation is "hidden" in this file, I dont see the benefit of 
    setting private the methods.
*/

template <class Type> class cDataSelector : public cMemCheck
{

    public :
    // friend class cSelector<Type>;
    // private :
        virtual bool Match(const Type &) const = 0;  ///< fundamuntal function
        virtual ~cDataSelector() ;
};


template <class Type> cDataSelector<Type>::~cDataSelector()
{
}


/* =============  cSelector<Type>  =============== */

template <class Type> cSelector<Type>::cSelector(cDataSelector<Type> * aPtr) :
  mDS(aPtr)
{
}

template <class Type> bool  cSelector<Type>::Match(const Type & aV) const
{
   return mDS->Match(aV);
}


/* ================================================= */


//====================== Interval ==========================

/// Implemantation of Intervall type
/**   Interval [V1,V2] , ]V1,V2] etc ...
      optional with no value mean "infinite" bound
*/

template <class Type> class cDataIntervSelector : public  cDataSelector<Type>
{   
    public :
       typedef std::optional<Type> tOObj;
       typedef const  tOObj *        tCOOPtr;

       cDataIntervSelector(tCOOPtr aV1,tCOOPtr aV2,bool aInclLow,bool InclUp) :
          mV1      (*Reorder(aV1,aV2,aInclLow,InclUp)),
          mV2      (*aV2),
          mInclLow (aInclLow),
          mInclUp  (InclUp)
       {
       }

       bool Match(const Type & aV) const override
       {
           if (mV1) // no value no use
           {
              if (mInclLow)  
              {
                 // if Incl and equal => ok
                 if (aV< mV1) return false;
              }
              else
              {
                 // if !Incl and equal => exclud
                 if (aV<= mV1) return false;
              }
           }

           if (mV2) // no value no use
           {
              if (mInclUp)  
              {
                 // if Incl and equal => ok
                 if (aV> mV2) return false;
              }
              else
              {
                 // if !Incl and equal => exclud
                 if (aV>= mV2) return false;
              }
           }

           return true;
       }
    private :
       static tCOOPtr   Reorder(tCOOPtr & aV1,tCOOPtr & aV2,bool & InclLow,bool & InclUp)
       {
           if (*aV1&& *aV2 && (*aV1>*aV2))
           {
              std::swap(aV1,aV2);
              std::swap(InclLow,InclUp);
           }
           return aV1;
       }


       tOObj             mV1;        ///< lower bound , optionnal no value-no use
       tOObj             mV2;        ///< upper bound
       bool              mInclLow;   ///< Is it open lower bound (Open <=> bound excluded)
       bool              mInclUp;    ///< Is it open upper bound 
};

template<class Type> cSelector<Type> GenIntervalSelector( const std::optional<Type> & aV1, 
                                                         const std::optional<Type> & aV2,
                                                         bool aInclLow,bool InclUp)
{
   return cSelector<Type>(new cDataIntervSelector<Type>(&aV1,&aV2,aInclLow,InclUp));
}

///  ]<- ,V] or ]<-,V[
template<class Type> cSelector<Type> LeftHalfIntervalSelector(const Type & aV, bool aIncl)
{
   std::optional<Type> aOptV(aV);
   std::optional<Type> aNone = std::nullopt;
   return cSelector<Type>(new cDataIntervSelector<Type>(&aNone,&aOptV,aIncl,aIncl));
}

///  ]V,->[ or [V,->[
template<class Type> cSelector<Type> RightHalfIntervalSelector(const Type & aV, bool aIncl)
{
   std::optional<Type> aOptV(aV);
   std::optional<Type> aNone = std::nullopt;
   return cSelector<Type>(new cDataIntervSelector<Type>(&aOptV,&aNone,aIncl,aIncl));
}

template<class Type> cSelector<Type> IntervalSelector(const Type&aV1,const Type&aV2,bool aInclLow,bool InclUp)
{
   std::optional<Type> aOptV1(aV1);
   std::optional<Type> aOptV2(aV2);
   return cSelector<Type>(new cDataIntervSelector<Type>(&aOptV1,&aOptV2,aInclLow,InclUp));
}



/*
*/


    // =====================  Negation =============

/// Implemantation of Negation
template <class Type> class cDataNegSelector : public  cDataSelector<Type>
{
    public :
         typedef cSelector<Type>  tSel;
         typedef const  tSel &    tCRefS;
         cDataNegSelector(tCRefS  aSel) :
            mSel (aSel)
         {
         }
         bool Match(const Type & aV) const override
         {
            return ! mSel.Match(aV);
         }
     private :
         tSel mSel;  ///< Selector negated
};
template<class Type> cSelector<Type> operator !  (const cSelector<Type> & aSel)
{
   return cSelector<Type>(new cDataNegSelector<Type>(aSel));
}

    // =====================  And =============

/// Implemantation of And on selectors
template <class Type> class cDataAndSelector : public  cDataSelector<Type>
{
    public :
         typedef cSelector<Type>  tSel;
         typedef const  tSel &    tCRefS;
         cDataAndSelector(tCRefS aSel1,tCRefS  aSel2) :
            mSel1 (aSel1),
            mSel2 (aSel2)
         {
         }
         bool Match(const Type & aV) const override
         {
            return  mSel1.Match(aV) && mSel2.Match(aV);
         }
     private :
         tSel mSel1;
         tSel mSel2;
};
template<class Type> cSelector<Type> operator &&  (const cSelector<Type> & aSel1,const cSelector<Type> & aSel2)
{
   return cSelector<Type>(new cDataAndSelector<Type>(aSel1,aSel2));
}

    // =====================  Or  =============

/// Implemantation of Or on selectors
template <class Type> class cDataOrSelector : public  cDataSelector<Type>
{
    public :
         typedef cSelector<Type>  tSel;
         typedef const  tSel &    tCRefS;
         cDataOrSelector(tCRefS aSel1,tCRefS  aSel2) :
            mSel1 (aSel1),
            mSel2 (aSel2)
         {
         }
         bool Match(const Type & aV) const override
         {
            return  mSel1.Match(aV) || mSel2.Match(aV);
         }
     private :
         tSel mSel1;
         tSel mSel2;
};
template<class Type> cSelector<Type> operator ||  (const cSelector<Type> & aSel1,const cSelector<Type> & aSel2)
{
   return cSelector<Type>(new cDataOrSelector<Type>(aSel1,aSel2));
}

    // =====================  Cste  =============

/// Implemantation of Cste selectors
template <class Type> class cDataCsteSelector : public  cDataSelector<Type>
{
    public :
         cDataCsteSelector(bool aVal) : mVal (aVal) {}
         bool Match(const Type & aV) const override { return  mVal; }
     private :
         bool mVal;
};
template<class Type> cSelector<Type> CsteSelector  (bool aVal)
{
   return cSelector<Type>(new cDataCsteSelector<Type>(aVal));
}



    // ======================= String to intervals ====================

/// Is next caracter an interval delimitor? If yes is iit inclusive (set in 'aIncl')
bool GetInclus(bool & aIncl,int aChar,bool begin)
{
   if (aChar==']') 
   {
      aIncl = ! begin;
      return true;
   }
   if (aChar=='[')  
   {
      aIncl = begin;
      return true;
   }

   return false;
}

typedef std::pair<std::string,bool> tElemStrInterv;

///   Prepare processing string 2 interval 

/** Make the low level job, pair of string an bool indicating inclusion
    "]ab,cd][,uu]" => {{ab,false},{cd,true}{,true}{uu,true}}
*/
std::vector<tElemStrInterv> LireInterv(const std::string & aStr)
{
   std::vector<tElemStrInterv>  aRes;
   const char * aC = aStr.c_str();
   while (*aC)
   {
      SkeepWhite(aC);
      // Get [ or ]
      bool aLowIncl,anUpIncl;
      if (!GetInclus(aLowIncl,*aC,true))
      {
         MMVII_INTERNAL_ASSERT_always(false,"Expected ] ou [ in read interv");
      }
      aC++;
      const char * aC0 = aC;  //memorize position a string begining
      // Skip to next  [ or ]
      while (! GetInclus(anUpIncl,*aC,false))
      {
         MMVII_INTERNAL_ASSERT_always(*aC,"No closing ] ou [ in read interv");
         aC++;
      }

      // Split string arround ','
      std::string a2Str(aC0,aC);
      std::string aStrLow,aStrUp;
      SplitStringArround(aStrLow,aStrUp,a2Str,',');

      aRes.push_back(tElemStrInterv(aStrLow,aLowIncl));
      aRes.push_back(tElemStrInterv(aStrUp,anUpIncl));

      aC++;
      SkeepWhite(aC);
   }

/*
   if (0)
   {
      for (size_t aK=0  ; aK<aRes.size() ; aK+=2)
      {
         std:: cout << (aRes[aK].second ? "[" : "]");
         std:: cout << aRes[aK].first << "," << aRes[aK+1].first ;
         std:: cout << (aRes[aK+1].second ? "]" : "[");
         std:: cout << "\n";
      }
   }
*/
   return aRes;
}

/// Compute union of intervals from a string
template <class Type> cSelector<Type> Str2Interv(const std::string & aStr)
{
    std::vector<tElemStrInterv>  aVI = LireInterv(aStr);
    cSelector<Type> aRes =  CsteSelector<Type>(false);


    for (size_t aK=0  ; aK<aVI.size() ; aK+=2)
    {
         std::optional<Type> aV1 = std::nullopt;
         std::optional<Type> aV2 = std::nullopt;
         if (aVI[aK].first!="")
            aV1 = cStrIO<Type>::FromStr(aVI[aK].first);
         if (aVI[aK+1].first!="")
            aV2 = cStrIO<Type>::FromStr(aVI[aK+1].first);
         aRes = aRes || GenIntervalSelector(aV1,aV2,aVI[aK].second,aVI[aK+1].second);
    }
     

    return aRes;
}


#define MACRO_INSTANTIATE_SELECTOR(Type)\
template cSelector<Type> GenIntervalSelector(const std::optional<Type> & aV1, const std::optional<Type> & aV2, bool aInclLow,bool InclUp);\
template cSelector<Type> LeftHalfIntervalSelector(const Type & aV, bool aIncl);\
template cSelector<Type> RightHalfIntervalSelector(const Type & aV, bool aIncl);\
template cSelector<Type> IntervalSelector(const Type&aV1,const Type&aV2,bool aInclLow,bool InclUp);\
template cSelector<Type> operator ! (const cSelector<Type> & aSel); \
template cSelector<Type> operator &&  (const cSelector<Type> & aSel1,const cSelector<Type> & aSel2);\
template cSelector<Type> operator ||  (const cSelector<Type> & aSel1,const cSelector<Type> & aSel2);\
template cSelector<Type> CsteSelector  (bool aVal);\
template cSelector<Type> Str2Interv(const std::string & aStr);\
template class cSelector<Type>;\
template class cDataSelector<Type>;\
template class cDataIntervSelector<Type>;\
template class cDataNegSelector<Type>;\
template class cDataAndSelector<Type>;\
template class cDataOrSelector<Type>;\
template class cDataCsteSelector<Type>;\


MACRO_INSTANTIATE_SELECTOR(int)
MACRO_INSTANTIATE_SELECTOR(std::string)

template class cSelector<void *>; // required by cExtSet<Type>::Filter

/* ======================================== */
/*                                          */
/*     cStdtRegex                           */
/*                                          */
/* ======================================== */

/// std implementation of Regex expression
class cDataStdRegex : public  cDataSelector<std::string>
{
    public :
        cDataStdRegex(const std::string & aName):
            mRegex (aName)
        {
        }
        bool Match(const std::string &  aStr) const override 
        {
             return regex_match(aStr,mRegex);
        }
    private :
        std::regex mRegex;
};

tNameSelector  AllocRegex(const std::string& aPat)
{
   if (aPat=="")
      return  CsteSelector<std::string>(true);
   return cSelector<std::string>(new cDataStdRegex(aPat));
}




void BenchSelector(cParamExeBench & aParam,const std::string & aDir)
{
    if (! aParam.NewBench("Selector")) return;

    int aLow = 5;
    int aUp = 9;
    cSelector<int> aSII_59 = GenIntervalSelector<int>(aLow,aUp,true,true);
    cSelector<int> aSEE_59 = GenIntervalSelector<int>(aLow,aUp,false,false);

    // For this one bound are unordered 5 include, 9 exclude but swap
    cSelector<int> aSIE_59    = IntervalSelector<int>(aUp,aLow,false,true);
    cSelector<int> aSIEBis_59 = GenIntervalSelector<int>(aLow,aUp,true,false);
    cSelector<int> aSEI_59    = IntervalSelector<int>(aLow,aUp,false,true);

    cSelector<int> aSE_X9 = GenIntervalSelector<int>(std::nullopt,aUp ,false,false);
    cSelector<int> aSEBis_X9 = LeftHalfIntervalSelector<int>(aUp ,false);

    cSelector<int> aSI_5X = GenIntervalSelector<int>(aLow,std::nullopt,true,true);
    cSelector<int> aSIBis_5X = RightHalfIntervalSelector<int>(aLow,true);


    cSelector<int> aSII_711 = IntervalSelector<int>(7,11,true,true);

    cSelector<int> aSelAnd = aSII_59 && aSII_711;
    cSelector<int> aSelOr  = aSII_59 || aSII_711;
    cSelector<int> aSelNot = ! aSII_59 ;

    cSelector<int> aSel0 = CsteSelector<int>(false);
    cSelector<int> aSel1 = CsteSelector<int>(true);


    cSelector<int> aS1 = Str2Interv<int>("],5]");
    cSelector<int> aS2 = Str2Interv<int>("[8,9[");
    cSelector<int> aS3 = Str2Interv<int>("]12,15]");
    cSelector<int> aS4 = Str2Interv<int>("]17,]");

    cSelector<int> aS5 = aS1 || aS2 || aS3 || aS4;
    cSelector<int> aS6 = Str2Interv<int>("],5] [8,9[ ]12,15] ]17,]");
   
    
    cExtSet<int> aSet5; // Test equivalence Set/Sel
    cExtSet<int> aSet6; // Test filter
    for (int aK=0 ; aK<20 ; aK++)
    {
      if (aS5.Match(aK))  
         aSet5.Add(aK);
       aSet6.Add(aK);
    }
    aSet6.Filter(aS6);

    for (int aK=0 ; aK<20 ; aK++)
    {
        MMVII_INTERNAL_ASSERT_bench(aSII_59.Match(aK)==((aK>=aLow)&&(aK<=aUp)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSEE_59.Match(aK)==((aK>aLow)&&(aK<aUp)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSIE_59.Match(aK)==((aK>=aLow)&&(aK<aUp)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSIEBis_59.Match(aK)==aSIE_59.Match(aK),"Selector");

        MMVII_INTERNAL_ASSERT_bench(aSEI_59.Match(aK)==((aK>aLow)&&(aK<=aUp)),"Selector");


        MMVII_INTERNAL_ASSERT_bench(aSE_X9.Match(aK)==(aK<aUp ),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSE_X9.Match(aK)==aSEBis_X9.Match(aK),"Selector");

        MMVII_INTERNAL_ASSERT_bench(aSI_5X.Match(aK)==(aK>=aLow),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSI_5X.Match(aK)==aSIBis_5X.Match(aK),"Selector");


        MMVII_INTERNAL_ASSERT_bench(aSelAnd.Match(aK)==(aSII_59.Match(aK)&&aSII_711.Match(aK)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSelOr.Match(aK)==(aSII_59.Match(aK)||aSII_711.Match(aK)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSelNot.Match(aK)!=aSII_59.Match(aK),"Selector");
        MMVII_INTERNAL_ASSERT_bench(!aSel0.Match(aK),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aSel1.Match(aK),"Selector");

        MMVII_INTERNAL_ASSERT_bench(aS1.Match(aK)== (aK<=5),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aS2.Match(aK)== ((aK>=8)&&(aK<9)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aS3.Match(aK)== ((aK>12)&&(aK<=15)),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aS4.Match(aK)== (aK>17),"Selector");
	
        MMVII_INTERNAL_ASSERT_bench(aS5.Match(aK)==aS6.Match(aK),"Selector");

        MMVII_INTERNAL_ASSERT_bench(aS5.Match(aK)==aSet5.Match(aK),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aS5.Match(aK)==aSet5.In(aK),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aS6.Match(aK)==aSet6.Match(aK),"Selector");
        MMVII_INTERNAL_ASSERT_bench(aS6.Match(aK)==aSet6.In(aK),"Selector");
    }
    aParam.EndBench();
}





/* ============================================ */
/*                                              */
/*               cDataExtSet<Type>              */
/*                                              */
/* ============================================ */

template <class Type> class cDataExtSet : public cDataSelector<Type>
{
    public :
         virtual ~cDataExtSet()  {}
         virtual cDataExtSet<Type> *  VDupl() const = 0; // return a duplicata
         virtual cDataExtSet<Type> *  VEmptySet() const = 0; // return an empty set of same type

         virtual bool Add(const Type &) = 0; 
         virtual bool In(const Type &) const = 0;
         virtual bool Suppress(const Type &)  = 0;
         virtual void    clear() = 0;
         virtual int    size() const = 0;

         virtual void  PutInVect(std::vector<const Type *> &,bool Sorted) const = 0; ///< Some type requires iteration 

         static  cDataExtSet<Type> * AllocFromType(eTySC);

         bool Match(const Type & aVal) const override
         {
             return In(aVal);
         }
};

/* ============================================ */
/*                                              */
/*               cExtSet<Type>                  */
/*                                              */
/* ============================================ */

#define ASSERT_SET_ISINIT()\
MMVII_INTERNAL_ASSERT_tiny(mDES!=0,"Operation on null set")


       // Class method, just call pointed object method

template <class Type> cExtSet<Type>::cExtSet(cDataExtSet<Type> * aRawPtr) :
   cSelector<Type>(0),
   mDES(aRawPtr)
{
}

template <class Type> cExtSet<Type>::cExtSet(eTySC aKindOf) :
    cExtSet<Type> (cDataExtSet<Type>::AllocFromType(aKindOf))
{
}

template <class Type> cExtSet<Type>::~cExtSet() 
{
}

template <class Type> bool cExtSet<Type>::IsInit() const
{
   return mDES!=0;
}


template <class Type> bool cExtSet<Type>::Match(const Type &aVal) const 
{
   ASSERT_SET_ISINIT();
   return mDES->In(aVal);
}


template <class Type> cExtSet<Type>   cExtSet<Type>::Dupl() const 
{
   ASSERT_SET_ISINIT();
   return cExtSet<Type>(mDES->VDupl());
}
template <class Type>   cExtSet<Type>   cExtSet<Type>::EmptySet() const 
{
   ASSERT_SET_ISINIT();
   return cExtSet<Type>(mDES->VEmptySet());
}
template <class Type> bool cExtSet<Type>::Add(const Type & aVal) 
{
   ASSERT_SET_ISINIT();
   return mDES->Add(aVal);
}
template <class Type> bool cExtSet<Type>::In(const Type & aVal)  const
{
   ASSERT_SET_ISINIT();
   return mDES->In(aVal);
}
template <class Type> bool cExtSet<Type>::Suppress(const Type & aVal) 
{
   ASSERT_SET_ISINIT();
   return mDES->Suppress(aVal);
}
template <class Type> void cExtSet<Type>::clear()
{
   ASSERT_SET_ISINIT();
   return mDES->clear();
}
template <class Type> int cExtSet<Type>::size() const
{
   ASSERT_SET_ISINIT();
   return mDES->size();
}
template <class Type>  void  cExtSet<Type>::PutInVect(std::vector<const Type *> & aV,bool Sorted) const 
{
   ASSERT_SET_ISINIT();
   mDES->PutInVect(aV,Sorted);
}

template <class Type> bool cExtSet<Type>::IncludedIn(const  cExtSet<Type> & aSet) const
{
    std::vector<const Type *> aV;
    PutInVect(aV,false);
    for (const auto & el : aV)
       if (! aSet.In(*el))
          return false;
    return true;
}

template <class Type> bool cExtSet<Type>::Equal(const  cExtSet<Type> & aSet) const
{
   return  IncludedIn(aSet) && aSet.IncludedIn(*this);
}

template <class Type>  void  cExtSet<Type>::Filter(const cSelector<Type> & aSel)
{
    std::vector<const Type *> aV;
    PutInVect(aV,false);
    for (const auto & ptr : aV)
    {
        if (! aSel.Match(*ptr))
           Suppress(*ptr);
    }
}

template <class Type>  void cExtSet<Type>::OpAff(const eOpAff & anOp,const cExtSet<Type> aS)
{
   if      (anOp==eOpAff::ePlusEq)  (*this) += aS;
   else if (anOp==eOpAff::eMulEq)   (*this) *= aS;
   else if (anOp==eOpAff::eMinusEq) (*this) -= aS;
   else if (anOp==eOpAff::eEq)      (*this)  = aS;
   else if (anOp==eOpAff::eReset)   this->clear();
   else
   {
       MMVII_INTERNAL_ASSERT_always(false,"Bad value in cExtSet<Type>::OpAff")
   }
}

       // ========================================================
       //   Global method : operator  + += - -= * *=
       // ========================================================

 //=========================== Intersection ================

   /** Intersection, in - situ
      Algo :  Parse aRes and suppress  those who are not in aFilter 
   */

template <class Type>  void  operator *= (cExtSet<Type> & aRes,const cExtSet<Type> & aFilter)
{
   std::vector<const Type *>  aV;
   aRes.PutInVect(aV,false);
   for (const auto & el : aV)
   {
       if ( ! aFilter.In(*el))
          aRes.Suppress(*el);
   }
}
   /**  Intersection, fonctionnal
        algo :  Parse S1 and add  in aRes those who are in S2
   */
template <class Type>  cExtSet<Type>  operator * (const cExtSet<Type> & aS1,const cExtSet<Type> & aS2)
{
   cExtSet<Type>  aRes = aS1.EmptySet();

   std::vector<const Type *>  aV1;
   aS1.PutInVect(aV1,false);

   for (const auto & el : aV1)
        if (aS2.In(*el))
           aRes.Add(*el);
   return  aRes;
}


 //=========================== Union  ================

   /** Union, in - situ
       Algo : parse toAdd and add in aRes
   */
template <class Type>  void   operator += (cExtSet<Type> & aRes,const cExtSet<Type> & toAdd)
{
   std::vector<const Type *>  aV;
   toAdd.PutInVect(aV,false);
   for (const auto & el : aV)
   {
        aRes.Add(*el);
   }
}
   /** Union, functionnal
       Algo : dupl S1, then += S2
   */
template <class Type>  cExtSet<Type>  operator + (const cExtSet<Type> & aS1,const cExtSet<Type> & aS2)
{
   cExtSet<Type>  aRes = aS1.Dupl();
   aRes += aS2;
   return  aRes;
}

 //=========================== Difference  ================

   /** Difference , in - situ
       Algo : Parse a filter and supress from res
   */
template <class Type>  void operator -= (cExtSet<Type> & aRes,const cExtSet<Type> & aFilter)
{
   std::vector<const Type *>  aV;
   aFilter.PutInVect(aV,false);
   for (const auto & el : aV)
   {
       aRes.Suppress(*el);
   }
}
   /// Difference , functionnal
template <class Type>  cExtSet<Type>   operator - (const cExtSet<Type> & aS1,const cExtSet<Type> & aS2)
{
   cExtSet<Type>  aRes = aS1.EmptySet();

   std::vector<const Type *>  aV1;
   aS1.PutInVect(aV1,false);

   for (const auto & el : aV1)
        if (! aS2.In(*el))
           aRes.Add(*el);
   return  aRes;
}



//================================================================================
//========================    Comparison/sort/helper =============================
//================================================================================

       /// Comparison on pointed value 
template <class Type> bool CmpOnPtrValue(const Type * & aV1,const Type * & aV2)
{
   return *aV1 < *aV2;
}
       /// Sort on pointed value and not adress  
template <class Type> void SortPtrValue(std::vector<Type*> & aV)
{
    std::sort(aV.begin(),aV.end(),CmpOnPtrValue<Type>);
}

       /// Specialization 4 void who would not compile
template <> void SortPtrValue(std::vector<const void*>&)
{
}

/* ============================================ */
/*                                              */
/*               cUnorderedSet<Type>            */
/*                                              */
/* ============================================ */

///  unordered_set implementation of cDataExtSet

/** This class is (one of the)  a concret implementation of
   the pure virtual interface class  cDataExtSet
*/

template  <class Type> class cUnorderedSet : public cDataExtSet<Type>
{
     public :
         cUnorderedSet() {}
         ~cUnorderedSet()  { }

         // Big three of a set
         bool Add(const Type & aVal) override
         { 
             return mUS.insert(aVal).second; 
         }
         bool Suppress(const Type & aVal)  override
         {
             return  mUS.erase(aVal) != 0;
         }
         bool In(const Type & aVal)  const  override
         {
            return mUS.find(aVal) !=  mUS.end();
         }
         int size() const override {return mUS.size();}

         //  For parsing
         void  PutInVect(std::vector<const Type*> & aV,bool Sorted) const override
         {
            aV.clear();
            for (const auto & el:mUS)
            {
                aV.push_back(&el);
            }
            if (Sorted)
               SortPtrValue(aV);
         }
         void    clear() override
         {
             mUS.clear();
         }

         // dupl
         cUnorderedSet<Type> * TypedDupl() const
         {
             cUnorderedSet<Type> * aRes = new cUnorderedSet<Type>;
             aRes->mUS = mUS;
             return aRes;
         }
         cDataExtSet<Type> *  VDupl() const override { return TypedDupl(); }

         // empty
         cUnorderedSet<Type> * TypedEmptySet() const     { return new cUnorderedSet<Type>; }
         cDataExtSet<Type> *  VEmptySet() const override { return TypedEmptySet(); }

     private :
          std::unordered_set<Type>  mUS;
};


template<class Type> cDataExtSet<Type> * cDataExtSet<Type>::AllocFromType(eTySC aKindOf)
{
   if (aKindOf==eTySC::US)
      return new cUnorderedSet<Type> ;
   if (aKindOf==eTySC::NonInit)
      return nullptr;
   MMVII_INTERNAL_ASSERT_always(false,"Bad type in cDataExtSet<Type>::AllocFromType")
   return nullptr;
}



    /// MACRO INSTANTIATION

#define INSTANTIATE_SET(Type)\
template  class cDataExtSet<Type>;\
template  class cExtSet<Type>;\
template  class cUnorderedSet<Type>;\
template  void   operator *= (cExtSet<Type> & aRes,const cExtSet<Type> & aFilter); \
template  cExtSet<Type>   operator *  (const cExtSet<Type> & aRes,const cExtSet<Type> & aFilter); \
template  void   operator += (cExtSet<Type> & aRes,const cExtSet<Type> & aFilter); \
template  cExtSet<Type>   operator +  (const cExtSet<Type> & aRes,const cExtSet<Type> & aFilter); \
template  void   operator -= (cExtSet<Type> & aRes,const cExtSet<Type> & aFilter); \
template  cExtSet<Type>   operator -  (const cExtSet<Type> & aRes,const cExtSet<Type> & aFilter); 
/*
*/



INSTANTIATE_SET(int)
INSTANTIATE_SET(void *)
INSTANTIATE_SET(std::string)
INSTANTIATE_SET(tNamePair)

/* ======================================================= */
/*                                                         */
/*         String specialization                           */
/*                                                         */
/* ======================================================= */

/*
    For string the input output is more complicated, as the 
   input can a pattern or a file, input and output can be V1 or V2
*/
tNameSet SetNameFromFile(const std::string& aNameFile,int aNumV)
{
    cMMVII_Appli::SignalInputFormat(aNumV);
    if (aNumV==1)
    {
       return MMV1InitSet(aNameFile);
    }

    tNameSet aSet(eTySC::US);
    ReadFromFileWithDef(aSet,aNameFile);

   return aSet;

}

tNameSet SetNameFromString(const std::string & aName,bool AllowPat)
{
   if (IsFileXmlOfGivenTag(true,aName,TagSetOfName)) // MMVII
   {
      return SetNameFromFile(aName,2);
   }
   else if (IsFileXmlOfGivenTag(false,aName,MMv1XmlTag_SetName))  // MMv1
   {
      return SetNameFromFile(aName,1);
   }
   else if (AllowPat)
   {
      return SetNameFromPat(aName);
   }
   // If we are here, we accept  file as empty set, but not file of bad format
   if (ExistFile(aName))
   {
      MMVII_UsersErrror(eTyUEr::eBadFileSetName,"Set from string, file has bad format for :" +aName);
   }
   return  tNameSet(); // emty set mode US
}

tNameSet SetNameFromPat(const std::string& aFullPat)
{
     std::string aDir,aPat;
     SplitDirAndFile(aDir,aPat,aFullPat,false);
     std::vector<std::string> aV;
     tNameSet aRes;

     GetFilesFromDir(aV,aDir,AllocRegex(aPat));
     for (const auto & el : aV)
        aRes.Add(el);
     return aRes;
}

std::vector<std::string>  ToVect(const tNameSet & aSet)
{
   std::vector<std::string> aRes;
   std::vector<const std::string *> aBuf;
   aSet.PutInVect(aBuf,true);
   for (const auto & aPStr : aBuf)
       aRes.push_back(*aPStr);

   return aRes;
}

/** Return Rel from xml file, noting if do not exist, result in &Exist
*/

tNameRel  RelNameFromXmlFileIfExist (const std::string& aNameFile,bool &Exist)
{
   Exist = true;
   if (IsFileXmlOfGivenTag(true,aNameFile,TagSetOfCpleName)) // MMVII
   {
       tNameRel aSet(eTySC::US);
       ReadFromFileWithDef(aSet,aNameFile);

       return aSet;
   }
   else if (IsFileXmlOfGivenTag(false,aNameFile,MMv1XmlTag_RelName))  // MMv1
   {
       return MMV1InitRel(aNameFile);
   }
   Exist = false;

   return  tNameRel(); // emty set mode US
}

/**  Return Rel from xml file, error if file exist but not to good format */

tNameRel  RelNameFromFile (const std::string& aNameFile)
{
   bool Exist;
   tNameRel  aRes= RelNameFromXmlFileIfExist(aNameFile,Exist);
   if (Exist)
      return aRes;

   if (ExistFile(aNameFile))
   {
      MMVII_UsersErrror(eTyUEr::eBadFileRelName,"Rel from string, file has bad format for :" +aNameFile);
   }
   return  tNameRel(); // emty rel mode US
}


/* ======================================================= */
/*                                                         */
/*             BENCH                                       */
/*                                                         */
/* ======================================================= */
template <class Type> Type FromInt (int);

template <> int FromInt<int> (int i){return i;}
template <> std::string FromInt<std::string> (int i){return ToStr(i);}
template <> void * FromInt<void *> (int i)
{
   static constexpr  int aNb=100;
   static char aTab[aNb];
   MMVII_INTERNAL_ASSERT_always((i>=0) && (i<aNb)," TplBenchSet FromInt/void *")

   return  (aTab+i);
}

    /// ========================= Basic Bench =============

template <class Type> void TplBenchSet(const std::string & aDir)
{
   // Ancien BenchSet
   {
      cExtSet<Type>  aSI;
      for (int aK=0 ; aK<10 ; aK++)
         aSI.Add(FromInt<Type>(aK*2));
      for (int aK=0 ; aK<20 ; aK++)
          MMVII_INTERNAL_ASSERT_bench(aSI.In(FromInt<Type>(aK))==(aK%2==0),"BenchSet");
   }


   {
      cExtSet<Type>  aS2;
      cExtSet<Type>  aS3;

      for (int aT=0 ; aT<2 ; aT++) // Normally we can add twice, it does not change anything
      {
          for (int aK=0 ; aK<100; aK++)
          {
               if ((aK%2)==0) aS2.Add(FromInt<Type>(aK));
               if ((aK%3)==0) aS3.Add(FromInt<Type>(aK));
          }
      }
      cExtSet<Type> aSP2;

      cExtSet<Type> aSPlus =  aS2 + aS3;
      cExtSet<Type> aSMoins = aS2 - aS3;
      cExtSet<Type> aSMul =   aS2 * aS3;

      for (int aK=0 ; aK<100; aK++)
      {
         MMVII_INTERNAL_ASSERT_bench(aSPlus.In(FromInt<Type>(aK))==((aK%2==0)||(aK%3==0)),"BenchSet+");
         MMVII_INTERNAL_ASSERT_bench(aSMoins.In(FromInt<Type>(aK))==((aK%2==0)&&(aK%3!=0)),"BenchSet-");
         MMVII_INTERNAL_ASSERT_bench(aSMul.In(FromInt<Type>(aK))==((aK%2==0)&&(aK%3==0)),"BenchSet*");
         if ((aK%2==0)||(aK%3==0)) aSP2.Add(FromInt<Type>(aK));
      }
      MMVII_INTERNAL_ASSERT_bench(aSP2.Equal(aSPlus),"BenchSet eq");
      MMVII_INTERNAL_ASSERT_bench(!aSP2.Equal(aSMoins),"BenchSet eq");
      MMVII_INTERNAL_ASSERT_bench(!aSP2.Equal(aSMul),"BenchSet eq");

      MMVII_INTERNAL_ASSERT_bench(aS2.IncludedIn(aSP2),"BenchSet eq");
      MMVII_INTERNAL_ASSERT_bench(aS3.IncludedIn(aSP2),"BenchSet eq");

      MMVII_INTERNAL_ASSERT_bench(!aSP2.IncludedIn(aSMoins),"BenchSet eq");
      MMVII_INTERNAL_ASSERT_bench( aSMoins.IncludedIn(aSP2),"BenchSet eq");
   }
}

void BenchSet(cParamExeBench & aParam,const std::string & aDir)
{
    if (! aParam.NewBench("Set")) return;

    cMMVII_Appli &  anAp = cMMVII_Appli::CurrentAppli();

    TplBenchSet<int>        (aDir);
    TplBenchSet<std::string> (aDir);
    TplBenchSet<void *> (aDir);

    // Basic bench of set of cple of name
    {
       tNameRel aTest;
       tNameRel aT2;

       aTest.Add(tNamePair("b","a"));
       aTest.Add(tNamePair("c","d"));
       aTest.Add(tNamePair("d","c"));
       
       MMVII_INTERNAL_ASSERT_bench(aTest.size()==2,"BenchSet");
       std::string aNameFile = anAp.TmpDirTestMMVII() + "TestRel.xml";
       SaveInFile(aTest,aNameFile);
       ReadFromFileWithDef(aT2,aNameFile);

       MMVII_INTERNAL_ASSERT_bench(aT2.size()==2,"BenchSet");
       MMVII_INTERNAL_ASSERT_bench(aT2.Equal(aTest),"BenchSet");
    }
    aParam.EndBench();
}

};

