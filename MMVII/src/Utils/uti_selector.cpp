#if (0)
#include "include/MMVII_all.h"
#include <boost/optional/optional_io.hpp>
#include <boost/regex.hpp>


/** \file uti_selector.cpp
    \brief Implementation of selector

   This file contains implementation of a set of class that allow to
  construct selectors, elemenatary one, or by boolean construction

*/





namespace MMVII
{
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
       typedef boost::optional<Type> tOObj;
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

template<class Type> cSelector<Type> GenIntervalSelector( const boost::optional<Type> & aV1, 
                                                         const boost::optional<Type> & aV2,
                                                         bool aInclLow,bool InclUp)
{
   return cSelector<Type>(new cDataIntervSelector<Type>(&aV1,&aV2,aInclLow,InclUp));
}

///  ]<- ,V] or ]<-,V[
template<class Type> cSelector<Type> LeftHalfIntervalSelector(const Type & aV, bool aIncl)
{
   boost::optional<Type> aOptV(aV);
   boost::optional<Type> aNone = boost::none;
   return cSelector<Type>(new cDataIntervSelector<Type>(&aNone,&aOptV,aIncl,aIncl));
}

///  ]V,->[ or [V,->[
template<class Type> cSelector<Type> RightHalfIntervalSelector(const Type & aV, bool aIncl)
{
   boost::optional<Type> aOptV(aV);
   boost::optional<Type> aNone = boost::none;
   return cSelector<Type>(new cDataIntervSelector<Type>(&aOptV,&aNone,aIncl,aIncl));
}

template<class Type> cSelector<Type> IntervalSelector(const Type&aV1,const Type&aV2,bool aInclLow,bool InclUp)
{
   boost::optional<Type> aOptV1(aV1);
   boost::optional<Type> aOptV2(aV2);
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

      // std::cout << "LLLLL :" << (<< aStrLow << "###" << aStrUp  << "}\n";
      aC++;
      SkeepWhite(aC);
   }

   if (0)
   {
      for (size_t aK=0  ; aK<aRes.size() ; aK+=2)
      {
         std::cout << (aRes[aK].second ? "[" : "]");
         std::cout << aRes[aK].first << "," << aRes[aK+1].first ;
         std::cout << (aRes[aK+1].second ? "]" : "[");
         std::cout << "\n";
      }
   }
   return aRes;
}

/// Compute union of intervals from a string
template <class Type> cSelector<Type> Str2Interv(const std::string & aStr)
{
    std::vector<tElemStrInterv>  aVI = LireInterv(aStr);
    cSelector<Type> aRes =  CsteSelector<Type>(false);


    for (size_t aK=0  ; aK<aVI.size() ; aK+=2)
    {
         boost::optional<Type> aV1 = boost::none;
         boost::optional<Type> aV2 = boost::none;
         if (aVI[aK].first!="")
            aV1 = cStrIO<Type>::FromStr(aVI[aK].first);
         if (aVI[aK+1].first!="")
            aV2 = cStrIO<Type>::FromStr(aVI[aK+1].first);
         aRes = aRes || GenIntervalSelector(aV1,aV2,aVI[aK].second,aVI[aK+1].second);
    }
     

    return aRes;
}


#define MACRO_INSTANTIATE_SELECTOR(Type)\
template cSelector<Type> GenIntervalSelector(const boost::optional<Type> & aV1, const boost::optional<Type> & aV2, bool aInclLow,bool InclUp);\
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
/*     cBoostRegex                          */
/*                                          */
/* ======================================== */

/// Boost implementation of Regex expression
class cDataBoostRegex : public  cDataSelector<std::string>
{
    public :
        cDataBoostRegex(const std::string & aName):
            mRegex (aName)
        {
        }
        bool Match(const std::string &  aStr) const override 
        {
             return regex_match(aStr,mRegex);
        }
    private :
        boost::regex mRegex;
};

tNameSelector  BoostAllocRegex(const std::string& aPat)
{
   if (aPat=="")
      return  CsteSelector<std::string>(true);
   return cSelector<std::string>(new cDataBoostRegex(aPat));
}




void BenchSelector(const std::string & aDir)
{
    int aLow = 5;
    int aUp = 9;
    cSelector<int> aSII_59 = GenIntervalSelector<int>(aLow,aUp,true,true);
    cSelector<int> aSEE_59 = GenIntervalSelector<int>(aLow,aUp,false,false);

    // For this one bound are unordered 5 include, 9 exclude but swap
    cSelector<int> aSIE_59    = IntervalSelector<int>(aUp,aLow,false,true);
    cSelector<int> aSIEBis_59 = GenIntervalSelector<int>(aLow,aUp,true,false);
    cSelector<int> aSEI_59    = IntervalSelector<int>(aLow,aUp,false,true);

    cSelector<int> aSE_X9 = GenIntervalSelector<int>(boost::none,aUp ,false,false);
    cSelector<int> aSEBis_X9 = LeftHalfIntervalSelector<int>(aUp ,false);

    cSelector<int> aSI_5X = GenIntervalSelector<int>(aLow,boost::none,true,true);
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
    }
}

};

#endif
