#if (0)
#include "include/MMVII_all.h"
#include <unordered_set>
#include <unordered_map>


/** \file uti_interf_stl.cpp
    \brief Instantiation of some template utilitaries

   This file contains implementation of a set of class that allow to
  construct selectors, elemenatary one, or by boolean construction

*/


namespace MMVII
{

/* ============================================ */
/*                                              */
/*               cDataExtSet<Type>              */
/*                                              */
/* ============================================ */

template <class Type> class cDataExtSet : public cMemCheck
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
   MMVII_INTERNAL_ASSERT_always(0,"Bad type in cDataExtSet<Type>::AllocFromType")
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

/* ======================================================= */
/*                                                         */
/*         String specialization                           */
/*                                                         */
/* ======================================================= */

/*
tNameSet SetNameFromString(const std::string&,bool AllowPat);
tNameSet SetNameFromFile(const std::string&);
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
   return  tNameSet(); // emty set mode US
}

tNameSet SetNameFromPat(const std::string& aFullPat)
{
     std::string aDir,aPat;
     SplitDirAndFile(aDir,aPat,aFullPat,false);
     std::vector<std::string> aV;
     tNameSet aRes;

     GetFilesFromDir(aV,aDir,BoostAllocRegex(aPat));
     for (const auto & el : aV)
        aRes.Add(el);
     return aRes;
}



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
      std::cout << "BenchSetBenchSetBenchSet \n";
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

      cExtSet<Type> aSPlus =  aS2 + aS3;
      cExtSet<Type> aSMoins = aS2 - aS3;
      cExtSet<Type> aSMul =   aS2 * aS3;

      for (int aK=0 ; aK<100; aK++)
      {
         // std::cout << "K=" << aK << " " << aS2->In(aK) << " " << aS3->In(aK) << " " << aSPlus->In(aK) << "\n";
         MMVII_INTERNAL_ASSERT_bench(aSPlus.In(FromInt<Type>(aK))==((aK%2==0)||(aK%3==0)),"BenchSet+");
         MMVII_INTERNAL_ASSERT_bench(aSMoins.In(FromInt<Type>(aK))==((aK%2==0)&&(aK%3!=0)),"BenchSet-");
         MMVII_INTERNAL_ASSERT_bench(aSMul.In(FromInt<Type>(aK))==((aK%2==0)&&(aK%3==0)),"BenchSet*");
      }
   }
}

void BenchSet(const std::string & aDir)
{
    TplBenchSet<int>        (aDir);
    TplBenchSet<std::string> (aDir);
    TplBenchSet<void *> (aDir);
}

};

#endif
