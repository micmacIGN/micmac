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

///  unordered_set implementation of cInterfSet

/** This class is (one of the)  a concret implementation of
   the pure virtual interface class  cInterfSet
*/

// template <class Type> void SortPtrValue(std::vector<const Type*> aV);

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




// bool  


template  <class Type> class cUnorderedSet : public cInterfSet<Type>
{
     public :
         cUnorderedSet() {}
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

         void  PutInSet(std::vector<const Type*> & aV,bool Sorted) const override
         {
            aV.clear();
            for (const auto & el:mUS)
            {
                aV.push_back(&el);
            }
            if (Sorted)
            {
               SortPtrValue(aV);
            }
         }
         void    clear() override
         {
             mUS.clear();
         }
         cInterfSet<Type> *  VDupl() const override
         {
             return TypedDupl();
         }

         cUnorderedSet<Type> * TypedDupl() const
         {
             cUnorderedSet<Type> * aRes = new cUnorderedSet<Type>;
             aRes->mUS = mUS;
             return aRes;
         }


         ~cUnorderedSet()  
         {
         }
         int size() const override {return mUS.size();}

     private :
          std::unordered_set<Type>  mUS;
};


/* ============================================ */
/*                                              */
/*               cInterfSet<Type>               */
/*                                              */
/* ============================================ */

  // definition of interface class

template <class Type> cInterfSet<Type>::~cInterfSet()
{
}

 //=========================== Intersection ================

   /// Intersection, in - situ

template <class Type>  cInterfSet<Type> &  operator *= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter)
{
   std::vector<const Type *>  aV;
   aRes.PutInSet(aV,false);
   for (const auto & el : aV)
   {
       if ( ! aFilter.In(*el))
          aRes.Suppress(*el);
   }
   return aRes;
}
   /// Intersection, fonctionnal
template <class Type>  cInterfSet<Type> *  operator * (const cInterfSet<Type> & aS1,const cInterfSet<Type> & aS2)
{
   cInterfSet<Type> * aRes = AllocUS<Type>();
   (*aRes) = aS1;
   (*aRes) *= aS2;
   return  aRes;
}

 //=========================== Union  ================

   /// Union, in - situ
template <class Type>  cInterfSet<Type> &  operator += (cInterfSet<Type> & aRes,const cInterfSet<Type> & toAdd)
{
   std::vector<const Type *>  aV;
   toAdd.PutInSet(aV,false);
   for (const auto & el : aV)
   {
        aRes.Add(*el);
   }
   return aRes;
}
   /// Union, functionnal
template <class Type>  cInterfSet<Type> *  operator + (const cInterfSet<Type> & aS1,const cInterfSet<Type> & aS2)
{
   cInterfSet<Type> * aRes = AllocUS<Type>();
   (*aRes) = aS1;
   (*aRes) += aS2;
   return  aRes;
}

 //=========================== Affectation  ================

template <class Type>  cInterfSet<Type> &  cInterfSet<Type>::operator = (const cInterfSet<Type> & toAdd)
{
   clear();
   *this += toAdd;
   return *this;
}


 //=========================== Difference  ================

   /// Difference , in - situ
template <class Type>  cInterfSet<Type> &  operator -= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter)
{
   std::vector<const Type *>  aV;
   aRes.PutInSet(aV,false);
   for (const auto & el : aV)
   {
       if (aFilter.In(*el))
          aRes.Suppress(*el);
   }
   return aRes;
}
   /// Difference , functionnal
template <class Type>  cInterfSet<Type> *  operator - (const cInterfSet<Type> & aS1,const cInterfSet<Type> & aS2)
{
   cInterfSet<Type> * aRes = AllocUS<Type>();
   (*aRes) = aS1;
   (*aRes) -= aS2;
   return  aRes;
}



template <class Type> cInterfSet<Type> * AllocUS()
{
   return new cUnorderedSet<Type>;
}


    /// MACRO INSTANTIATION

#define INSTANTIATE_SET(Type)\
template  class cInterfSet<Type>;\
template  class cUnorderedSet<Type>;\
template  cInterfSet<Type> &  operator *= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> &  operator += (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> &  operator -= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> *  operator +  (const cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> *  operator *  (const cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> *  operator -  (const cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> * AllocUS<Type>(); ///<  unordered_set

INSTANTIATE_SET(int)
INSTANTIATE_SET(void *)
INSTANTIATE_SET(std::string)


    /// ========================= Basic Bench =============

void BenchSet(const std::string & aDir)
{
   // Ancien BenchSet
   {
      cInterfSet<int> * aSI = AllocUS<int>();
      for (int aK=0 ; aK<10 ; aK++)
         aSI->Add(aK*2);
      for (int aK=0 ; aK<20 ; aK++)
          MMVII_INTERNAL_ASSERT_bench(aSI->In(aK)==(aK%2==0),"BenchSet");
      delete aSI;
      std::cout << "BenchSetBenchSetBenchSet \n";
   }

   {
      cInterfSet<int> * aS2= AllocUS<int>();
      cInterfSet<int> * aS3= AllocUS<int>();

      for (int aT=0 ; aT<2 ; aT++) // Normally we can add twice, it does not change anything
      {
          for (int aK=0 ; aK<100; aK++)
          {
               if ((aK%2)==0) aS2->Add(aK);
               if ((aK%3)==0) aS3->Add(aK);
          }
      }

      cInterfSet<int> * aSPlus = (*aS2) + (*aS3);
      cInterfSet<int> * aSMoins = (*aS2) - (*aS3);
      cInterfSet<int> * aSMul = (*aS2) * (*aS3);

      for (int aK=0 ; aK<100; aK++)
      {
         // std::cout << "K=" << aK << " " << aS2->In(aK) << " " << aS3->In(aK) << " " << aSPlus->In(aK) << "\n";
         MMVII_INTERNAL_ASSERT_bench(aSPlus->In(aK)==((aK%2==0)||(aK%3==0)),"BenchSet+");
         MMVII_INTERNAL_ASSERT_bench(aSMoins->In(aK)==((aK%2==0)&&(aK%3!=0)),"BenchSet-");
         MMVII_INTERNAL_ASSERT_bench(aSMul->In(aK)==((aK%2==0)&&(aK%3==0)),"BenchSet*");
      }

      delete aS2;
      delete aS3;
      delete aSPlus;
      delete aSMoins;
      delete aSMul;
   }
}


};

