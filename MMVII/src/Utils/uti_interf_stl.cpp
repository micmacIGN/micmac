#include "include/MMVII_all.h"
#include <unordered_set>
#include <unordered_map>


namespace MMVII
{


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

         void  PutInSet(std::vector<const Type*> & aV) const override
         {
            for (const auto & el:mUS)
                aV.push_back(&el);
         }
         void    clear() override
         {
             mUS.clear();
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

template <class Type> cInterfSet<Type>::~cInterfSet()
{
}

template <class Type>  cInterfSet<Type> &  operator *= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter)
{
   std::vector<const Type *>  aV;
   aRes.PutInSet(aV);
   for (const auto & el : aV)
   {
       if ( ! aFilter.In(*el))
          aRes.Suppress(*el);
   }
   return aRes;
}
template <class Type>  cInterfSet<Type> *  operator * (const cInterfSet<Type> & aS1,const cInterfSet<Type> & aS2)
{
   cInterfSet<Type> * aRes = AllocUS<Type>();
   (*aRes) = aS1;
   (*aRes) *= aS2;
   return  aRes;
}

template <class Type>  cInterfSet<Type> &  operator += (cInterfSet<Type> & aRes,const cInterfSet<Type> & toAdd)
{
   std::vector<const Type *>  aV;
   toAdd.PutInSet(aV);
// std::cout << "PutInSet " << aV.size() << "\n";
   for (const auto & el : aV)
   {
        aRes.Add(*el);
   }
   return aRes;
}
template <class Type>  cInterfSet<Type> *  operator + (const cInterfSet<Type> & aS1,const cInterfSet<Type> & aS2)
{
   cInterfSet<Type> * aRes = AllocUS<Type>();
// std::cout << "SSSSS " << aRes->size() << " " << aS1.size() << " " << aS2.size() << "\n";
   (*aRes) = aS1;
// std::cout << "SSSSS " << aRes->size() << "\n";
   (*aRes) += aS2;
// std::cout << "SSSSS " << aRes->size() << "\n";
   return  aRes;
}

template <class Type>  cInterfSet<Type> &  cInterfSet<Type>::operator = (const cInterfSet<Type> & toAdd)
{
// std::cout << "=========== " << size()<< " " << toAdd.size() << "\n";
   clear();
   *this += toAdd;
   return *this;
}


template <class Type>  cInterfSet<Type> &  operator -= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter)
{
   std::vector<const Type *>  aV;
   aRes.PutInSet(aV);
   for (const auto & el : aV)
   {
       if (aFilter.In(*el))
          aRes.Suppress(*el);
   }
   return aRes;
}
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

/*
template  class cInterfSet<int>;
template  class cUnorderedSet<int>;
template  cInterfSet<int> * AllocUS<int>(); ///<  unordered_set
*/

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

