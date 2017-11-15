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
            for (auto el:mUS)
                aV.push_back(&el);
         }
         void    clear() override
         {
             mUS.clear();
         }

         ~cUnorderedSet()  
         {
         }

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
   for (auto el : aV)
   {
       if ( ! aFilter.In(*el))
          aRes.Suppress(*el);
   }
   
   return aRes;
}


/*
        cInterfSet &  operator *=(const cInterfSet &);
         cInterfSet &  operator +=(const cInterfSet &);
         cInterfSet &  operator -=(const cInterfSet &);
         cInterfSet &  operator =(const cInterfSet &);
*/


template <class Type> cInterfSet<Type> * AllocUS()
{
   return new cUnorderedSet<Type>;
}

#define INSTANTIATE_SET(Type)\
template  class cInterfSet<Type>;\
template  class cUnorderedSet<Type>;\
template  cInterfSet<Type> &  operator *= (cInterfSet<Type> & aRes,const cInterfSet<Type> & aFilter); \
template  cInterfSet<Type> * AllocUS<Type>(); ///<  unordered_set

INSTANTIATE_SET(int)
INSTANTIATE_SET(void *)
INSTANTIATE_SET(std::string)

/*
template  class cInterfSet<int>;
template  class cUnorderedSet<int>;
template  cInterfSet<int> * AllocUS<int>(); ///<  unordered_set
*/


};

