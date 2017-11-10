#include "include/MMVII_all.h"
#include <unordered_set>
#include <unordered_map>


namespace MMVII
{


template  <class Type> class cUnorderedSet : public cInterfSet<Type>
{
     public :
          cUnorderedSet() {}
         void Add(const Type & aVal) override
         { 
             mUS.insert(aVal); 
         }

         bool In(const Type & aVal)   override
         {
            return mUS.find(aVal) !=  mUS.end();
         }

         ~cUnorderedSet()  
         {
         }

     private :
          std::unordered_set<Type>  mUS;
};


template <class Type> cInterfSet<Type>::~cInterfSet()
{
}

template <class Type> cInterfSet<Type> * AllocUS()
{
   return new cUnorderedSet<Type>;
}

#define INSTANTIATE_SET(Type)\
template  class cInterfSet<Type>;\
template  class cUnorderedSet<Type>;\
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

