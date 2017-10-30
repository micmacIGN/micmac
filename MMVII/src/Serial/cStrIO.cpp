#include "include/MMVII_all.h"

namespace MMVII
{

/**
    This file contains the implemenation of conversion between strings and 
   atomic object
*/

static char BufStrIO[1000];

   // ================  int ==============================================

template <>  std::string cStrIO<int>::ToS(const int & anI)
{
   sprintf(BufStrIO,"%d",anI);
   return BufStrIO;
}
template <>  int cStrIO<int>::FromS(const std::string & aStr)
{
    int anI;
    sscanf(aStr.c_str(),"%d",&anI);
    return anI;
}

   // ================  double ==============================================

template <>  std::string cStrIO<double>::ToS(const double & anI)
{
   sprintf(BufStrIO,"%lf",anI);
   return BufStrIO;
}
template <>  double cStrIO<double>::FromS(const std::string & aStr)
{
    double anI;
    sscanf(aStr.c_str(),"%lf",&anI);
    return anI;
}

   // ================  std::string ==============================================

template <>  std::string cStrIO<std::string>::ToS(const std::string & aStr)
{
   return aStr;
}
template <>  std::string cStrIO<std::string>::FromS(const std::string & aStr)
{
    return aStr;
}


};
