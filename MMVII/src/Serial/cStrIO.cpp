#include "include/MMVII_all.h"

namespace MMVII
{

/**
    This file contains the implemenation of conversion between strings and 
   atomic object
*/

static char BufStrIO[1000];

   // ================  int ==============================================

template <>  std::string cStrIO<int>::ToStr(const int & anI)
{
   sprintf(BufStrIO,"%d",anI);
   return BufStrIO;
}
template <>  int cStrIO<int>::FromStr(const std::string & aStr)
{
    int anI;
    sscanf(aStr.c_str(),"%d",&anI);
    return anI;
}

   // ================  double ==============================================

template <>  std::string cStrIO<double>::ToStr(const double & anI)
{
   sprintf(BufStrIO,"%lf",anI);
   return BufStrIO;
}
template <>  double cStrIO<double>::FromStr(const std::string & aStr)
{
    double anI;
    sscanf(aStr.c_str(),"%lf",&anI);
    return anI;
}

   // ================  std::string ==============================================

template <>  std::string cStrIO<std::string>::ToStr(const std::string & aStr)
{
   return aStr;
}
template <>  std::string cStrIO<std::string>::FromStr(const std::string & aStr)
{
    return aStr;
}


};
