#include "include/V1VII.h"
#include "include/ext_stl/numeric.h"

namespace MMVII
{

double NC_KthVal(std::vector<double> & aV, double aProportion)
{
   return ::KthValProp(aV,aProportion);
}


double Cst_KthVal(const std::vector<double> & aV, double aProportion)
{
     std::vector<double> aDup= aV;
     return NC_KthVal(aDup,aProportion);
}

};
