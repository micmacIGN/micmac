#include "include/MMVII_all.h"

namespace MMVII
{

/**
    This file contains the implemenation of conversion between strings and 
   atomic object
*/

static char BufStrIO[1000];

//  vector<int>  => [1,2,3]

template <class Type>  std::string Vect2Str(const std::vector<Type>  & aV)
{
   std::string aRes ="[";
   for (int aK=0 ; aK<(int)aV.size() ; aK++)
   {
      if (aK>0)
         aRes += ",";
      aRes += ToStr(aV[aK]);
   }
   aRes += "]";
   return aRes;
}

template <class Type>  std::vector<Type> Str2Vec(const std::string & aStrGlob)
{
   std::vector<Type> aRes;
   const char * aC=aStrGlob.c_str();
   if (*aC!='[')
       MMVII_UsersErrror(eTyUEr::eParseError,"expected [ at beging of vect");
   aC++;
   while((*aC) && *aC!=']')
   {
       std::string aStrV;
       while ((*aC) && (*aC!=',') && (*aC!=']'))
          aStrV += *(aC++);
       if (!(*aC))
          MMVII_UsersErrror(eTyUEr::eParseError,"unexpected end of string while expecting \",\"");
       aRes.push_back(cStrIO<Type>::FromStr(aStrV)); 
       if (*aC==',')
          aC++;
   }
   if (*aC!=']')
      MMVII_UsersErrror(eTyUEr::eParseError,"unexpected end of string while expecting \"]\"");
   aC++;

   return  aRes;
}

                          //   - - std::vector<std::string>  - -

template <>  std::string cStrIO<std::vector<std::string> >::ToStr(const std::vector<std::string>  & aV)
{
   return  Vect2Str(aV);
}
template <>  std::vector<std::string> cStrIO<std::vector<std::string> >::FromStr(const std::string & aStr)
{
    return Str2Vec<std::string>(aStr);
}
template <>  const std::string cStrIO<std::vector<std::string>>::msNameType = "std::vector<std::string>";




void OneBenchStrIO(std::string aStr,const  std::vector<std::string> & aV)
{
   // std::string aStr =
   std::vector<std::string> aV2 =  cStrIO<std::vector<std::string> >::FromStr(aStr);
   if(aV!=aV2)
   {
      std::cout << "STR=" << aStr << "\n";
      std::cout << "VEC=[";
      for (int aK=0 ; aK<int(aV2.size()) ; aK++)
      {
         if (aK!=0) std::cout << ",";
         std::cout << "{" << aV2[aK] << "}";
      }
      std::cout << "]\n";
      MMVII_INTERNAL_ASSERT_bench((aV==aV2),"OneBenchStrIO");
   }
}

void BenchStrIO()
{
   OneBenchStrIO("[1,2,3]",{"1","2","3"});
   OneBenchStrIO("[1]",{"1"});
   OneBenchStrIO("[]",{});
   OneBenchStrIO("[,1]",{"","1"});
   OneBenchStrIO("[,,,1]",{"","","","1"});
   OneBenchStrIO("[,]",{""});
   OneBenchStrIO("[1,2,]",{"1","2"});
   OneBenchStrIO("[1,2,,]",{"1","2",""});
   OneBenchStrIO("[1,2,,,,]",{"1","2","","",""});
   // OneBenchStrIO("[,,1,,3,]",{"","","1","","3",""});
   // Check that we get an error catched
   //    OneBenchStrIO("[",{});
   //    OneBenchStrIO("[1,2",{});
   // getchar();
}

   // ================  bool ==============================================

template <>  std::string cStrIO<bool>::ToStr(const bool & anI)
{
   return  anI ? "true" : "false";
}
template <>  bool cStrIO<bool>::FromStr(const std::string & aStr)
{
    if ((aStr=="1") || UCaseEqual(aStr,"true")) return true;
    if ((aStr=="0") || UCaseEqual(aStr,"false")) return false;

    MMVII_INTERNAL_ASSERT_user(eTyUEr::eBadBool,"Bad value for boolean :["+aStr+"]");

    return false;
}

template <>  const std::string cStrIO<bool>::msNameType = "bool";

   // ================  int ==============================================

template <>  std::string cStrIO<int>::ToStr(const int & anI)
{
   sprintf(BufStrIO,"%d",anI);
   return BufStrIO;
}
template <>  int cStrIO<int>::FromStr(const std::string & aStr)
{
    // can be convenient that empty string correspond to zero
    if (aStr.empty())
       return 0;
    int anI;
    sscanf(aStr.c_str(),"%d",&anI);
    return anI;
}
template <>  const std::string cStrIO<int>::msNameType = "int";

std::string ToStr(int aVal,int aSzMin)
{
   std::string aRes = ToStr(std::abs(aVal));
   while (int(aRes.size())<aSzMin)
       aRes = "0" + aRes;
   if (aVal<0)
       aRes = "-" + aRes;
   return aRes;
}
/*
std::string  ToS_NbDigit(int aNb,int aNbDig,bool AcceptOverFlow)
{
   std::string aRes = ToS(aNb);
   int aSz = (int)aRes.size();
   if ((!AcceptOverFlow) && (aSz>aNbDig))
   {
       MMVII_INTERNAL_ASSERT_user(eTyUEr::eTooBig4NbDigit,"Pas assez de digit dans ToStringNBD")
   }
   for (;aSz<aNbDig ; aSz++)
   {
       aRes = "0" + aRes;
   }
   return aRes;
}
*/


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
template <>  const std::string cStrIO<double>::msNameType = "double";

std::string FixDigToStr(double aSignedVal,int aNbDig)
{
   std::string aFormat = "%."+ToS(aNbDig) + "f";
   char aBuf[100];
   sprintf(aBuf,aFormat.c_str(),aSignedVal);
   return aBuf;
/*
   printf("%3.2f\n",d);
   double aAbsV = std::abs(aSignedVal);
   double aFrac = FracPart(aAbsV);
   int aIVal = aVal - aFrac;

   std::string aRes = ToStr(aIVal) + "." + ToStr(round_ni(aFrac*pow(10.0,aNdDig)),aNbDig);

   return ToStr(aIVal)
*/
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

template <>  const std::string cStrIO<std::string>::msNameType = "string";

};
