#include "include/MMVII_all.h"

namespace MMVII
{

std::string timePointAsString(const std::chrono::system_clock::time_point& tp) 
{
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::string ts = std::ctime(&t);
    ts.resize(ts.size()-1);
    return ts;
}

// 3600.0 * 24.0

cMMVII_Ap_CPU::cMMVII_Ap_CPU() :
   mT0          (std::chrono::system_clock::now())     ,
   mPid         (mmvii_GetPId())   ,
   mNbProcSys   (mmvii_NbProcSys())
{
  // Very tricky and dirty, but I dont have courage for now to understand time/clok in C++
  // To change one day ....
  {
     const auto p0 = std::chrono::time_point<std::chrono::system_clock>{};  // Epoch 1/1/70
     std::chrono::duration<double, std::milli> fp_ms = mT0 -p0;  // duration from epoch
     double aT = fp_ms.count()/1000.0;  // In millisecond
     double  aSecPerDay = 24 * 3600;   // 24h/Day, 3600 sec/h ....
     aT = aT - (365.24219 *49 )  * aSecPerDay;  // Make epoch more or less to 1/1/2019 , use gregorian year in day
     aT = aT / aSecPerDay;  /// 
     aT += 0.76736 -  0.816726 ;  // Experimental difference !!
     int aNbDay = round_down(aT);
     aT = aT - aNbDay;  // take fractionnal part
     aT = aT * aSecPerDay ;  // 
     int aSec = round_down(aT);
     int aMili10 = std::min(9999,round_ni(1e4*(aT-aSec)));
     mStrIdTime =   ToStr(aNbDay,4)  + "_" + ToStr(aSec,5) +"_" + ToStr(aMili10,4) ;
  }
}

std::string    cMMVII_Ap_CPU::StrDateBegin() const    {return  timePointAsString(mT0);}
std::string    cMMVII_Ap_CPU::StrDateCur() const    {return  timePointAsString(std::chrono::system_clock::now());}
const std::string  &  cMMVII_Ap_CPU::StrIdTime() const  {return mStrIdTime;}


double cMMVII_Ap_CPU::SecFromT0() const
{
    tTime aT1 = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = aT1-mT0;
    return fp_ms.count()/1000.0;

}


};

