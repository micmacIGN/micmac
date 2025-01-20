#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"

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

/***********************************/
/*                                 */
/*           cMMVII_Ap_CPU         */
/*                                 */
/***********************************/

cMMVII_Ap_CPU::cMMVII_Ap_CPU() :
   mT0             (std::chrono::system_clock::now())     ,
   mPid            (mmvii_GetPId())   ,
   mNbProcSystem   (mmvii_NbProcSys()),
   mNbProcAllowed  (mNbProcSystem),
   mMulNbInMk      (10.0),
   mTimeSegm       (this)
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


cTimerSegm&  cMMVII_Ap_CPU::TimeSegm() {return mTimeSegm;}

double cMMVII_Ap_CPU::SecFromT0() const
{
    tTime aT1 = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = aT1-mT0;
    return fp_ms.count()/1000.0;

}

/***********************************/
/*                                 */
/*          cTimeSequencer         */
/*                                 */
/***********************************/

cTimeSequencer::cTimeSequencer(double aPeriod):
    mPeriod   (aPeriod),
    mLastime  (cMMVII_Appli::CurrentAppli().SecFromT0())
{
}

bool cTimeSequencer::ItsTime2Execute()
{
     double aTime = cMMVII_Appli::CurrentAppli().SecFromT0();

     if (aTime < (mLastime+mPeriod))
        return false;

     mLastime += mPeriod;

     return true;
}


/***********************************/
/*                                 */
/*          cAutoTimerSegm         */
/*                                 */
/***********************************/

cAutoTimerSegm::cAutoTimerSegm(cTimerSegm & aTS,const tIndTS & anInd) :
   mTS      (&aTS), 
   mSaveInd (mTS->mLastIndex)
{
   mTS->SetIndex(anInd);
}

cAutoTimerSegm::cAutoTimerSegm(const tIndTS & anInd) :
    cAutoTimerSegm(GlobAppTS(),anInd)
{
}

cAutoTimerSegm::cAutoTimerSegm(cTimerSegm * aTS,const tIndTS & anInd) :
	mTS (aTS)
{
   if (aTS)
   {
      mSaveInd =  aTS->mLastIndex;
      mTS->SetIndex(anInd);
   }
}

cAutoTimerSegm::~cAutoTimerSegm()
{
   if (mTS)
      mTS->SetIndex(mSaveInd);
}

/***********************************/
/*                                 */
/*          cTimerSegm             */
/*                                 */
/***********************************/

static const std::string DefTime("OTHERS");

cTimerSegm::cTimerSegm(cMMVII_Ap_CPU * anAppli) :
   mLastIndex     (DefTime),
   mAppli         (anAppli),
   mCurBeginTime  (mAppli->SecFromT0()),
   mShowAtDel     (true)
{
}

double cTimerSegm::CurBeginTime() const {return  mCurBeginTime;}

void  cTimerSegm::SetNoShowAtDel()
{
    mShowAtDel = false;
}

cTimerSegm::~cTimerSegm()
{
   if (mShowAtDel && (mTimers.size() >=2)) // is something was added
     Show();
}

cTimerSegm & GlobAppTS()
{
   return cMMVII_Appli::CurrentAppli().TimeSegm();
}

const tTableIndTS &  cTimerSegm::Times() const {return mTimers;}

void cTimerSegm::SetIndex(const tIndTS & aInd)
{
   double aCurTime =  mAppli->SecFromT0();
   mTimers[mLastIndex] += aCurTime-mCurBeginTime;
   mCurBeginTime = aCurTime;
   mLastIndex = aInd;
}

void cTimerSegm::Show() 
{
   {
      cAutoTimerSegm aATS(*this,DefTime);
   }

   double aSom = 0.0;
   StdOut()  <<  " ========== TIMING ===========" << std::endl;
   for (const auto & aPair : mTimers)
   {
       aSom += aPair.second;
       StdOut() << " * "  << FixDigToStr(aPair.second,4,4) << " : " << aPair.first << std::endl;
   }

   StdOut() << " *** SOM " << aSom  <<  " " << mAppli->SecFromT0() << std::endl;
}


};

