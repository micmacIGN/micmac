#include "MMVII_Stringifier.h"
#include "cMMVII_Appli.h"

namespace MMVII
{

/*=============================================*/
/*                                             */
/*      cMMVII_Duration::                      */
/*                                             */
/*=============================================*/

std::string  cMMVII_Duration::ToDaisyStr(std::string * aFormat,bool Full) const
{
   return ToString(':',2,aFormat,Full);
}

std::string  cMMVII_Duration::ToString(char aSep,int aNbDigFrac,std::string * aFormat,bool Full) const
{
   bool FormatDone= false;
   std::string aRes;
   if (mNbDay || Full)
   {
      aRes = ToStr(mNbDay,2) + aSep;
      if (aFormat && (!FormatDone)) 
      {
         *aFormat= "dd:hh:mm:ss";
         FormatDone= true;
      }
   }

   if (mNbHour|| (aRes!="") || Full)
   {
      aRes += ToStr(mNbHour,2) + aSep;
      if (aFormat && (!FormatDone)) 
      {
         *aFormat= "hh:mm:ss";
         FormatDone= true;
      }
   }

   if (mNbMin|| (aRes!="") || Full)
   {
      aRes += ToStr(mNbMin,2) + aSep;
      if (aFormat && (!FormatDone)) 
      {
         *aFormat= "mm:ss";
         FormatDone= true;
      }
   }


   aRes += ToStr(mNbSec,2) + ".";
   aRes += ToStr(round_down(mFrac*pow(10,aNbDigFrac)),aNbDigFrac);

   if (aFormat && (!FormatDone)) 
   {
      *aFormat= "ss";
      FormatDone= true;
   }

   return aRes;
}

void cMMVII_Duration::Normalise(eTyUnitTime UnitUsed)
{
     tINT8  aNbSecSup =  lround_down (mFrac);
     mNbSec += aNbSecSup;
     mFrac  -=  aNbSecSup;

     if (UnitUsed >= eTyUnitTime::eUT_Min)
     {
        tINT8 aNbMinSup = mNbSec/60;
        mNbMin += aNbMinSup;
        mNbSec -= aNbMinSup*60;
     }

     if (UnitUsed >= eTyUnitTime::eUT_Hour)
     {
         tINT8 aNbHourSup = mNbMin/60;
         mNbHour += aNbHourSup;
         mNbMin  -= aNbHourSup * 60;
     }

     if (UnitUsed >= eTyUnitTime::eUT_Day)
     {
        tINT8 aNbDaySup = mNbHour /24;
        mNbDay += aNbDaySup;
        mNbHour -= aNbDaySup * 24;
     }
}

cMMVII_Duration::cMMVII_Duration() :
   mNbDay  (0),
   mNbHour (0),
   mNbMin  (0),
   mNbSec  (0),
   mFrac   (0)
{
}

cMMVII_Duration cMMVII_Duration::FromSecond(double aNbSec,eTyUnitTime aTUT)
{
   cMMVII_Duration aRes;
   aRes.mFrac  = aNbSec;
   aRes.Normalise(aTUT);

   return aRes;
}

void Bench_Duration_Daisy(double aT,const std::string & aStr)
{
    std::string aDS = cMMVII_Duration::FromSecond(aT).ToDaisyStr();
    if (aDS != aStr)
    {
        StdOut() << aT << " " << aDS << " " << aStr << "\n";
        MMVII_INTERNAL_ASSERT_bench(false,"Bench_Duration_Daisy");
    }
}


class cBenchTimeCopyVect
{
   public :
      cBenchTimeCopyVect(int aNbVect,int aSzVect) :
         mNbV (aNbVect),
         mSzV (aSzVect),
         mVV ( aNbVect,std::vector(aSzVect,0.0)),
         mV  ( aSzVect,1.0)
      {
      }
      void CPByEl()
      {
           for (size_t aKV=0 ; aKV< mNbV ; aKV++)
           {
                for (size_t aKEl=0 ; aKEl<mSzV ; aKEl++)
                    mVV[aKV][aKEl] = mV[aKEl];
           }
      }
      void CPByVect()
      {
           for (size_t aKV=0 ; aKV< mNbV ; aKV++)
           {
               mVV[aKV] = mV;
           }
      }
      void CPByMemcp()
      {
           size_t aSzV= sizeof(double) * mSzV;
           for (size_t aKV=0 ; aKV< mNbV ; aKV++)
           {
               memcpy(mVV[aKV].data(),mV.data(),aSzV);
           }
      }
   private :
       size_t mNbV;
       size_t mSzV;
       std::vector<std::vector<double>>  mVV;
       std::vector<double>               mV;
};

void OneBenchTimeVect(int aNbVect,int aSzVect)
{
    int aNbOp = 1e8;
    int aNbTimes = aNbOp /(aNbVect*aSzVect);

    cBenchTimeCopyVect aBench(aNbVect,aSzVect);

    double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0();
    for (int aK=0 ;aK<aNbTimes ; aK++)
    {
        aBench.CPByEl();
     }

    double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
    for (int aK=0 ;aK<aNbTimes ; aK++)
        aBench.CPByVect();

    double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();
    for (int aK=0 ;aK<aNbTimes ; aK++)
        aBench.CPByMemcp();
    double aT3 = cMMVII_Appli::CurrentAppli().SecFromT0();


    StdOut()  <<  "NbV=" <<aNbVect << " NbEl=" << aSzVect << "\n";
    StdOut()  <<  "El=" << aT1-aT0 << " Vec=" << aT2-aT1  << " Mcp=" << aT3-aT2 << "\n\n";
}

void BenchTimeVect()
{
    OneBenchTimeVect(30,2);
    OneBenchTimeVect(10,10);
    OneBenchTimeVect(100,100);
   // int aNbOp = 1e;
   // int aSzV = 
}

void Bench_Duration(cParamExeBench & aParam)
{
   if (! aParam.NewBench("Duration")) return;

   if (aParam.Show())
      BenchTimeVect();

   Bench_Duration_Daisy(0.0101,"00.01");
   Bench_Duration_Daisy(1.0101,"01.01");
   Bench_Duration_Daisy(10.0101,"10.01");
   Bench_Duration_Daisy(100.0101,"01:40.01");
   // Bench_Duration_Daisy(100.031,"01:40.01"); => Bench le bench ;-)

   Bench_Duration_Daisy(3700.0101,"01:01:40.01");
   Bench_Duration_Daisy(24*3600 + 2 *3600 + 17*60 + 32.0401,"01:02:17:32.04");

   aParam.EndBench();
}




};

