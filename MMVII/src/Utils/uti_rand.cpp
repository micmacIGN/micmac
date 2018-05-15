#include "include/MMVII_all.h"
#include <random>

/** \file uti_rand.cpp
    \brief Implementation of random generator

    Use C++11, implement a very basic interface,
    will evolve probably when object manipulation
    will be needed for more sophisticated services.

*/


namespace MMVII
{


void Bench_Random()
{
   std::cout << "Begin Bench_Random\n";
   {
      int aNb = 1e6;
      std::vector<double> aVR;
      for (int aK=0 ; aK< aNb ; aK++)
          aVR.push_back(RandUnif_0_1());

      std::sort(aVR.begin(),aVR.end());

      double aDistMoy=0;
      double aDistMax=0;
      double aCorrel = 0;
      double aCorrel10 = 0;
      for (int aK=0 ; aK< aNb ; aK++)
      {
          double aD = std::abs(aVR[aK] - aK/double(aNb));
          aDistMoy += aD;
          aDistMax = std::max(aD,aDistMax);
          if (aK!=0)
             aCorrel += (aVR[aK]-0.5) * (aVR[aK-1]-0.5);
          if (aK>=10)
             aCorrel10 += (aVR[aK]-0.5) * (aVR[aK-10]-0.5);
      }
      aDistMoy /= aNb;
      aCorrel /= aNb-1;
      aCorrel10 /= aNb-10;
      // Purely heuristique bound, on very unlikely day may fail
      MMVII_INTERNAL_ASSERT_bench(aDistMoy<1.0/sqrt(aNb),"Random Moy Test");
      MMVII_INTERNAL_ASSERT_bench(aDistMax<4.0/sqrt(aNb),"Random Moy Test");

      // => Apparently correlation is very high : 0.08 !! maybe change the generator ?
      // std::cout << "Correl Rand " << aCorrel  << " " << aCorrel10 << "\n";
      
   }



   std::cout << "Bench_Random " << RandUnif_0_1() << " " << RandUnif_0_1() << "\n";
}



/// class cRandGenerator maybe exported later if  more sophisticated services are required

class cRandGenerator : public cMemCheck
{
   public :
       virtual double Unif_0_1() = 0;
       virtual int    Unif_N(int aN) = 0;
       static cRandGenerator * TheOne();
       virtual ~cRandGenerator(){};
    private :
       static cRandGenerator * msTheOne;
};

void FreeRandom() 
{
   delete cRandGenerator::TheOne();
}
double RandUnif_0_1()
{
   return cRandGenerator::TheOne()->Unif_0_1();
}
double RandUnif_N(int aN)
{
   return cRandGenerator::TheOne()->Unif_N(aN);
}


/// class cRand19937 concrete implemenation

class cRand19937 : public cRandGenerator
{
     public :
         cRand19937(int aSeed);
         double Unif_0_1() override ;
         int    Unif_N(int aN) override;
         ~cRand19937() {}
     private :
          // std::random_device mRD;                    //Will be used to obtain a seed for the random number engine
          std::mt19937       mGen;                   //Standard mersenne_twister_engine seeded with rd()
          std::uniform_real_distribution<> mDis01;
          std::unique_ptr<std::uniform_int_distribution<> > mDisInt;
          int                                               mLastN;
          //uniform_01<mt19937> mU01;
};


cRand19937::cRand19937(int aSeed) :
    // mRD      (),
    mGen     (aSeed),
    mDis01   (0.0,1.0),
    mDisInt  (nullptr)
{
}

double cRand19937::Unif_0_1()
{
    return mDis01(mGen);
}

int    cRand19937::Unif_N(int aN) 
{
   if ((mDisInt==nullptr) || (mLastN!=aN))
   {
       mLastN = aN;
       mDisInt.reset(new  std::uniform_int_distribution<>(0,aN-1));
   }

   return (*mDisInt)(mGen);
}



cRandGenerator * cRandGenerator::msTheOne = nullptr;

cRandGenerator * cRandGenerator::TheOne()
{
   if (msTheOne==0)
      msTheOne = new cRand19937(cMMVII_Appli::SeedRandom());
   return msTheOne;
}


};

