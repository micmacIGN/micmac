#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_SetITpl.h"
#include <random>

/** \file uti_rand.cpp
    \brief Implementation of random generator

    Use C++11, implement a very basic interface,
    will evolve probably when object manipulation
    will be needed for more sophisticated services.

*/


namespace MMVII
{

void AssertIsSetKN(int aK,int aN,const std::vector<int> &aSet)
{
  MMVII_INTERNAL_ASSERT_bench(int(aSet.size())==aK,"Random Set");
  for (const auto & aE : aSet)
  {
       MMVII_INTERNAL_ASSERT_bench((aE>=0) && (aE<aN),"Random Set");
       int aNbEq=0;
       for (const auto & aE2 : aSet)
       {
           if (aE==aE2)
              aNbEq++;
       }
       MMVII_INTERNAL_ASSERT_bench(aNbEq==1,"Random Set");
  }
}

void OneBench_Random(cParamExeBench & aParam)
{
   // Generate subset at K element among N, check they are that
   for (int aTime=0 ; aTime< 100 ; aTime++)
   {
       int aNb = 10 + aTime/10;
       int aK = 3+aTime/10;
       std::vector<int> aSet =  RandSet(aK,aNb);
       AssertIsSetKN(aK,aNb,aSet);
       for (int aD=1 ; aD<=3 ; aD++)
       {
          aSet = RandNeighSet(aD,aNb,aSet);  // Generate a set at distance D
          AssertIsSetKN(aK,aNb,aSet);
       }
   }
   // StdOut() << "Begin Bench_Random\n";
   {
      int aNb = std::min(3e6,1e6 *(1+pow(aParam.Level(),1.5)));
      std::vector<double> aVInit;
      for (int aK=0 ; aK< aNb ; aK++)
          aVInit.push_back(RandUnif_0_1());

      std::vector<double> aVSorted = aVInit;
      std::sort(aVSorted.begin(),aVSorted.end());

      double aDistMoy=0;
      double aDistMax=0;
      double aCorrel = 0;
      double aCorrel10 = 0;
      for (int aK=0 ; aK< aNb ; aK++)
      {
          // Theoretically VSorted should converd to distrib X (cumul of uniform dist)
          double aD = std::abs(aVSorted[aK] - aK/double(aNb));
          aDistMoy += aD;
          aDistMax = std::max(aD,aDistMax);
          if (aK!=0)
             aCorrel += (aVInit[aK]-0.5) * (aVInit[aK-1]-0.5);
          if (aK>=10)
             aCorrel10 += (aVInit[aK]-0.5) * (aVInit[aK-10]-0.5);
      }
      aDistMoy /= aNb;
      aCorrel /= aNb-1;
      aCorrel10 /= aNb-10;
      // Purely heuristique bound, on very unlikely day may fail
      MMVII_INTERNAL_ASSERT_bench(aDistMoy<2.0/sqrt(aNb),"Random Moy Test");
      MMVII_INTERNAL_ASSERT_bench(aDistMax<8.0/sqrt(aNb),"Random Moy Test");
      MMVII_INTERNAL_ASSERT_bench(std::abs(aCorrel)  <0.5/sqrt(aNb),"Random Correl1 Test");
      MMVII_INTERNAL_ASSERT_bench(std::abs(aCorrel10)<0.5/sqrt(aNb),"Random Correl10 Test");

      // => Apparently correlation is very high : 0.08 !! maybe change the generator ?
      
   }
}

void Bench_Random(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Random")) return;

    int aNb = std::min(5,1+aParam.Level()*2);
    for (int aK=0 ; aK<aNb  ; aK++)
       OneBench_Random(aParam);

    aParam.EndBench();
}

template <typename tSet>  void OneBenchSet()
{
    for (const auto &aK : {0,1,2,5,10})
    {
        int aN = 10;
	std::vector<tSet>  aLSet =  SubKAmongN<tSet>(aK,aN);
        MMVII_INTERNAL_ASSERT_bench(aLSet.size()==iBinomialCoeff(aK,aN),"Subset Int "); // Check good number of subset
	for (const auto & aSet : aLSet)
	{
            MMVII_INTERNAL_ASSERT_bench(aK==(int)aSet.Cardinality(),"Subset Int ");  // Check each subset has good number of elem
	    std::vector<int> aV = aSet.ToVect();
            MMVII_INTERNAL_ASSERT_bench(aK==(int)aV.size(),"Card to Vect");

	    for (const auto & anEl : aV)
	    {
                 MMVII_INTERNAL_ASSERT_bench((anEl>=0)&&(anEl<aN),"Subset Int bad el ");  // Check each subset has good number of elem
	    }
	}
    }
}

void Bench_SetI(cParamExeBench & aParam)
{
    if (! aParam.NewBench("SetInt")) return;

    OneBenchSet<cSetISingleFixed<tU_INT2> >();

    aParam.EndBench();
}


/// class cRandGenerator maybe exported later if  more sophisticated services are required

class cRandGenerator : public cMemCheck
{
   public :
       virtual double Unif_0_1() = 0;
       virtual int    Unif_N(int aN) = 0;
       static cRandGenerator * TheOne();
       static void Close();
       static void Open();
       virtual ~cRandGenerator() {};
    private :
       static cRandGenerator * msTheOne;
};

double RandUnif_0_1()
{
   return cRandGenerator::TheOne()->Unif_0_1();
}

std::vector<double> VRandUnif_0_1(int aNb)
{
    std::vector<double> aRes;
    for (int aK=0 ; aK<aNb ; aK++)
        aRes.push_back(RandUnif_0_1());
    return aRes;
}


double RandUnif_C()
{
   return (RandUnif_0_1()-0.5) * 2.0;
}

double RandInInterval(double a,double b)
{
   return b+ (a-b) * RandUnif_0_1() ;
}


double RandUnif_C_NotNull(double aEps)
{
   double aRes = RandUnif_C();
   while (std::abs(aRes)<aEps)
         aRes = RandUnif_C();
   return aRes;
}
double RandUnif_NotNull(double aEps) {return std::abs(RandUnif_C_NotNull(aEps));}

double RandUnif_N(int aN)
{
   return cRandGenerator::TheOne()->Unif_N(aN);
}

bool HeadOrTail()
{
     return  RandUnif_0_1() > 0.5;
}



cFctrRR cFctrRR::TheOne;
double cFctrRR::F(double) const {return 1.0;}

std::vector<int> RandSet(int aSzSubSet,int aSzSetGlob,cFctrRR & aBias )
{
    MMVII_INTERNAL_ASSERT_strong(aSzSubSet<=aSzSetGlob,"RandSet");
   //  in VP : x->K , y -> priority
    std::vector<cPt2dr> aVP;
    for (int aK=0; aK<aSzSetGlob ; aK++)
       aVP.push_back(cPt2dr(aK,aBias.F(aK)*RandUnif_0_1()));

    // Sort on y()
    std::sort(aVP.begin(),aVP.end(),CmpCoord<double,2,1>);
  
    // Extract the aSzSubSet "best" values
    std::vector<int> aRes;
    for (int aJ=0 ; aJ<aSzSubSet ; aJ++)
       aRes.push_back(round_ni(aVP.at(aJ).x()));
    return aRes;
}

std::vector<int> RandPerm(int aN,cFctrRR & aBias)
{
    return RandSet(aN,aN,aBias);
}

int MaxElem(const std::vector<int> & aSet)
{
   int aRes=-1;
   for (const auto & anElem : aSet)
   {
      aRes = std::max(aRes,anElem);
   }
   return aRes;
}

std::vector<int> ComplemSet(int aN,const std::vector<int> & aSet)
{
   std::vector<bool> aVBelong(std::max(aN,MaxElem(aSet)+1),false);
   for (const auto & anElem : aSet)
   {
      aVBelong.at(anElem) = true;
   }

   std::vector<int>  aRes;
   for (int aK=0 ; aK<int(aVBelong.size()) ; aK++)
      if (!aVBelong.at(aK))
         aRes.push_back(aK);

   return aRes;
}

std::vector<int> RandNeighSet(int aK,int aN,const std::vector<int> & aSet)
{
    std::vector<int> aComp = ComplemSet(aN,aSet);
    std::vector<int> aIndToAdd = RandSet(aK,aComp.size());
    std::vector<int> aIndToSupr = RandSet(aK,aSet.size());

    std::vector<int> aRes = aSet;
    for (int aJ=0 ; aJ<aK ; aJ++)
        aRes.at(aIndToSupr.at(aJ)) = aComp.at(aIndToAdd.at(aJ));

   return aRes;
}

/*  Random or deterministic selectors */

bool SelectWithProp(int aK,double aProp)
{
    double aPH1 =  aK * aProp;
    double aPH2 =  (aK+1) * aProp;
    return  round_ni(aPH1) != round_ni(aPH2);
}

bool SelectQAmongN(int aK,int aQ,int aN)
{
    return SelectWithProp(aK,double(aQ)/double(aN));
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

// This variable allow to check that no random is allocated before the main appli is created
void cRandGenerator::Open()
{
    static bool FirstCall=true;
    MMVII_INTERNAL_ASSERT_bench(FirstCall,"Multiple Open Random");
    msTheOne = new cRand19937(cMMVII_Appli::SeedRandom());
    FirstCall = false;
}
void OpenRandom()
{
    cRandGenerator::Open();
}

// static int aCPT = 0; 

cRandGenerator * cRandGenerator::TheOne()
{
   MMVII_INTERNAL_ASSERT_bench(msTheOne!=nullptr,"Not Open Random");
   return msTheOne;
}

void cRandGenerator::Close()
{
   MMVII_INTERNAL_ASSERT_bench(msTheOne!=nullptr,"Multiple Close Random");
   delete msTheOne;
   msTheOne = nullptr;
}
void CloseRandom()
{
    cRandGenerator::Close();
}

};

