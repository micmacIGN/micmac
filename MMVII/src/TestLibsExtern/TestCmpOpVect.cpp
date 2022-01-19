#include "include/MMVII_all.h"
#include "ExternalInclude/Eigen/Dense"

#if (MMVII_WITH_CERES)
#include "ceres/jet.h"
using ceres::Jet;
namespace MMVII
{
template <class Type,const int SzTEigen> 
    void CeresTestCreate(const int aNb)
{
    Jet<Type, SzTEigen> aJet;
    for (int aK=0 ; aK<aNb ; aK++)
    {
        aJet = aJet + aJet -Type(10.0);
        aJet = (aJet + Type(10.0))/Type(2.0);
    }
    DoNothingWithIt(&aJet); // To avoid too clever compiler supress the loop 
}
}

#else
namespace MMVII
{
template <class Type,const int SzTEigen> 
    void CeresTestCreate(const int aNb)
{
}
}
#endif

namespace MMVII
{

template <class Type,const int SzTEigen> class cTestOperationVector
{
   public :
      typedef  float tTEigen;
      typedef  Eigen::Array<tTEigen,1,Eigen::Dynamic>  tEigenSubArray;
      typedef  Eigen::Map<tEigenSubArray > tEigenWrap;

      static void  DoIt();
};


template <class Type,const int SzTEigen> void cTestOperationVector<Type,SzTEigen>::DoIt()
{
        Eigen::Array<tTEigen, 1, SzTEigen>  aAFix ; // = Eigen::Array<tTEigen, 1, SzTEigen>::Random();

        Eigen::Array<tTEigen,1,Eigen::Dynamic>                aADyn(SzTEigen);
        Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn1(1,1);
        Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn2(1,SzTEigen);

         
        int aNb=  1e7 * (90.0/SzTEigen) ;
        double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0();

        // Eigen vecteur fix
        for (int aK=0 ; aK<aNb ; aK++)
        {
             aAFix = aAFix + aAFix -10;
             aAFix = (aAFix + 10)/2;
        }
        DoNothingWithIt(&aAFix); // To avoid too clever compiler supress the loop 
        double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();

        // Eigen vecteur dyn
        for (int aK=0 ; aK<aNb ; aK++)
        {
             aADyn = aADyn + aADyn -10;
             aADyn = (aADyn + 10)/2;
        }
        double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();

        if (0)  // Eigen vecteur 1, quite long
        {
           for (int aK=0 ; aK<aNb*SzTEigen ; aK++)
           {
               aADyn1 = aADyn1 + aADyn1 -10;
               aADyn1 = (aADyn1 + 10)/2;
           }
        }
        double aT3 = cMMVII_Appli::CurrentAppli().SecFromT0();

        // Eigen using a bloc vector
        Eigen::Array<tTEigen,1,Eigen::Dynamic>   aBloc = aADyn.head(SzTEigen-1);
        aBloc(0,0) = aADyn(0,0) + 1;
        
        for (int aK=0 ; aK<aNb ; aK++)
        {
             if (aK==0)
             {
                  std::cout << "AAAAADr  " << &(aBloc(0,0)) - &(aADyn(0,0)) << "\n";
                  std::cout << "AAAAADr  " << aBloc(0,0)   << " " << aADyn(0,0) << "\n";
             }
             // aBloc = aBloc + aBloc -10;
             // aBloc = (aBloc + 10)/2;
             aADyn.head(SzTEigen-1) =  aADyn.head(SzTEigen-1) +  aADyn.head(SzTEigen-1) -10;
             aADyn.head(SzTEigen-1) = ( aADyn.head(SzTEigen-1) + 10)/2;
        }
        double aT4 = cMMVII_Appli::CurrentAppli().SecFromT0();

        // Eigen elem by elem
        for (int aK=0 ; aK<aNb ; aK++)
        {
            for (int aX=0 ; aX<SzTEigen ; aX++)
            {
                aADyn2(aX) = aADyn2(aX) + aADyn2(aX) -10;
                aADyn2(aX) = (aADyn2(aX) + 10)/2;
            }
        }
        double aT5 = cMMVII_Appli::CurrentAppli().SecFromT0();

        // Raw data
        for (int aK=0 ; aK<aNb ; aK++)
        {
            tTEigen * aData = &  aADyn(0) ;
            for (int aX=0 ; aX<SzTEigen ; aX++)
            {
                aData[aX] =  aData[aX] + aData[aX] -10;
                aData[aX] = (aData[aX] + 10)/2;
            }
        }
        double aT6 = cMMVII_Appli::CurrentAppli().SecFromT0();

        for (int aK=0 ; aK<aNb ; aK++)
        {
             tEigenWrap aWrap(&aADyn(0),1,SzTEigen-1);
             // aWrap += aWrap ;
             // aWrap += 10;
             aWrap = aWrap + aWrap -10;
             aWrap = (aWrap + 10)/2;
        }
        double aT7 = cMMVII_Appli::CurrentAppli().SecFromT0();

        // Raw data
        for (int aK=0 ; aK<aNb ; aK++)
        {
            tTEigen * aData = &  aADyn(0) ;
            tTEigen * aEnd = aData + SzTEigen;
            while (aData != aEnd)
            {
                *aData =  *aData + *aData -10;
                *aData = (*aData + 10)/2;
                aData++;
            }
        }
        double aT8 = cMMVII_Appli::CurrentAppli().SecFromT0();


	CeresTestCreate<Type,SzTEigen>(aNb);
        double aT9 = cMMVII_Appli::CurrentAppli().SecFromT0();


        std::cout << " T01-EigenFix " << aT1-aT0 << " T12-EigenDyn " << aT2-aT1 
                  << " T23 " << aT3-aT2 << " T34-EigenBloc " << aT4-aT3  << "\n"
                  << " T45-EigenElem " << aT5-aT4 << " T56_RawData " << aT6-aT5 
                  << " T67-EigenWrap " << aT7-aT6   << " T78-RawRaw " << aT8-aT7
                  << " T89-Jets " << aT9-aT8   
                  << "\n";
        std::cout << "FIXSZ " << aAFix.rows() << " C:" <<  aAFix.cols() << "\n";
        std::cout << "DYNSZ " << aADyn.rows() << " C:" <<  aADyn.cols() << "\n";
}


void   BenchCmpOpVect(cParamExeBench & aParam)
{
    if (! aParam.NewBench("CompareVectOperator",true)) return;

   // Run TestRatkoswky with static obsevation an inital guess 
    cTestOperationVector<float,90>::DoIt();
    cTestOperationVector<float,128>::DoIt();
    cTestOperationVector<double,128>::DoIt();

    aParam.EndBench();
}

};// namespace MMVII


