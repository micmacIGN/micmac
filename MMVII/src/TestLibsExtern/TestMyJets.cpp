#include "MMVII_Derivatives.h"
#include "cMMVII_Appli.h"

namespace MMVII
{

/*
class cXMMVII_Ofs : public cMemCheck
{
};
*/



/* ************************************** */
/*                                        */
/*         cTestJets<Type,Nb>             */
/*                                        */
/* ************************************** */

#if (0)

#endif

// Test coherence between cEpsNum and  cVarEpsNum

template <const int Nb> void  TplBenchDifJets()
{
    {
       static int aCpt=0,aNbTot=0,aNbNot0=0; aCpt++;

       // Generate an EpsNum with a random density of non 0
       cEpsNum<Nb> anEps1 = cEpsNum<Nb>::Random(RandUnif_0_1());
       // Convert to var
       cVarEpsNum  aVEps1 = anEps1.ToVEN();
       // Compute differnce
       double aDif1 = EpsDifference(anEps1,aVEps1);
       aNbTot+= Nb;
       aNbNot0 +=  aVEps1.mVInd.size();

       // Need a minimum number before we can compare experimental to theoreticall density
       if (aCpt>100)
       {
           double aProp  = aNbNot0/double(aNbTot);
           // Verif que les  proportion sont respectees pour que test soit probants
           MMVII_INTERNAL_ASSERT_bench((aProp>0.25) && (aProp<0.75),"TplBenchDifJets");
       }
       // Verif conversion, 
       MMVII_INTERNAL_ASSERT_bench(aDif1<1e-5,"TplBenchDifJets");


       // Generate five cEpsNum and their conversion in cVarEpsNum  
       cEpsNum<Nb> anEps2 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps2 = anEps2.ToVEN();

       cEpsNum<Nb> anEps3 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps3 = anEps3.ToVEN();

       cEpsNum<Nb> anEps4 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps4 = anEps4.ToVEN();

       cEpsNum<Nb> anEps5 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps5 = anEps5.ToVEN();

       // Make some calculus
       cEpsNum<Nb> anEpsCmp = 2.7+ 1.2*anEps1*1.3 + anEps2*COS(anEps3)+3.14 -(anEps4-1)/anEps5;
       cVarEpsNum  aVEpsCmp = 2.7+ 1.3*aVEps1*1.2 + aVEps2*COS(aVEps3)+3.14 -(aVEps4-1)/aVEps5;

       // Cheks the values are identical
       double 	aDifCmp = EpsDifference(anEpsCmp,aVEpsCmp);
       MMVII_INTERNAL_ASSERT_bench(aDifCmp<1e-5,"TplBenchDifJets");
    }

    // Construct tow equivalent cVarEpsNum and cEpsNum
    double aVNum =   RandUnif_C(); // Constant "Num" Val
    // Create two numbers with no epsilon 4 Now
    cVarEpsNum  aVE(aVNum);
    cEpsNum<Nb> aE(aVNum);
    int aNbCoeff = RandUnif_N(Nb);
    for (int aTime=0 ; aTime<aNbCoeff ; aTime++)
    {
        // Randomize value and index of derivative
        int aK =  RandUnif_N(Nb);
        double aVal = RandUnif_C();
        aVE.AddVal(aVal,aK);  // Push in Var
        aE.mEps[aK] += aVal;  // Add in fix
        // Push a null value in var which should have no influence
        if (aTime%2==0)
        {
           aVE.mVEps.push_back(0.0);
           aVE.mVInd.push_back(aK+Nb);  // parfois dehors, parfois dedans, pas d'influence si nul
        }
    }
    MMVII_INTERNAL_ASSERT_bench(EpsDifference(aE,aVE)<1e-5,"TplBenchDifJets");

    // Add a Value to VarEps, disjoint from Eps, and modify difference
    double aTheoDif = 0.0;
    double aV0 = RandUnif_C();
    aTheoDif += std::abs(aV0);
    aVE.AddVal(aV0,Nb+2);
    double aD0 = EpsDifference(aE,aVE); // Now they are different but we can predict
    MMVII_INTERNAL_ASSERT_bench(std::abs(aD0-aTheoDif)<1e-5,"TplBenchDifJets");


    // Add a value in Eps and not in VarEps and check the difference
    double aV1 = RandUnif_C();
    aE.mEps[1] += aV1;
    aTheoDif += std::abs(aV1);
    double aD1 = EpsDifference(aE,aVE);
    MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aTheoDif)<1e-5,"TplBenchDifJets");

    // Do same modification wich must not modify difference
    for (int aTime=0 ; aTime<3 ; aTime++)
    {
        int aK =  RandUnif_N(Nb);
        double aVal = RandUnif_C();
        aVE.AddVal(aVal,aK);
        aE.mEps[aK] += aVal;
        if (aTime%2==0)
        {
           aVE.mVEps.push_back(0.0);
           aVE.mVInd.push_back(aK+Nb);  // parfois dehors, parfois dedans, pas d'influence si nul
        }
    }
    MMVII_INTERNAL_ASSERT_bench(std::abs(aTheoDif-EpsDifference(aE,aVE))<1e-5,"TplBenchDifJets");
}



void BenchMyJets(cParamExeBench & aParam)
{
   if (! aParam.NewBench("JetsDerivatives")) return;

   {
       Eigen::Matrix<double, 1, 10> aVec;
       double * aD0 = & aVec(0);
       double * aD1 = & aVec(1);

       MMVII_INTERNAL_ASSERT_bench((aD1-aD0)==1,"Assertion on Eigen memory organization");
       if (0)
       {
          StdOut() << "TEsEigenMEm " << aD0 << " " << aD1 << " " << (aD1-aD0) << "\n";
          getchar();
       }
   }


   for (int aK=0 ; aK<1000 ; aK++)
   {
      TplBenchDifJets<10>();
      TplBenchDifJets<60>();
   }
   

   //=====

/*
   TplTestJet<2>(1e-5);
   TplTestJet<6>(1e-5);
   TplTestJet<12>(1e-5);
   TplTestJet<18>(1e-5);
   TplTestJet<24>(1e-5);
   TplTestJet<48>(1e-5);
   TplTestJet<96>(1e-5);
   TplTestJet<192>(1e-5);
   TplTestJet<484>(1e-5);
*/

   // Test that we can compile with, for example, points on jets
   cEpsNum<20>  aEps(3.0,1);
   cPtxd<cEpsNum<20>,3> aP1(aEps,aEps,aEps);
   cPtxd<cEpsNum<20>,3> aP2;
   cPtxd<cEpsNum<20>,3> aP3 = aP1+aP2;
   
   IgnoreUnused(aP3);
   
   aParam.EndBench();
}
bool NEVER=false;


};
