#include "include/MMVII_all.h"
#include "include/MMVII_Derivatives.h"


namespace MMVII
{



/* ************************************** */
/*                                        */
/*         cTestJets<Type,Nb>             */
/*                                        */
/* ************************************** */


// class cProjCamRad<double>;
// class cProjCamRad<cEpsNum<6> >;


template <class Type,int Nb>  class cTestJets
{
    public :
       Type   ComputeExpansed(Type * Parameter) const;
       Type   ComputeLoop(Type * Parameter) const;
       double  Phase[Nb];
       double  Freq[Nb];

       cTestJets() ;
};


#define TJ(K) (COS(Freq[K]*Square(Param[K]+0.2) + Phase[K]))
///  TJ(K) (COS(Freq[K]*Square(Param[K]+0.2) + Phase[K]))
template <int Nb>  cEpsNum<Nb> TJ_DerAnalytique(const cTestJets<double,Nb> & aTJ,double * aParam)
{
    static double Tmp[Nb];
    double Som = 0.0;
    const double* Phase = aTJ.Phase;
    const double* Freq  = aTJ.Freq;
    for (int aK=0 ; aK<Nb ; aK++)
    {
       Tmp[aK] = Freq[aK]*Square(aParam[aK]+0.2)+Phase[aK];
       if (aK==2)
          Som += cos(Tmp[aK]);
       else
          Som -= cos(Tmp[aK]);
    }
    Som = 1/Som;
    double V2 = Square(Som);
    Eigen::Matrix<double, 1, Nb> mEps;

    for (int aK=0 ; aK<Nb ; aK++)
    {
       // double aP2= aParam[aK]+0.2;
       mEps[aK] = 2*(aParam[aK]+0.2)*Freq[aK] *sin(Tmp[aK]) *V2;
       if (aK==2)
          mEps[aK] *= -1;
    }

    return cEpsNum<Nb>(Som,mEps);
}





template <class Type,int Nb> cTestJets<Type,Nb>::cTestJets() 
{
   for (int aK=0 ; aK<Nb ; aK++)
   {
      Phase[aK]  = aK/2.7 +  1.0 /(1 + aK);
      Freq[aK]  = 1 + aK;
   }
}



template <class Type,int Nb> Type cTestJets<Type,Nb>::ComputeExpansed(Type * Param) const
{
    if (Nb==2)
        return   Type(1)/ (TJ(0)+TJ(1));
    if (Nb==6)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5));
    if (Nb==12)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5)+TJ(6)+TJ(7)+TJ(8)+TJ(9)+TJ(10)+TJ(11));

    if (Nb==18)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5)+TJ(6)+TJ(7)+TJ(8)+TJ(9)+TJ(10)+TJ(11)+TJ(12)+TJ(13)+TJ(14)+TJ(15)+TJ(16)+TJ(17));


    if (Nb==24)
        return   Type(1)/ (TJ(0)+TJ(1)-TJ(2)+TJ(3)+TJ(4)+TJ(5)+TJ(6)+TJ(7)+TJ(8)+TJ(9)+TJ(10)+TJ(11)+TJ(12)+TJ(13)+TJ(14)+TJ(15)+TJ(16)+TJ(17)+TJ(18)+TJ(19)+TJ(20)+TJ(21)+TJ(22)+TJ(23));

/*
    MMVII_INTERNAL_ASSERT_always(false,"COMMPUUTE");
    return   Type(1)/ TJ(0);
*/
    return ComputeLoop(Param);
}

template <class Type,int Nb> Type cTestJets<Type,Nb>::ComputeLoop(Type * Param) const
{
    Type aRes = TJ(0);
    for (int aK=1 ; aK<Nb ; aK++)
    {
        if (aK==2)
           aRes = aRes -TJ(aK);
        else 
           aRes = aRes +TJ(aK);
    }
    return Type(1.0) / aRes;
}


extern bool NEVER;

template <int Nb>  void TplTestJet(double aTiny)
{
   cEpsNum<Nb>  TabJets[Nb];
   double      TabDouble[Nb];
   // On initialise le Jet et les double a la meme valeur
   for (int aK=0 ; aK<Nb ; aK++)
   {
       TabDouble[aK] = tan(aK);
       TabJets[aK] = cEpsNum<Nb>(TabDouble[aK],aK);
   }
   cTestJets<cEpsNum<Nb>,Nb > aTestJet;
   cTestJets<double,Nb >      aTestDouble;
   
   cEpsNum<Nb> aDerAn =  TJ_DerAnalytique(aTestDouble,TabDouble);
   cEpsNum<Nb>  aJetDer = aTestJet.ComputeExpansed(TabJets);
   cEpsNum<Nb>  aJetLoop = aTestJet.ComputeLoop(TabJets);
   for (int aKv=0 ; aKv<Nb ; aKv++)
   {
       double aDerJ = aJetDer.mEps[aKv];
       double  aTdPlus[Nb];
       double  aTdMoins[Nb];
       for (int aKd=0 ; aKd<Nb ; aKd++)
       {
            double aDif   = (aKd==aKv) ? aTiny : 0.0;
            aTdPlus[aKd]  = TabDouble[aKd] + aDif;
            aTdMoins[aKd] = TabDouble[aKd] - aDif;
       }
       double aVPlus =  aTestDouble.ComputeExpansed(aTdPlus);
       double aVMoins = aTestDouble.ComputeExpansed(aTdMoins);
       double aDerNum = (aVPlus-aVMoins) / (2.0 * aTiny);
       double aDif1 = RelativeDifference(aDerJ,aDerNum,nullptr);
       double aDif2 = RelativeDifference(aDerJ,aDerAn.mEps[aKv],nullptr);
       double aDif3 = RelativeDifference(aDerJ,aJetLoop.mEps[aKv],nullptr);
       // StdOut() << "Der; Jet/num=" << aDif1 << " Jet/An=" << aDif2 << " Jet/Loop=" << aDif3 << "\n";
       //  StdOut() << "Der; Jet/num=" << aDerJ << " Jet/An=" << aDerAn.mEps[aKv]  << "\n";
       if (Nb<100)
       {
           MMVII_INTERNAL_ASSERT_bench(aDif1<1e-3,"COMMPUUTE");
       }
// StdOut()  << "__22222: "  << aDif2 << "\n";
       MMVII_INTERNAL_ASSERT_bench(aDif2<1e-10,"COMMPUUTE");
       MMVII_INTERNAL_ASSERT_bench(aDif3<1e-10,"COMMPUUTE");
   }

   for (int aK=0 ; aK< 3; aK++)
   {
      int Time=int(1e6/Nb);
      double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0();
      for (int aK=0 ; aK<Time; aK++)
      {
          cEpsNum<Nb> aDerJet = aTestJet.ComputeExpansed(TabJets);
          if (NEVER)
             StdOut() << aDerJet.mNum << "\n";
      }
      double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
      for (int aK=0 ; aK<Time; aK++)
      {
          cEpsNum<Nb> aDerAn =  TJ_DerAnalytique(aTestDouble,TabDouble);
          IgnoreUnused(aDerAn);
          if (NEVER)
             StdOut() << aDerAn.mNum << "\n";
      }
      double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();

      for (int aK=0 ; aK<Time; aK++)
      {
          cEpsNum<Nb> aDerLoop = aTestJet.ComputeLoop(TabJets);
          if (NEVER)
             StdOut() << aDerLoop.mNum << "\n";
      }
      double aT3 = cMMVII_Appli::CurrentAppli().SecFromT0();
      
      StdOut() << "TIME; Ratio Jet/An=" << (aT1-aT0) / (aT2-aT1) 
               << " Jets= " << aT1-aT0 
               << " An=" << aT2-aT1 
               << " Loop=" << aT3-aT2 
               << "\n";
   }
   StdOut() << "============== Nb=" << Nb << "\n";
}


template <const int Nb> void  TplBenchDifJets()
{
    {
       static int aCpt=0,aNbTot=0,aNbNot0=0; aCpt++;

       cEpsNum<Nb> anEps1 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps1 = anEps1.ToVEN();
       double aDif1 = EpsDifference(anEps1,aVEps1);
       aNbTot+= Nb;
       aNbNot0 +=  aVEps1.mVInd.size();

       if (aCpt>100)
       {
           double aProp  = aNbNot0/double(aNbTot);
           // Verif que les  proportion sont respectees pour que test soit probants
           MMVII_INTERNAL_ASSERT_bench((aProp>0.25) && (aProp<0.75),"TplBenchDifJets");
       }
       // Verif conversion, 
       MMVII_INTERNAL_ASSERT_bench(aDif1<1e-5,"TplBenchDifJets");


       //  operation algebrique
       cEpsNum<Nb> anEps2 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps2 = anEps2.ToVEN();

       cEpsNum<Nb> anEps3 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps3 = anEps3.ToVEN();

       cEpsNum<Nb> anEps4 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps4 = anEps4.ToVEN();

       cEpsNum<Nb> anEps5 = cEpsNum<Nb>::Random(RandUnif_0_1());
       cVarEpsNum  aVEps5 = anEps5.ToVEN();

       //
       cEpsNum<Nb> anEpsCmp = 2.7+ 1.2*anEps1*1.3 + anEps2*COS(anEps3)+3.14 -(anEps4-1)/anEps5;
       cVarEpsNum  aVEpsCmp = 2.7+ 1.3*aVEps1*1.2 + aVEps2*COS(aVEps3)+3.14 -(aVEps4-1)/aVEps5;

       double 	aDifCmp = EpsDifference(anEpsCmp,aVEpsCmp);
       MMVII_INTERNAL_ASSERT_bench(aDifCmp<1e-5,"TplBenchDifJets");
    }
    double aVNum =   RandUnif_C();
    cVarEpsNum  aVE(aVNum);
    cEpsNum<Nb> aE(aVNum);
    int aNbCoeff = RandUnif_N(Nb);
    for (int aTime=0 ; aTime<aNbCoeff ; aTime++)
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
    MMVII_INTERNAL_ASSERT_bench(EpsDifference(aE,aVE)<1e-5,"TplBenchDifJets");

    double aTheoDif = 0.0;
    double aV0 = RandUnif_C();
    aTheoDif += std::abs(aV0);
    aVE.AddVal(aV0,Nb+2);
    double aD0 = EpsDifference(aE,aVE);
    MMVII_INTERNAL_ASSERT_bench(std::abs(aD0-aTheoDif)<1e-5,"TplBenchDifJets");


    double aV1 = RandUnif_C();
    aE.mEps[1] += aV1;
    aTheoDif += std::abs(aV1);
    double aD1 = EpsDifference(aE,aVE);
    MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aTheoDif)<1e-5,"TplBenchDifJets");

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



void BenchMyJets()
{

   BenchJetsCam();

   for (int aK=0 ; aK<1000 ; aK++)
   {
      TplBenchDifJets<10>();
      TplBenchDifJets<60>();
   }
   

   //=====

   TplTestJet<2>(1e-5);
   TplTestJet<6>(1e-5);
   TplTestJet<12>(1e-5);
   TplTestJet<18>(1e-5);
   TplTestJet<24>(1e-5);
   TplTestJet<48>(1e-5);
   TplTestJet<96>(1e-5);
   TplTestJet<192>(1e-5);
   TplTestJet<484>(1e-5);

   // Test that we can compile with, for example, points on jets
   cEpsNum<20>  aEps(3.0,1);
   cPtxd<cEpsNum<20>,3> aP1(aEps,aEps,aEps);
   cPtxd<cEpsNum<20>,3> aP2;
   cPtxd<cEpsNum<20>,3> aP3 = aP1+aP2;
   
   IgnoreUnused(aP3);

}
bool NEVER=false;


};
