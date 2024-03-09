#include "SymbDer/SymbolicDerivatives.h"
#include "cMMVII_Appli.h"



/** \file TutoFormalDeriv.cpp

    This file contains an illustration of the formal derivative. It is not specifically targeted for
    practicle use, but to understand a bit, what is inside.


    We have the function  F_ab(x,y) =  cos (ax+b) ^2 -y

    We generate the formula corresponding ,

*/

namespace MMVII
{

namespace  SD = NS_SymbolicDerivative;

template <class Type>
std::vector<Type> FitCube
                  (
                      const std::vector<Type> & aVUk,
                      const std::vector<Type> & aVObs,
		      bool OrderRand
                  )
{
    const Type & x = aVUk[0];
    const Type & y = aVUk[1];

    const Type & a  = aVObs[0];
    const Type & b  = aVObs[1];

    return {cos(square(a+b *x))- y};

    //  return {(a+b *x)*(x*b+a)*(a+b *x) - y};
}

void TestCubeFormula(bool OrderRand,bool GenCode)
{
    std::cout <<  "===================== TestFoncCube  ===================\n";

    // Create a context where values are stored on double and :
    //    2 unknown, 2 observations, a buffer of size 100
    //    aCFD(100,2,2) would have the same effect for the computation
    //    The variant with vector of string, will fix the name of variables, it
    //    will be usefull when will generate code and will want  to analyse it
    SD::cCoordinatorF<double>  aCFD("FitCube",100,{"x","y"},{"a","b"});

    // Inspect vector of unknown and vector of observations
    {
        for (const auto & aF : aCFD.VUk())
           std::cout << "Unknowns : "<< aF->Name() << "\n";
        for (const auto & aF : aCFD.VObs())
           std::cout << "Observation : "<< aF->Name() << "\n";
    }

    // Create the formula corresponding to residual
    std::vector<SD::cFormula<double>>  aVResidu = FitCube(aCFD.VUk(),aCFD.VObs(),OrderRand);
    SD::cFormula<double>  aResidu = aVResidu[0];

   // Inspect the formula
    std::cout  << "RESIDU FORMULA, Num=" << aResidu->NumGlob() << " Name=" <<  aResidu->Name() <<"\n";
    std::cout  << " PP=[" << aResidu->InfixPPrint() <<"]\n";

    // Inspect the derivative  relatively to b
    auto aDerb = aResidu->Derivate(0);
    std::cout  << "DERIVATE FORMULA , Num=" << aDerb->NumGlob() << " Name=" <<  aDerb->Name() <<"\n";
    std::cout  << " PP=[" << aDerb->InfixPPrint() <<"]\n";

    // Set the formula that will be computed
    aCFD.SetCurFormulasWithDerivative(aVResidu);

    // Print stack of formula
    std::cout << "====== Stack === \n";
    aCFD.ShowStackFunc();
    // All the formula generated are not required (reached) for example
    // ---0- F12_ => F8_*F5_   = (F7_+F3_) * x =  (F4_*F5_+a) * x = (b*x_+a) * x
    //
    //
    //  PP= 2x  *  (a+bx)^2 +  (a+bx)^2 * x
    if (GenCode)
    {
       aCFD.GenerateCode("TestDer_");
    }
}


/* *********************************************************** */
/*                                                             */
/*                cAppliTutoFormalDeriv                        */
/*                                                             */
/* *********************************************************** */

class cAppliTutoFormalDeriv : public cMMVII_Appli
{
     public :

        cAppliTutoFormalDeriv(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
};


cAppliTutoFormalDeriv::cAppliTutoFormalDeriv
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli   (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliTutoFormalDeriv::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
   ;
}


cCollecSpecArg2007 & cAppliTutoFormalDeriv::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
                  anArgOpt
          ;
}



int  cAppliTutoFormalDeriv::Exe()
{
    // Inspect vector of unknown and vector of observations


    TestCubeFormula(true,true);

    

   return EXIT_SUCCESS;
}

/* *********************************************************** */
/*                                                             */
/*                           ::                                */
/*                                                             */
/* *********************************************************** */


tMMVII_UnikPApli Alloc_cAppliTutoFormalDeriv(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliTutoFormalDeriv(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TutoFormalDeriv
(
     "TutoFormalDeriv",
      Alloc_cAppliTutoFormalDeriv,
      "Tutorial for serialization",
      {eApF::Project,eApF::Test},
      {eApDT::None},
      {eApDT::Xml},
      __FILE__
);


/*
---0- F0_ => _C0 ; Val=0
---0- F1_ => _C1 ; Val=1
---0- F2_ => _C2 ; Val=2
-0-1- F3_ => a
-0-1- F4_ => b
-0-4- F5_ => x
-0-1- F6_ => y
-1-1- F7_ => F4_*F5_
-2-9- F8_ => F7_+F3_
-3-2- F9_ => F8_*F8_
-4-1- F10_ => F8_*F9_
-5-1- F11_ => F10_-F6_
---0- F12_ => F8_*F5_   = (F7_+F3_) * x =  (F4_*F5_+a) * x = (b*x_+a) * x
-1-1- F13_ => F5_+F5_
-3-1- F14_ => F13_*F8_
---0- F15_ => F2_*F5_
-4-1- F16_ => F14_*F8_
-4-1- F17_ => F9_*F5_
-5-1- F18_ => F16_+F17_
-3-1- F19_ => F8_+F8_
---0- F20_ => F8_*F19_
---0- F21_ => F7_*F2_
---0- F22_ => F2_*F3_
---0- F23_ => F8_*F2_
-4-1- F24_ => F8_+F19_
-5-1- F25_ => F8_*F24_
---0- F26_ => _C3 ; Val=3
---0- F27_ => F26_*F3_
---0- F28_ => F7_+F21_
---0- F29_ => F27_+F28_
 */



void InspectCube() { TestCubeFormula(false,false); }

};
