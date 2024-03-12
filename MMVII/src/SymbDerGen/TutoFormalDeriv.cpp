#include "SymbDer/SymbolicDerivatives.h"
#include "cMMVII_Appli.h"



/** \file TutoFormalDeriv.cpp

    This file contains an illustration of the formal derivative. It is not specifically targeted for
    practicle use, but to understand a bit, what is inside.


    We have the function  F_ab(x,y) =  cos (ax+b) ^2 - 42*y

    We generate the formula corresponding ,

*/

namespace MMVII
{

namespace  SD = NS_SymbolicDerivative;
using namespace SD;

template <class Type>
std::vector<Type> FitCube
                  (
                      const std::vector<Type> & aVUk,
                      const std::vector<Type> & aVObs
                  )
{
    const Type & x = aVUk[0];
    const Type & y = aVUk[1];

    const Type & a  = aVObs[0];
    const Type & b  = aVObs[1];
    Type  aCst42 = CreateCste(42.0,a);

    return {cos(square(a+b *x))- y*aCst42};

    //  return {(a+b *x)*(x*b+a)*(a+b *x) - y};
}

void  MyShowTreeFormulaRec(const cFormula<tREAL8>& aF,int aLevel)
{
     // To implemant   we want for example that x+(a*b) write something like
     //
     //     +
     //         x
     //         *
     //             a
     //             b
     //   We will use two method of class formula  for this :
     //       Name4Print()  : return the "atomic" name as "+ * cos a x 44 ..."
     //       Ref()         : return the vector of "sub-formumla" (size "2" for "+",  size "0" for "x" ...)
     //
     //
     for (int aK=0 ; aK<aLevel; aK++)
     {
         StdOut() << "   ";
     }
     StdOut() << aF->Name4Print() << std::endl;
     for (const auto & aChild : aF->Ref())
     {
         MyShowTreeFormulaRec(aChild,1+aLevel);
     }
}

void  MyShowTreeFormula(const cFormula<tREAL8>& aF) 
{
   // StdOut()  << aF->Name() << std::endl;
   // StdOut()  << aF->GenCodeFormName() << std::endl;
   // StdOut()  << aF->GenCodeExpr() << std::endl;
   StdOut()  << aF->Name4Print() << std::endl;
   StdOut()  << aF->Ref().size() << std::endl;
   MyShowTreeFormulaRec(aF,0);
}

void TestCubeFormula(bool OrderRand,bool GenCode)
{
    std::cout <<  "===================== TestFoncCube  ===================\n";

    //
    //   A  coordinator is the object that contain all the information necessary
    //   to process a formula and its derivative, its main role is to maintain a map
    //   of all existing  to create an efficient DAG 
    //
    //   For creating a coorinator  we essentially give a vector of unknwon and observations;
    //   the key-point of these vector is their size, if we change the name, it will give
    //   the same code (dummy variable) as long as they generate valide code (dont call them "0-@" for example ...)
    //
    SD::cCoordinatorF<double>  aCFD("FitCube",100,{"x","y"},{"a","b"});

    // Inspect vector of unknown and vector of observations
    {
        for (const auto & aF : aCFD.VUk())
           std::cout << "Unknowns : "<< aF->Name() << "\n";
        for (const auto & aF : aCFD.VObs())
           std::cout << "Observation : "<< aF->Name() << "\n";
    }
  //  getchar();

    // Create the formula corresponding to residual
    std::vector<SD::cFormula<double>>  aVResidu = FitCube(aCFD.VUk(),aCFD.VObs());
    SD::cFormula<double>  aResidu = aVResidu[0];

    MyShowTreeFormula(aResidu);

//    MyShowTreeFormula(aResidu->Derivate(0));
//    MyShowTreeFormula(aResidu->Derivate(0)->Derivate(0));  // => will generate unreachable code

aResidu->Derivate(0)->Derivate(0);

   // Inspect the formula
    //std::cout  << "RESIDU FORMULA, Num=" << aResidu->NumGlob() << " Name=" <<  aResidu->Name() <<"\n";
    //std::cout  << " PP=[" << aResidu->InfixPPrint() <<"]\n";
    //getchar();

    // Inspect the derivative  relatively to b
   // auto aDerx = aResidu->Derivate(0);
   // std::cout  << "DERIVATE FORMULA , Num=" << aDerx->NumGlob() << " Name=" <<  aDerx->Name() <<"\n";
   // std::cout  << " PP=[" << aDerx->InfixPPrint() <<"]\n";
   // getchar();


    // Set the formula that will be computed
    aCFD.SetCurFormulasWithDerivative(aVResidu);

    // Print stack of formula
     // std::cout << "====== Stack === \n";
    aCFD.ShowStackFunc();
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


void InspectCube() { TestCubeFormula(false,false); }

};
