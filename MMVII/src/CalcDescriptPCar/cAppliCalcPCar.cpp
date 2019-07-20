#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

template <class Type> class  cTplAppliCalcDescPCar;
class cAppliCalcDescPCar;

///  Class to manage template INT/REAL in cMMVII_Appli
/**
    To avoid code duplication, only one cAppliCalcDescPCar exist,
    code that require specialisation is done in cTplAppliCalcDescPCar
*/

template <class Type> class  cTplAppliCalcDescPCar
{
    public :
        typedef cGaussianPyramid<Type> tPyr;
        typedef std::shared_ptr<tPyr>  tSP_Pyr;

        cTplAppliCalcDescPCar(cAppliCalcDescPCar & anAppli);
        void Exe();
    private :
        cAppliCalcDescPCar & mAppli;
        tSP_Pyr              mPyr;
};


///  Application class for computation of Aime Descritor
/** 
*/

class cAppliCalcDescPCar : public cMMVII_Appli
{
     public :
        friend  class cTplAppliCalcDescPCar<tREAL4>;
        friend  class cTplAppliCalcDescPCar<tINT2>;

        cAppliCalcDescPCar(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        std::string mNameIm;  ///< Input image pattern/name
        std::vector<std::string>  mVSetIm;  ///< Vector of image resulting from Pattern
        bool        mIntPyram;  ///< Impose gray image
};

cCollecSpecArg2007 & cAppliCalcDescPCar::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIm,"Name of input file",{{eTA2007::MPatIm,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliCalcDescPCar::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
             << AOpt2007(mIntPyram,"IntPyr","Gauss Pyramid in integer",{eTA2007::HDV})
   ;
}

int cAppliCalcDescPCar::Exe() 
{
   mVSetIm = VectMainSet(0);

   if (mVSetIm.size() != 1)  // Multiple image, run in parall 
   {
      ExeMultiAutoRecallMMVII("0",mVSetIm);
      return EXIT_SUCCESS;
   }
   // Single image, do the job ...

   if (mIntPyram)  // Case integer pyramids
   {
      cTplAppliCalcDescPCar<tINT2> aTplA(*this);
      aTplA.Exe();
   }
   else   // Case floating points pyramids
   {
      cTplAppliCalcDescPCar<tREAL4> aTplA(*this);
      aTplA.Exe();
   }
   
   return EXIT_SUCCESS;
}

cAppliCalcDescPCar:: cAppliCalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mIntPyram    (false)
{
}

/*  =============================================== */
/*                                                  */
/*             cTplAppliCalcDescPCar<Type>          */
/*                                                  */
/*  =============================================== */

template<class Type> cTplAppliCalcDescPCar<Type>::cTplAppliCalcDescPCar(cAppliCalcDescPCar & anAppli) :
   mAppli (anAppli),
   mPyr   (nullptr)
{
}

template<class Type> void cTplAppliCalcDescPCar<Type>::Exe()
{
   // std::shared_ptr<cGaussianPyramid<Type>>  aGP = cGaussianPyramid<tREAL4>::Alloc(cGP_Params(cPt2di(400,700),5,5,3));
   mPyr = tPyr::Alloc(cGP_Params(cPt2di(400,700),5,5,3));
   mPyr->Show();
}

/*  =============================================== */
/*                                                  */
/*                        ::                        */
/*                                                  */
/*  =============================================== */

tMMVII_UnikPApli Alloc_CalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCalcDescPCar(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCalcDescPCar
(
     "TieP-AimePCar",
      Alloc_CalcDescPCar,
      "Compute caracteristic points and descriptors, using Aime method",
      {eApF::TieP,eApF::ImProc},
      {eApDT::Image},
      {eApDT::TieP},
      __FILE__
);




};
