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

        /// Run all the process
        void ExeGlob();
    private :
        /// Run the process for one image box
        void ExeOneBox(const cPt2di & aP,const cParseBoxInOut<2> & aPBI);

        cAppliCalcDescPCar & mAppli;   ///< Reference the application
        tSP_Pyr              mPyr;     ///< Pointer on gaussian pyramid
        cDataFileIm2D        mDFI;     ///< Structure on image file
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
        std::string               mNameIm;                ///< Input image pattern/name 
        bool        mIntPyram;  ///< Impose integer  image
        int         mSzTile;    ///<  sz of tiles for spliting in sub processe
        int         mOverlap;   ///< sz of  overlap between tiles
};

/* =============================================== */
/*                                                 */
/*            cTplAppliCalcDescPCar<Type>          */
/*                                                 */
/* =============================================== */

template<class Type> cTplAppliCalcDescPCar<Type>::cTplAppliCalcDescPCar(cAppliCalcDescPCar & anAppli) :
   mAppli (anAppli),
   mPyr   (nullptr),
   mDFI   (cDataFileIm2D::Create(mAppli.mNameIm,true))
{
}

template<class Type> void cTplAppliCalcDescPCar<Type>::ExeGlob()
{
   cParseBoxInOut<2> aPBI = cParseBoxInOut<2>::CreateFromSizeCste(cRect2(cPt2di(0,0),mDFI.Sz()),mAppli.mSzTile);

   for (const auto & anIndex : aPBI.BoxIndex())
   {
       ExeOneBox(anIndex,aPBI);
   }


}


template<class Type>  void cTplAppliCalcDescPCar<Type>::ExeOneBox(const cPt2di & anIndex,const cParseBoxInOut<2>& aPBI)
{
    StdOut() << "AnIndex " << anIndex << "\n";
}

   // mPyr = tPyr::Alloc(cGP_Params(cPt2di(400,700),5,5,3));
   // mPyr->Show();

/* =============================================== */
/*                                                 */
/*               cAppliCalcDescPCar                */
/*                                                 */
/* =============================================== */

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
   {
      const std::vector<std::string> &  aVSetIm = VectMainSet(0);

      if (aVSetIm.size() != 1)  // Multiple image, run in parall 
      {
         ExeMultiAutoRecallMMVII("0",aVSetIm);
         return EXIT_SUCCESS;
      }
   }
   // Single image, do the job ...

   if (mIntPyram)  // Case integer pyramids
   {
      cTplAppliCalcDescPCar<tINT2> aTplA(*this);
      aTplA.ExeGlob();
   }
   else   // Case floating points pyramids
   {
      cTplAppliCalcDescPCar<tREAL4> aTplA(*this);
      aTplA.ExeGlob();
   }
   
   return EXIT_SUCCESS;
}

cAppliCalcDescPCar:: cAppliCalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mIntPyram    (false),
  mSzTile      (7000),
  mOverlap     (300)
{
}


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

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
