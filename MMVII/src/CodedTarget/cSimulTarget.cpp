#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


// Test git branch

namespace MMVII
{

namespace  cNS_CodedTarget
{


/*  *********************************************************** */
/*                                                              */
/*              cAppliSimulCodeTarget                           */
/*                                                              */
/*  *********************************************************** */


class cAppliSimulCodeTarget : public cMMVII_Appli
{
     public :
        typedef tREAL4  tElem;
        typedef cIm2D<tElem>  tIm;


        cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // =========== other methods ============

        void    AddPosTarget(int aNum);  ///< Add the position of the target, don insert it


        // =========== Mandatory args ============
	std::string mNameIm;
	std::string mNameTarget;

        // =========== Optionnal args ============
        cPt2dr              mRayMinMax;

        // =========== Internal param ============
        tIm                 mImIn;
        cParamCodedTarget   mPCT;
	std::string         mDirTarget;
};


/* *************************************************** */
/*                                                     */
/*              cAppliSimulCodeTarget                  */
/*                                                     */
/* *************************************************** */

cAppliSimulCodeTarget::cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mRayMinMax    (15.0,60.0),
   mImIn         (cPt2di(1,1))
{
}

cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameTarget,"Name of target file")
   ;
}


cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
	        anArgOpt
             <<   Arg2007(mRayMinMax,"Min/Max ray for gen target",{eTA2007::HDV})
   ;
}


class cGeomSim
{
    public :
    private :
       cPt2dr mC;
    
};

void   cAppliSimulCodeTarget::AddPosTarget(int aNum)
{
     // cBox2r aBox 
     while (true)
     {
           // cPt2dr aP = ToR(mIm.DIm().GeneratePointInside();
     }
}

int  cAppliSimulCodeTarget::Exe()
{
   mPCT.InitFromFile(mNameTarget);
   mDirTarget =  DirOfPath(mNameTarget);

    mIm = tIm::FromFile(mNameIm);

   for (int aNum = 0 ; aNum<mPCT.NbCodeAvalaible() ; aNum++)
   {
        std::string aName = mDirTarget + mPCT.NameFileOfNum(aNum);
        StdOut() << "NNN= " << aName << "\n";
   }


   return EXIT_SUCCESS;
}
};


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_CodedTarget;

tMMVII_UnikPApli Alloc_SimulCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliSimulCodeTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecSimulCodedTarget
(
     "CodedTargetSimul",
      Alloc_SimulCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


};
