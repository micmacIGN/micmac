#include "include/MMVII_all.h"
#include "include/V1VII.h"

namespace MMVII
{
namespace NS_FormatTDEDM
{

class cAppliFormatTDEDM ;  // format 4 training data on epipolar dense matching

class cAppliFormatTDEDM : public cMMVII_Appli
{
     public :

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cAppliFormatTDEDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        std::string ComConvertImage(const std::string & aIm1,const std::string & aDirAdd,const std::string & aPrefix) const;

        std::string NameRes(const std::string & aPref,const std::string & aName) const
        {
             return DirProject() + aPref + Prefix (aName) + ".tif";
        }
     private :
       // =========== Data ========
            // Mandatory args
            std::string mPatIm1;
};


/*  ============================================== */
/*                                                 */
/*              cAppli                             */
/*                                                 */
/*  ============================================== */

cAppliFormatTDEDM::cAppliFormatTDEDM
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliFormatTDEDM::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   
   return 
      anArgObl  
         << Arg2007(mPatIm1,"Pattern or Xml for modifying",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliFormatTDEDM::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
   ;
}


std::string  cAppliFormatTDEDM::ComConvertImage
             (
                  const std::string & aName,
                  const std::string & aDirAdd,
                  const std::string & aPrefix
              ) const
{
    std::string aDirIm = DirOfPath(mPatIm1,false) + aDirAdd;
       
    std::string aCom =   "convert -colorspace Gray -compress none " + aDirIm + aName 
                       + " " + NameRes(aPrefix,aName) 
                       // + " " + DirProject() + aPrefix + Prefix(aName) + ".tif"
    ;

    return aCom;
}

int cAppliFormatTDEDM::Exe()
{
   std::list<std::string> aLCom;
   for (const auto & aName : VectMainSet(0))
   {
      aLCom.push_back(ComConvertImage(aName,"","Im1_"));
      aLCom.push_back(ComConvertImage(aName,"../colored_1/","Im2_"));
      aLCom.push_back(ComConvertImage(aName,"../disp_occ/","PxInit_"));
   }

   ExeComParal(aLCom);
   aLCom.clear();

   for (const auto & aName : VectMainSet(0))
   {
        std::string aNameIn = NameRes("PxInit_",aName);

        std::string aComDyn = "mm3d Nikrup \"/ " + aNameIn + " -256.0\" " + NameRes("Px_",aName);
        std::string aComMasq = "mm3d Nikrup \"* 255 !=  " + aNameIn + " 0\" " + NameRes("Masq_",aName);

        aLCom.push_back(aComDyn);
        aLCom.push_back(aComMasq);
   }
   ExeComParal(aLCom);
   aLCom.clear();

   for (const auto & aName : VectMainSet(0))
   {
        std::string aNameIn = NameRes("PxInit_",aName);
        RemoveFile(aNameIn,false);
   }

   return EXIT_SUCCESS;
}


/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_FormatTDEDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliFormatTDEDM(aVArgs,aSpec));
}

}; //  NS_FormatTDEDM
}


namespace MMVII
{
cSpecMMVII_Appli  TheSpecFormatTDEDM
(
     "DMFormatTD",
      NS_FormatTDEDM::Alloc_FormatTDEDM,
      "Dense Match: Format Training data",
      {eApF::Match},
      {eApDT::Image,eApDT::FileSys},
      {eApDT::Image,eApDT::FileSys},
      __FILE__
);

};

