#include "MMVII_Bench.h"
#include "MMVII_Class4Bench.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_Geom2D.h"
#include "MMVII_PCSens.h"
#include "Serial.h"
#include "MMVII_MeasuresIm.h"


/** \file DemoSerial.cpp
    \brief file for generating spec+samples of serialization

*/

namespace MMVII
{

/* *********************************************************** */
/*                                                             */
/*                   cAppliSpecSerial                          */
/*                                                             */
/* *********************************************************** */

class cAppliSpecSerial : public cMMVII_Appli
{
     public :

        cAppliSpecSerial(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
};


cAppliSpecSerial::cAppliSpecSerial
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli   (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliSpecSerial::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
   ;
}


cCollecSpecArg2007 & cAppliSpecSerial::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
                  anArgOpt
          ;
}


void GenSpec_BitEncoding(const std::string & aDir);
void GenSpec_SysCoordV1(const std::string & aDir);



int  cAppliSpecSerial::Exe()
{
   std::string aDir = DirRessourcesMMVII() + "SpecifSerial/";

   // SpecificationSaveInFile<cTestSerial1>(aDir+"cTestSerial1.xml");
   // SpecificationSaveInFile<cTestSerial1>(aDir+"cTestSerial1.json");

   GenSpec_BitEncoding(aDir);
   GenSpec_SysCoordV1(aDir);
   SpecificationSaveInFile<tNameSet>(aDir+"SetName.xml");
   SpecificationSaveInFile<cSetMesPtOf1Im>(aDir+"SetMesureIm.xml");
   SpecificationSaveInFile<cSetMesGCP>(aDir+"SetMesureGCP.xml");

   return EXIT_SUCCESS;
}

/* *********************************************************** */
/*                                                             */
/*                           ::                                */
/*                                                             */
/* *********************************************************** */


tMMVII_UnikPApli Alloc_cAppliSpecSerial(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliSpecSerial(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SpecSerial
(
     "GenerateSpecifSerial",
      Alloc_cAppliSpecSerial,
      "Generate specification+some sample for serialization",
      {eApF::Project,eApF::Test},
      {eApDT::None},
      {eApDT::Xml},
      __FILE__
);

/*
*/




};
