#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ReadFileStruct.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"


/**
   \file importStaticScan.cpp

   \brief import static scan into instrument geometry
*/


namespace MMVII
{
   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportStaticScan                    */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportStaticScan : public cMMVII_Appli
{
     public :
        cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :


	// Mandatory Arg
	std::string              mNameFile;
	std::string              mStationName;
    std::string              mScanName;

	// Optional Arg
	tREAL8                   mAngTolerancy;

};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mAngTolerancy   (1e-6)
{
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
           <<  Arg2007(mStationName ,"Station name",{eTA2007::Topo}) // TODO: change type to future station
           <<  Arg2007(mScanName ,"Scan name",{eTA2007::Topo}) // TODO: change type to future scan
           ;
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mAngTolerancy,"AngTol","Angle tolerancy",{eTA2007::HDV})
    ;
}


int cAppli_ImportStaticScan::Exe()
{
    cTriangulation3D<float> aTriangulation3D(mNameFile);

    StdOut() << "Got " <<aTriangulation3D.NbPts() <<" points.\n";
    if (aTriangulation3D.HasPtAttribute())
    {
        StdOut() << "Intensity found.\n";
    }

    StdOut() << "Sample:\n";
    for (size_t i=0; (i<10)&&(i<aTriangulation3D.VPts().size()); ++i)
    {
        StdOut() << aTriangulation3D.KthPts(i);
        if (aTriangulation3D.HasPtAttribute())
            StdOut() << " " << aTriangulation3D.KthPtsPtAttribute(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportStaticScan::Samples() const
{
   return 
   {

   };
}


tMMVII_UnikPApli Alloc_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportStaticScan(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportStaticScan
(
     "ImportStaticScan",
      Alloc_ImportStaticScan,
      "Import static scan cloud point into instrument raster geometry",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::MMVIICloud},
      __FILE__
);

}; // MMVII

