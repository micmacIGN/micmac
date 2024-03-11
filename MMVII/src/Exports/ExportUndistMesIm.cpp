#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"

namespace MMVII
{

//class declaration
/*
class cAppli_ExportUndistMesIm : public cMMVII_Appli
{
    public:
        cAppli_ExportUndistMesIm(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec);

    private:
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        cPhotogrammetricProject  mPhProj;
        std::string              mSpecImIn;
        bool                     mShow;

};
*/

//constructor
/*
cAppli_ExportUndistMesIm::cAppli_ExportUndistMesIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mShow        (false)

{
}
*/

//handling mandatory arguments
/*
cCollecSpecArg2007 & cAppli_ExportUndistMesIm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}) //vector of eTA2007 giving meta information on the arg
             << ... //orientation folder
             << ... //points measure 2D/3D folder
      ;
}
*/

//handling optionnal arguments
/*
cCollecSpecArg2007 & cAppli_ExportUndistMesIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               <<  ... //output points measure 2D/3D folder
               << ... //bool to display
            ;
}
*/

//function as a main entry to the application
/*
int cAppli_ExportUndistMesIm::Exe()
{
    //construction of the photogrammetric project manager
    ...

    //retreive list of images in the pattern via VectMainSet(int aK) interface to MainSet
    ...
    
    //iterate over the images
    for (const std::string& aCImage : aVecIm)
    {
		//declare a set of measure Imgs GCPs via cSetMesImGCP (class for storing GCP :  3D measures + 2D image measure)
		...
		
		//load calibration via cPerspCamIntrCalib
		...

		//load GCPs
		...

		//load image measurements
		...

		//image measurements to export via cSetMesPtOf1Im (class for representing a set of measure in an image)
		...
		
		//iterate over cSetMesImGCP object
		for(const auto & aVMes : aSetMes.MesImInit())
		{
			//retreive image name
			...

			//retreive a vector of cMesIm1Pt (class for representing  the measure of a point in an image)
			...
			
			//iterate over cMesIm1Pt object
			for(const auto & aMes : aVMes.Measures())
			{
				//retreive point name
				...
				
				//retreive image coordinates of the point
				..

				//compute the corrected coordinates of the point from the distorsion 
				..
				
				//display
				if(mShow)
				{
					...
				}
				
				//fill a new object of type cMesIm1Pt with the new coordinates
				...
				
				//add the measure to the object of type cSetMesPtOf1Im
				...

			}
		}
		
		//write in a file the object of type cSetMesPtOf1Im
		...
	}

    return EXIT_SUCCESS;
}
*/

//function to allocates the application
/*
tMMVII_UnikPApli Alloc_Test_ExportUndistMesIm(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppli_ExportUndistMesIm(aVArgs,aSpec));
}
*/

//details about the application
/*
cSpecMMVII_Appli  TheSpec_ExportUndistMesIm
(
     "...", //application name
      ..., //allocator name
      "...", //comment
      {...}, //vector of feature
      {...}, //vector of type of input data
      {...}, //vector of type of output data
      __FILE__
);
*/

}
