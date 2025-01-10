#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_V2ImportCalib                       */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportOrgi : public cMMVII_Appli
{
     public :
        cAppli_ImportOrgi(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string                 mOrgiFolder;
	std::string                 mNameFileModel;
	std::string                 mNameFileAmer;
	std::string                 mNameFileAppui;
	std::string                 mNameFileLiaison;

	std::map<int,std::string>   mMapNum2Im; //
	std::vector<std::string>    mVecNum2Pt; //
	std::string                 mNameOut;
	std::vector<std::string>    mChgImageName;
         
};

cAppli_ImportOrgi::cAppli_ImportOrgi(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli       (aVArgs,aSpec),
   mPhProj            (*this),
   mNameFileModel     ("MODELES.TXT"),
   mNameFileAmer      ("AMERS.TXT"),
   mNameFileAppui     ("APPUIS.TXT"),
   mNameFileLiaison   ("LIAISONS.TXT"),
   mNameOut           ("ORGI")
{
}

cCollecSpecArg2007 & cAppli_ImportOrgi::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mOrgiFolder ,"Folder where ORGI's data are stored")
	      // <<  Arg2007(mExtCalib ,"Calib to import")
              // <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportOrgi::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    
    return anArgOpt
           // << AOpt2007(mNameFileModel,"NameModel","Name for file of \"model\"",{eTA2007::HDV})
           // << AOpt2007(mNameFileAmer,"NameAmer","Name for file of 3D-GCP measure (AKA \"ORGI::amers\")",{eTA2007::HDV})
           // << AOpt2007(mNameFileAppui,"NameAppuis","Name for file of 2D-GCP measure in images (AKA \"ORGI::appuis\")",{eTA2007::HDV})
           << AOpt2007(mChgImageName,"ChImage","For changing name of images [Pat,Chg]",{{eTA2007::ISizeV,"[2,2]"}})
       //  << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       //  << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       //  << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       //  << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)")
    ;
}


int cAppli_ImportOrgi::Exe()
{
    mPhProj.DPGndPt2D().SetDirOut(mNameOut);
    mPhProj.DPGndPt3D().SetDirOut(mNameOut);
    mPhProj.DPMulTieP().SetDirOut(mNameOut);

    mPhProj.FinishInit();

    //  Read the models and construct the map int -> Name
    {
         cReadFilesStruct aFileModel(mOrgiFolder+mNameFileModel,"IES",0,-1,'#');
	 aFileModel.Read();

	 for (int aK=0 ; aK< aFileModel.NbRead() ; aK++)
	 {
             std::string aNameIm = aFileModel.VNameIm().at(aK);
	     ChgName(mChgImageName,aNameIm);
             mMapNum2Im[aFileModel.VInts().at(aK).at(0)] = aNameIm;
	 }
    }

    // Read the GCPs
    {
	cSetMesGnd3D   aMesGCP(mNameOut);
        //  Read the 3D measures of GCP 
        {
             cReadFilesStruct aFileAmers(mOrgiFolder+mNameFileAmer,"NXYZ",0,-1,'#');
	     aFileAmers.Read();
	     for (int aK=0 ; aK<aFileAmers.NbRead() ; aK++)
	     {
                 const std::string & aNamePt = aFileAmers.VNamePt().at(aK);
                 cPt3dr  aPt         =  aFileAmers.VXYZ().at(aK);
	         mVecNum2Pt.push_back(aNamePt);
	         aMesGCP.AddMeasure3D(cMes1Gnd3D(aPt,aNamePt,1.0)); 
	     }
        }

        //  Read the 2D measures of GCP
        //  Fichier  APPUIS.TXT  (ORGI)
        std::map<int,cSetMesPtOf1Im> aMapMesIm;
        {
             cReadFilesStruct aFileAppuis(mOrgiFolder+mNameFileAppui,"EEXY",0,-1,'#');
	     aFileAppuis.Read();
	     for (int aK=0 ; aK<aFileAppuis.NbRead() ; aK++)
	     {
                  int aNumPt = aFileAppuis.VInts().at(aK).at(0)-1;  // 1 (and not 0) is number of  fist Line in ORGI conv
                  int aNumIm = aFileAppuis.VInts().at(aK).at(1);
	          cPt2dr aPIm = PSymXY(Proj(aFileAppuis.VXYZ().at(aK)));
    
	          cSetMesPtOf1Im & aMesIm  = aMapMesIm[aNumIm];
	          std::string aNameIm = mMapNum2Im[aNumIm];
                  aMesIm.SetNameIm(aNameIm);

	          aMesIm.AddMeasure(cMesIm1Pt(aPIm,mVecNum2Pt.at(aNumPt),1.0));
	     }
        }

        mPhProj.SaveGCP3D(aMesGCP);
        for (const auto & [aNum,aMesIm] : aMapMesIm)
        {
             mPhProj.SaveMeasureIm(aMesIm);
        }
    }

    // Read the TieP
    {
        std::map<std::string,cVecTiePMul>     aMapTieP;
        cReadFilesStruct aFileLiason(mOrgiFolder+mNameFileLiaison,"EEXY",0,-1,'#');
        aFileLiason.Read();
	for (int aK=0 ; aK<aFileLiason.NbRead() ; aK++)
	{
            int aNumPt = aFileLiason.VInts().at(aK).at(0)-1;  // 1 (and not 0) is number of  fist Line in ORGI conv
            int aNumIm = aFileLiason.VInts().at(aK).at(1);
	    cPt2dr aPIm = PSymXY(Proj(aFileLiason.VXYZ().at(aK)));
	    std::string aNameIm = mMapNum2Im[aNumIm];
            auto & aVTMP = aMapTieP[aNameIm];
            if (aVTMP.mNameIm =="")
               aVTMP.mNameIm = aNameIm;
            aVTMP.mVecTPM.push_back(cTiePMul(aPIm,aNumPt));

        }
        for (const auto & [aName,aVecMTp] : aMapTieP)
        {
            mPhProj.SaveMultipleTieP(aVecMTp,aName);
        }
    }

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportOrgi::Samples() const
{
	return {"MMVII V2ImportCalib ../../Pannel/ BA_725 CalibInit725"};
}


tMMVII_UnikPApli Alloc_ImportOrgi(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportOrgi(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportORGI
(
     "ImportORGI",
      Alloc_ImportOrgi,
      "Import data coming from \"ORGI\" format",
      {eApF::GCP,eApF::TieP},
      {eApDT::FileSys},
      {eApDT::TieP,eApDT::ObjMesInstr,eApDT::ObjCoordWorld},
      __FILE__
);


}; // MMVII

