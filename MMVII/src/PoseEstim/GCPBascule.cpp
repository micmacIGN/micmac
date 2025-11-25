#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Matrix.h"

namespace MMVII
{
	
	class cAppli_GCPBascule : public cMMVII_Appli//heritage de cMMVII_Appli
	{
		public:
			cAppli_GCPBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec);
			void AddData(const cAuxAr2007 & anAuxInit);
			
		private:
			int Exe() override;//principale fonction qui execute tout le pipeline
			cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;//quasi systematiques arg obligatoire
			cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;//quasi systematiques arg optionnels
			cPhotogrammetricProject mPhProj;
			std::string mSpecImIn;
			bool mShow;
			cSimilitud3D<tREAL8> mSim;
			double mScale;
			cPt3dr mTr;
			cRotation3D<tREAL8> mRot;
			double mRes2;
			bool mCSVReport;
			bool mWriteSim;
	};
	
	
	cCollecSpecArg2007 & cAppli_GCPBascule::ArgObl(cCollecSpecArg2007 & anArgObl)
	{
		return anArgObl
			<< Arg2007(mSpecImIn, "Pattern/File of images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
			<< mPhProj.DPOrient().ArgDirInMand()
			<< mPhProj.DPGndPt2D().ArgDirInMand()
			<< mPhProj.DPGndPt3D().ArgDirInMand()
			<< mPhProj.DPOrient().ArgDirOutMand()
			;
	}
	
	cCollecSpecArg2007 & cAppli_GCPBascule::ArgOpt(cCollecSpecArg2007 & anArgOpt)
	{
		return anArgOpt
			<< AOpt2007(mShow,"show","show some useful details", {eTA2007::HDV})//hdv = has default value
			<< AOpt2007(mCSVReport,"CSVReport","save residuals to a csv file", {eTA2007::HDV})
			<< AOpt2007(mWriteSim,"Sim", "write similarity", {eTA2007::HDV})
		;
	}
	
	cAppli_GCPBascule::cAppli_GCPBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec):
		cMMVII_Appli(aVArgs,aSpec),
		mPhProj  (*this),
		mShow  (true),
		mCSVReport (true),
		mWriteSim (true)
		
	{
		//
	}
	
	//serialisation
	void cAppli_GCPBascule::AddData(const cAuxAr2007 & anAuxInit)
	{
		cAuxAr2007 anAux("Similitude 3d", anAuxInit);
		MMVII::AddData(cAuxAr2007("Scale", anAux), mScale);
		MMVII::AddData(cAuxAr2007("Translation", anAux), mTr);
		MMVII::AddData(cAuxAr2007("Rotation", anAux), mRot);
	}
	
	void AddData(const cAuxAr2007 & anAuxInit, cAppli_GCPBascule & aAppliBascule)
	{
		aAppliBascule.AddData(anAuxInit);
	}

	int cAppli_GCPBascule::Exe()
	{
		
			
		// 0 : construction du projet photogrammetrique
		mPhProj.FinishInit();
			
		// 1 : pseudo intersect. des cibles
		
		std::vector<std::string> aVImg = VectMainSet(0);//VectMainSet une méthode de l'appli
		
		if (mShow)
		{
			for (const auto & aNameIm : aVImg)
			{
                StdOut() << "aNameIm=" << aNameIm << std::endl;
			}
		}
		
		//to store gcps
		cSetMesGndPt aSet;
		
		//load input 3d measure
		mPhProj.LoadGCP3D(aSet);
		
		//load input 2d measure
		for (const auto & aNameIm : aVImg)
			{
				//read sensor
				cSensorCamPC* aCam = mPhProj.ReadCamPC(aNameIm, true);//PC = Pose Calibration
				//load 2d measure
				mPhProj.LoadIm(aSet, nullptr, *aCam);//LoadIm to load 2d measure
				
			}
			
		//pseudo intersection
		cSetMesGnd3D aInSet, aOutSet;
		
		for (const auto & aMesIm : aSet.MesImOfPt())
		{
			int aNumPt = aMesIm.NumPt();
			const cMes1Gnd3D & aMes3D = aSet.MesGCPOfNum(aNumPt);
			
			if(aMesIm.VMeasures().size() >= 2)
			{
				cMes1Gnd3D aInMes;
				aInMes.mPt = aSet.BundleInter(aMesIm);
				aInMes.mNamePt = aMes3D.mNamePt;
				aInSet.AddMeasure3D(aInMes);
				
				cMes1Gnd3D aOutMes;
				aOutMes.mPt = aMes3D.mPt;
				aOutMes.mNamePt = aMes3D.mNamePt;
				aOutSet.AddMeasure3D(aOutMes);
				
				StdOut() << "name = " << aInMes.mNamePt << " origin frame = " << aInMes.mPt
						<< " Target Frame = " << aOutMes.mPt <<"\n";
			}
		}
		
		//we need at leat 3 correspondences
		if(aInSet.Measures().size() < 3)
		{
			StdOut() << "not enough points ! " << "\n";
			return EXIT_FAILURE;
		}
		
		
		std::vector<cPt3dr> aInPts, aOutPts;
		for(const auto & aMes : aInSet.Measures())
			aInPts.push_back(aMes.mPt);
			
		for(const auto & aMes : aOutSet.Measures())
			aOutPts.push_back(aMes.mPt);
		
		mSim = mSim.StdGlobEstimate(aInPts, aOutPts,
						&mRes2,nullptr,cParamCtrlOpt::Default());
		if (mShow)

		{
			StdOut() << "simres=" << mRes2 << "\n";
			StdOut() << "scale=" << mSim.Scale() << "\n";
			StdOut() << "trans=" << mSim.Tr() << "\n";
			StdOut() << "rot=" << mSim.Rot().Mat() << "\n";
		}
		
		
		for (const auto & aIm : aVImg)
		{
			//read sensor
			cSensorCamPC* aCam = mPhProj.ReadCamPC(aIm,true);
			
			//get image pose
			const tPoseR & aPose = aCam->Pose();
			
			//apply sim to pose
			tPoseR aTransformedPose = TransfoPose(mSim, aPose);
			aCam -> SetPose(aTransformedPose);
			
			//save
			mPhProj.SaveSensor(*aCam);
		}
		
		
				// 2 : estimation de la similitude
		// 3 : application de la similitude aux ori en entrée
		// 4 : generate report
		if(mCSVReport)
		{
			//csv file name
			std::string aReportFileName = "3D-Sim-Residuals";
			
			//csv header
			InitReportCSV(aReportFileName, "csv", "false", {"pt_name", "x_res", "y_res", "z_res"});
		
			//vector to store transformed points
			std::vector<cPt3dr> aVTransformedPts;
			
			for (size_t i=0;i<aInSet.Measures().size();i++)
			{
				cPt3dr aInPt = aInSet.Measures().at(i).mPt;
				cPt3dr aTransformedPt = mSim.Value(aInPt);
				cPt3dr aOutPt = aOutSet.Measures().at(i).mPt;
				//residuals
				double dx = aTransformedPt.x() - aOutPt.x();
				double dy = aTransformedPt.y() - aOutPt.y();
				double dz = aTransformedPt.z() - aOutPt.z();
				
				//add to csv file
				AddOneReportCSV(aReportFileName, {aInSet.Measures().at(i).mNamePt,
								ToStr(dx),ToStr(dy),ToStr(dz)});
			}
			
		}
		
		if(mWriteSim)
		{
			//file name
			std::string aNameFile = mPhProj.DPOrient().FullDirOut()
									+ "3D-Similarity"
									+ "_"
									+ mPhProj.DPOrient().DirIn()
									+ "_"
									+ mPhProj.DPOrient().DirOut()
									+std::string(".xml");
			
			
			//assign
			mScale = mSim.Scale();
			mTr = mSim.Tr();
			mRot = mSim.Rot();
			//save file
			SaveInFile(*this,aNameFile);
			StdOut() << "Sim params wrote to file :" << aNameFile << '\n';
		}
		
	
	
		return EXIT_SUCCESS;	
	}
	
	
	
	
	//pour faire des trucs avec la memoire - obligatoire
	tMMVII_UnikPApli Alloc_GCPBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_GCPBascule(aVArgs, aSpec));
	}
	
	cSpecMMVII_Appli TheSpec_GCPBascule
	(
		"GCPBascule",
		Alloc_GCPBascule,
		"Perform a bascule based on GCPs",
		//metadonnees
		{eApF::Ori,eApF::GCP},//features
		{eApDT::ObjCoordWorld, eApDT::ObjMesInstr},//inputs
		{eApDT::Console},//output
		__FILE__
	);
		
	

	//args obl:
	//pattern image
	//input ori
	//input mes2d
	//input mes3d
	//output name
	
	//args opt :
	// show res

	
} //MMVII
