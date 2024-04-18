#include "V1VII.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PoseTriplet.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file ImportTriplet.cpp

   \brief file for importing triplets from mmv1
*/


namespace MMVII
{


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportTriplet                       */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportTriplet : public cMMVII_Appli
{
     public :
        typedef cIsometry3D<tREAL8>  tPose;

        cAppli_ImportTriplet(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;
    bool                     mFileBin;


    // Optionall Arg
    std::string              mNameTriSet;
    tNameSet                 mSetFilter;


    /*
        // contains all nodes (views) with global orientation
        // list of triplets of nodes
        // for every triplet, for every edge of the triplet -> a list of triplets that are adjacent (have an edge in common)
        // for every node, a list of triplets it appears in
        // functions:
        //    partition the graph => create several  */

};

cAppli_ImportTriplet::cAppli_ImportTriplet(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mFileBin      (true)
{
}

cCollecSpecArg2007 & cAppli_ImportTriplet::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
          << Arg2007(mNameFile ,"Name of file with a list of triplet")
          << mPhProj.DPOrient().ArgDirInMand()
          << mPhProj.DPOriTriplets().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportTriplet::ArgOpt(cCollecSpecArg2007 & anArgObl)
{
    
    return anArgObl
            << AOpt2007(mNameTriSet,"NameTri", "Name of the output triplet set; def=TripletSet")
            << AOpt2007(mFileBin,"FileBin","Binary file format, def=true")
               ;
}




int cAppli_ImportTriplet::Exe()
{
    mPhProj.FinishInit();

    cTripletSet aSetOfTri;
    if (IsInit(&mNameTriSet))
        aSetOfTri.SetName(mNameTriSet);

    cXml_TopoTriplet aXml3 =  StdGetFromSI(mNameFile,Xml_TopoTriplet);

    for (auto aTriXML : aXml3.Triplets())
    {

        std::string aNameDirTwoView = DirOfFile(mNameFile) + aTriXML.Name1() + "/" + aTriXML.Name2() + "/";
        std::string aNameTriplet = aNameDirTwoView + "Triplet-OriOpt-" + aTriXML.Name3() + (mFileBin ? ".dat" : ".xml");
        //StdOut() << aNameTriplet << std::endl;

        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aNameTriplet,Xml_Ori3ImInit);

        /// rotations
        cXml_Rotation aIm2in1_rot = aXml3Ori.Ori2On1();

        const cPtxd<double,3>  aI21(aIm2in1_rot.Ori().L1().x,
                                    aIm2in1_rot.Ori().L1().y,
                                    aIm2in1_rot.Ori().L1().z);
        const cPtxd<double,3> aJ21(aIm2in1_rot.Ori().L2().x,
                                    aIm2in1_rot.Ori().L2().y,
                                    aIm2in1_rot.Ori().L2().z);
        const cPtxd<double,3> aK21(aIm2in1_rot.Ori().L3().x,
                                    aIm2in1_rot.Ori().L3().y,
                                    aIm2in1_rot.Ori().L3().z);

        cRotation3D<tREAL8> aR21(aI21,aJ21,aK21,false);

        cXml_Rotation aIm3in1_rot = aXml3Ori.Ori3On1();
        const cPtxd<double,3>  aI31(aIm3in1_rot.Ori().L1().x,
                                    aIm3in1_rot.Ori().L1().y,
                                    aIm3in1_rot.Ori().L1().z);
        const cPtxd<double,3> aJ31(aIm3in1_rot.Ori().L2().x,
                                    aIm3in1_rot.Ori().L2().y,
                                    aIm3in1_rot.Ori().L2().z);
        const cPtxd<double,3> aK31(aIm3in1_rot.Ori().L3().x,
                                    aIm3in1_rot.Ori().L3().y,
                                    aIm3in1_rot.Ori().L3().z);

        cRotation3D<tREAL8> aR31(aI31,aJ31,aK31,false);

        /// centers
        cPtxd<double,3> aCenter21(aIm2in1_rot.Centre().x,
                                aIm2in1_rot.Centre().y,
                                aIm2in1_rot.Centre().z);
        cPtxd<double,3> aCenter31(aIm3in1_rot.Centre().x,
                                aIm3in1_rot.Centre().y,
                                aIm3in1_rot.Centre().z);

        //StdOut() << aTriXML.Name1() << " " << aTriXML.Name2() << " " << aTriXML.Name3() << std::endl;
        //StdOut() << "centers: " << aCenter21 << " " << aCenter31 << std::endl;

        tPose aPose1 = cIsometry3D<tREAL8>::Identity();
        tPose aPose21(aCenter21,aR21);
        tPose aPose31(aCenter31,aR31);

        cTriplet aTri;

        aTri.PVec()[0] = (cView(tPose::Identity(),aTriXML.Name1()));
        aTri.PVec()[1] = (cView(tPose(aCenter21,aR21),aTriXML.Name2()));
        aTri.PVec()[2] = (cView(tPose(aCenter31,aR31),aTriXML.Name3()));


        ///metrics (B/h, residual)
        aTri.BH() = aXml3Ori.BSurH();
        aTri.Residual() = aXml3Ori.ResiduTriplet();


        /// push triplet to the set
        aSetOfTri.PushTriplet(aTri);

        //cPerspCamIntrCalib * aCalib = mPhProj.InternalCalibFromImage(aTriXML.Name1());
        //StdOut() << "Calib: " << aCalib->F() << " " << aCalib->PP() << std::endl;



    }


    mPhProj.SaveTriplets(aSetOfTri);

    return EXIT_SUCCESS;
}




tMMVII_UnikPApli Alloc_ImportTriplet(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportTriplet(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportTriplet
(
     "ImportTriplet",
      Alloc_ImportTriplet,
      "Import/Convert triplets in MMVII format",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);


}; // MMVII

