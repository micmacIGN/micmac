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

class cView; ///< class storing a view of a triplet
class cTriplet; ///< a class storing three views

typedef cIsometry3D<tREAL8>  tPose;


/* ********************************************************** */
/*                                                            */
/*                        cView                               */
/*                                                            */
/* ********************************************************** */

cView::cView(const tPose aPose,const std::string aName) :
    mName(aName),
    mPose(aPose)
    {
    };

cView::cView() :
    cView(tPose::Identity(),"NONE")
{}

void cView::AddData(const cAuxAr2007 &anAuxInit)
{
    cAuxAr2007 anAux("View",anAuxInit);

    MMVII::AddData(cAuxAr2007("Name",anAux),mName);
    MMVII::AddData(cAuxAr2007("Ori",anAux),mPose);

}

void AddData(const  cAuxAr2007 &anAux,cView &aV)
{
    aV.AddData(anAux);
}



/* ********************************************************** */
/*                                                            */
/*                        cTriplet                            */
/*                                                            */
/* ********************************************************** */
cTriplet::cTriplet() :
    mPoses(std::vector<cView>())
{
    mPoses.push_back(cView());
    mPoses.push_back(cView());
    mPoses.push_back(cView());
}

void cTriplet::AddData(const cAuxAr2007 &anAuxInit)
{
    cAuxAr2007 anAux("Triplet",anAuxInit);
    //
    // Save the relative poses
    // Pose1, Pose 21, Pose31
    //
    MMVII::AddData(cAuxAr2007("Id",anAux),mId);
    MMVII::AddData(cAuxAr2007("Pose1",anAux),mPoses[0]);
    MMVII::AddData(cAuxAr2007("Pose21",anAux),mPoses[1]);
    MMVII::AddData(cAuxAr2007("Pose31",anAux),mPoses[2]);
    MMVII::AddData(cAuxAr2007("BH",anAux),mBH);
    MMVII::AddData(cAuxAr2007("Residual",anAux),mResidual);

}

void AddData(const cAuxAr2007& anAux,cTriplet& aTri)
{
    aTri.AddData(anAux);
}



   /* ********************************************************** */
   /*                                                            */
   /*                     cTripletSet                            */
   /*                                                            */
   /* ********************************************************** */

cTripletSet::cTripletSet() :
    mName("v0")
{}

void cTripletSet::PushTriplet(cTriplet &aTri)
{
    mSet.push_back(aTri);
}

void cTripletSet::ToFile(const std::string &aName) const
{
    SaveInFile(this->mSet,aName);
}

cTripletSet * cTripletSet::FromFile(const std::string &aName)
{
    StdOut() << aName << std::endl;
    cTripletSet * aRes = new cTripletSet;

    ReadFromFile(aRes->Set(),aName);

    return aRes;
}

void cTripletSet::AddData(const  cAuxAr2007 & anAuxInit)
{
    cAuxAr2007 anAux("TripletSet",anAuxInit);
    // ...
    // Put the data in  tag "cTripletSet"

    // Add data for
    //    mName
    //    ...
    //
    //MMVII::AddData(cAuxAr2007("Name",anAux),mName);
    MMVII::AddData(cAuxAr2007("Sets",anAux),mSet);


}

void AddData(const  cAuxAr2007 & anAux,cTripletSet & aSet)
{
    aSet.AddData(anAux);
}


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
         // << mPhProj.DPOrient().ArgDirInMand()
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



#if (MMVII_KEEP_LIBRARY_MMV1)
int cAppli_ImportTriplet::Exe()
{
    mPhProj.FinishInit();

    cTripletSet aSetOfTri;
    if (IsInit(&mNameTriSet))
        aSetOfTri.SetName(mNameTriSet);

    cXml_TopoTriplet aXml3 =  StdGetFromSI(mNameFile,Xml_TopoTriplet);

    int aId=0;
    for (auto aTriXML : aXml3.Triplets())

    {

        std::string aNameDirTwoView = DirOfFile(mNameFile) + aTriXML.Name1() + "/" + aTriXML.Name2() + "/";
        std::string aNameTriplet = aNameDirTwoView + "Triplet-OriOpt-" + aTriXML.Name3() + (mFileBin ? ".dmp" : ".xml");
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


        cTriplet aTri;

        aTri.Id() = aId++;

        aTri.PVec()[0] = (cView(tPose::Identity(),aTriXML.Name1()));
        aTri.PVec()[1] = (cView(tPose(aCenter21,aR21),aTriXML.Name2()));
        aTri.PVec()[2] = (cView(tPose(aCenter31,aR31),aTriXML.Name3()));


        ///metrics (B/h, residual)
        aTri.BH() = aXml3Ori.BSurH();
        aTri.Residual() = aXml3Ori.ResiduTriplet();


        /// push triplet to the set
        aSetOfTri.PushTriplet(aTri);



    }

    /// save triplets' graph to mmvii format
    mPhProj.SaveTriplets(aSetOfTri);



    return EXIT_SUCCESS;
}

#else // MMVII_KEEP_LIBRARY_MMV1

int cAppli_ImportTriplet::Exe()
{
    MMVII_INTERNAL_ERROR("This functionality requires compiling MMV2 with MMV1");
    return EXIT_FAILURE;
}
#endif // MMVII_KEEP_LIBRARY_MMV1


tMMVII_UnikPApli Alloc_ImportTriplet(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportTriplet(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportTriplet
(
     "ImportTripletV1",
      Alloc_ImportTriplet,
      "Import/Convert triplets in MMVII format",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);


}; // MMVII

