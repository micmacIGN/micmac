#include "StdAfx.h"
#include "V1VII.h"
#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "../Serial/Serial.h"

namespace MMVII
{

extern void TestReadXML(const std::string&);

/* =============================================== */
/*                                                 */
/*                 cPushbroomSensor                */
/*                                                 */
/* =============================================== */

/** Class for XXX */

class cPushbroomSensor : public cSensorImage
{
    public :
        cPushbroomSensor();

    private:


};

/* =============================================== */
/*                                                 */
/*                 cPolyn                          */
/*                                                 */
/* =============================================== */

/**  A class containing a polynomial   */

class cPolyn
{
    public:
        cPolyn(){};

        void Initialise(const cSerialTree &,const std::string&);
        void Show();

        const double * Coeffs() const {return mCoeffs;}
        double * Coeffs() {return mCoeffs;}
        double Val(const cPt3dr &) const;

    private:
        double mCoeffs[20];

};

void cPolyn::Initialise(const cSerialTree & aData,const std::string& aPrefix)
{
    for (int aK=1; aK<21; aK++)
    {
        const cSerialTree * aItem = aData.GetUniqueDescFromName(aPrefix+ToString(aK));
        mCoeffs[aK-1] =  std::stod(aItem->UniqueSon().Value()); // coeffs
    }
}

void cPolyn::Show()
{
    for (int aK=0; aK<20; aK++)
    {
        StdOut() << mCoeffs[aK] << "\t";

        if ( (aK+1)%5 == 0 )
            StdOut() << std::endl;

    }
    StdOut() << std::endl;
}
/* =============================================== */
/*                                                 */
/*                 cRatioPolyn                     */
/*                                                 */
/* =============================================== */

/**  A class containing a rational polynomial  */

class cRatioPolyn//1
{
    public:
        cRatioPolyn(){};

        const cPolyn & NumPoly() const {return mNumPoly;} // read access
        cPolyn & NumPoly() {return mNumPoly;}             // write access

        const cPolyn & DenPoly() const {return mDenPoly;} // read access
        cPolyn & DenPoly() {return mDenPoly;}             // write access

        void Show();
        double Val(const cPt3dr &) const;



    private:
        cPolyn mNumPoly; // numerator polynomial
        cPolyn mDenPoly; // denominator polynomial


};

void cRatioPolyn::Show()
{
    StdOut() << "Numerator:" << std::endl;
    mNumPoly.Show();

    StdOut() << "Denumerator:" << std::endl;
    mDenPoly.Show();
}

//  class cRatioPolynXY   cRatioPolyn1 mX;  cRatioPolyn  mY;
//  cPt2dr Val(const cPtd3r &) const;
class cRatioPolynXY
{
    public:
        cRatioPolynXY(){};

        const cRatioPolyn & X() const {return mX;}
        cRatioPolyn & X() {return mX;}

        const cRatioPolyn & Y() const {return mY;}
        cRatioPolyn & Y() {return mY;}

        cPt2dr Val(const cPt3dr &) const;
        void   Show();

    private:
        cRatioPolyn mX;
        cRatioPolyn mY;

};

void cRatioPolynXY::Show()
{
    StdOut() << "\t Coordinate 1:" << std::endl;
    mX.Show();

    StdOut() << "\t Coordinate 2:" << std::endl;
    mY.Show();
}


/* =============================================== */
/*                                                 */
/*                 cDataRPC                        */
/*                                                 */
/* =============================================== */

/**  A class containing RPCs  */

class cDataRPC
{
    public:
        cDataRPC();
        void ReadXML(const std::string&);
        double ReadXMLItem(const cSerialTree&,const std::string&);
        void   ReadXMLModel(const cSerialTree&,const std::string&,cRatioPolynXY *);
        void   ReadXMLNorms(const cSerialTree&);

        void Show();

        const cRatioPolynXY& DirectRPC() {return *mDirectRPC;}
        const cRatioPolynXY& InverseRPC() {return *mInverseRPC;}

        const cPt2dr & ImOffset() const {return mImOffset;}
        cPt2dr & ImOffset() {return mImOffset;}

        const cPt2dr & ImScale() const {return mImScale;}
        cPt2dr & ImScale() {return mImScale;}

        const cPt3dr & GroundOffset() const {return mGroundOffset;}
        cPt3dr & GroundOffset() {return mGroundOffset;}

        const cPt3dr & GroundScale() const {return mGroundScale;}
        cPt3dr & GroundScale() {return mGroundScale;}

    private:
        cRatioPolynXY * mDirectRPC; // rational polynomial for ground to image projection
        cRatioPolynXY * mInverseRPC; // rational polynomial for image to ground projection

        /// coordinate normalisation:  coord_norm = coord * coord_scale + coord_offset
        cPt2dr mImOffset; // sample and line offsets
        cPt2dr mImScale;  // sample and line scales

        cPt3dr mGroundOffset; // ground coordinate (e.g., lambda, phi, h) offets
        cPt3dr mGroundScale;  // ground coordinate (e.g., lambda, phi, h) scales
};

cDataRPC::cDataRPC() :
    mDirectRPC(nullptr),
    mInverseRPC(nullptr)
{}

void cDataRPC::ReadXMLModel(const cSerialTree& aTree,const std::string& aPrefix,cRatioPolynXY * aModel)
{
    const cSerialTree * aDirect = aTree.GetUniqueDescFromName(aPrefix);

    aModel->X().NumPoly().Initialise(*aDirect,"LINE_NUM_COEFF_");
    aModel->X().DenPoly().Initialise(*aDirect,"LINE_DEN_COEFF_");

    aModel->Y().NumPoly().Initialise(*aDirect,"SAMP_NUM_COEFF_");
    aModel->Y().DenPoly().Initialise(*aDirect,"SAMP_DEN_COEFF_");

}

void cDataRPC::ReadXMLNorms(const cSerialTree& aTree)
{
    const cSerialTree * aData = aTree.GetUniqueDescFromName("RFM_Validity");

    mGroundOffset.x() = ReadXMLItem(*aData,"LAT_OFF");
    mGroundOffset.y() = ReadXMLItem(*aData,"LONG_OFF");
    mGroundOffset.z() = ReadXMLItem(*aData,"HEIGHT_OFF");

    mGroundScale.x() = ReadXMLItem(*aData,"LAT_SCALE");
    mGroundScale.y() = ReadXMLItem(*aData,"LONG_SCALE");
    mGroundScale.z() = ReadXMLItem(*aData,"HEIGHT_SCALE");

    mImScale.x() = ReadXMLItem(*aData,"LINE_SCALE");
    mImScale.y() = ReadXMLItem(*aData,"SAMP_SCALE");

    mImOffset.x() = ReadXMLItem(*aData,"LINE_OFF");
    mImOffset.y() = ReadXMLItem(*aData,"SAMP_OFF");
}

double cDataRPC::ReadXMLItem(const cSerialTree & aData,const std::string& aPrefix)
{
    return std::stod(aData.GetUniqueDescFromName(aPrefix)->UniqueSon().Value());
}

void cDataRPC::ReadXML(const std::string& aNameFile)
{
    cSerialFileParser * aSFP = cSerialFileParser::Alloc(aNameFile,eTypeSerial::exml);
    cSerialTree  aTree(*aSFP);


    // read the direct model
    mDirectRPC = new cRatioPolynXY();
    ReadXMLModel(aTree,"Direct_Model",mDirectRPC);


    // read the inverse model
    mInverseRPC = new cRatioPolynXY();
    ReadXMLModel(aTree,"Inverse_Model",mInverseRPC);


    // read the normalisation data (offset, scales)
    ReadXMLNorms(aTree);


    delete aSFP;


}

void cDataRPC::Show()
{
    StdOut() << "\t======= Direct model =======" << std::endl;
    mDirectRPC->Show();

    if (mInverseRPC!=nullptr)
    {
        StdOut() << "\t======= Inverse model =======" << std::endl;
        mInverseRPC->Show();
    }

    StdOut() << "\t======= Normalisation data =======" << std::endl;
    StdOut() << "IMAGE OFFSET=\t" << mImOffset << std::endl;
    StdOut() << "IMAGE SCALE=\t" << mImScale << std::endl;
    StdOut() << "GROUND OFFSET=\t" << mGroundOffset << std::endl;
    StdOut() << "GROUND SCALE=\t" << mGroundScale << std::endl;

}

void TestDataRPCReasXML(const std::string& aNameFile)
{
    cDataRPC aRPC;
    aRPC.ReadXML(aNameFile);
    aRPC.Show();
}


/*
 *

    DataRPC contains 2 obj of RatioPolyn, as well as validity

        constructor par def qui fait rien
        function ReadXML qui lit
*/

/* =============================================== */
/*                                                 */
/*                 cAppliImportPushbroom           */
/*                                                 */
/* =============================================== */

/**  A basic application for  */

class cAppliImportPushbroom : public cMMVII_Appli
{
     public :

        cAppliImportPushbroom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     // --- Mandatory ----
    std::string mNameSensorIn;

     // --- Optionnal ----
    std::string mNameSensorOut;

     // --- Internal ----
};

cAppliImportPushbroom::cAppliImportPushbroom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliImportPushbroom::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return anArgObl
      <<   Arg2007(mNameSensorIn,"Name of input sensor gile", {eTA2007::FileDirProj,eTA2007::Orient})
   ;
}

cCollecSpecArg2007 & cAppliImportPushbroom::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameSensorOut,CurOP_Out,"Name of output file if correction are done")
   ;
}

int cAppliImportPushbroom::Exe()
{

    //TestReadXML(mNameSensorIn);
    TestDataRPCReasXML(mNameSensorIn);


    return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_ImportPushbroom(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliImportPushbroom(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecImportPushbroom
(
     "ImportPushbroom",
      Alloc_ImportPushbroom,
      "Import a pushbroom sensor",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

};
