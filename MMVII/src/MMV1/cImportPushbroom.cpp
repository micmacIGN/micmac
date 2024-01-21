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

double cPolyn::Val(const cPt3dr& aP) const
{
    return    mCoeffs[0]
            + mCoeffs[5]  * aP.y() * aP.z()
            + mCoeffs[10] * aP.x() * aP.y() * aP.z()
            + mCoeffs[15] * aP.x() * aP.x() * aP.x()
            + mCoeffs[1]  * aP.y()
            + mCoeffs[6]  * aP.x() * aP.z()
            + mCoeffs[11] * aP.y() * aP.y() * aP.y()
            + mCoeffs[16] * aP.x() * aP.z() * aP.z()
            + mCoeffs[2]  * aP.x()
            + mCoeffs[7]  * aP.y() * aP.y()
            + mCoeffs[12] * aP.y() * aP.x() * aP.x()
            + mCoeffs[17] * aP.y() * aP.y() * aP.z()
            + mCoeffs[3]  * aP.z()
            + mCoeffs[8]  * aP.x() * aP.x()
            + mCoeffs[13] * aP.y() * aP.z() * aP.z()
            + mCoeffs[18] * aP.x() * aP.x() * aP.z()
            + mCoeffs[4]  * aP.y() * aP.x()
            + mCoeffs[9]  * aP.z() * aP.z()
            + mCoeffs[14] * aP.y() * aP.y() * aP.x()
            + mCoeffs[19] * aP.z() * aP.z() * aP.z();

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

double cRatioPolyn::Val(const cPt3dr &aP) const
{
    return mNumPoly.Val(aP) / mDenPoly.Val(aP);
}

void cRatioPolyn::Show()
{
    StdOut() << "Numerator:" << std::endl;
    mNumPoly.Show();

    StdOut() << "Denumerator:" << std::endl;
    mDenPoly.Show();
}

/* =============================================== */
/*                                                 */
/*                 cRatioPolynXY                   */
/*                                                 */
/* =============================================== */

/**  A class containing two rational polynomials,
     each to predict one coordinate :
     mX(lat,lon) = j,     mY(lat,lon) = i or
     mX(j,i)     = lat,   mY(j,i)     = lon
*/

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

cPt2dr cRatioPolynXY::Val(const cPt3dr &aP) const
{
    cPt2dr aRes;

    aRes.x() = mX.Val(aP);
    aRes.y() = mY.Val(aP);

    return aRes;
}

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
        cDataRPC(const std::string&);
        void ReadXML(const std::string&);
        void Exe();

        cPt2dr GroundToImage(const cPt3dr&);
        cPt3dr ImageZToGround(const cPt2dr&,const double);



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
        cPt2dr NormIm(const cPt2dr &aP,bool);
        cPt3dr NormGround(const cPt3dr &aP,bool);
        double NormZ(const double,bool);

        double ReadXMLItem(const cSerialTree&,const std::string&);
        void   ReadXMLModel(const cSerialTree&,const std::string&,cRatioPolynXY *);
        void   ReadXMLNorms(const cSerialTree&);

        cRatioPolynXY * mDirectRPC; // rational polynomial for image to ground projection
        cRatioPolynXY * mInverseRPC; // rational polynomial for ground to image projection

        /// coordinate normalisation:  coord_norm = (coord - coord_offset) / coord_scale
        cPt2dr mImOffset; // line and sample offsets
        cPt2dr mImScale;  // line and sample scales

        cPt3dr mGroundOffset; // ground coordinate (e.g., lambda, phi, h) offets
        cPt3dr mGroundScale;  // ground coordinate (e.g., lambda, phi, h) scales

        std::string mNameRPC;
};

cDataRPC::cDataRPC(const std::string& aNameRPC) :
    mDirectRPC(nullptr),
    mInverseRPC(nullptr),
    mNameRPC(aNameRPC)
{}

cPt2dr cDataRPC::NormIm(const cPt2dr &aP,bool Direct)
{
    return  Direct ?
            DivCByC(aP - mImOffset,mImScale) : MulCByC(aP,mImScale) + mImOffset;
}

cPt3dr cDataRPC::NormGround(const cPt3dr &aP,bool Direct)
{
    return  Direct ?
            DivCByC(aP - mGroundOffset,mGroundScale) : MulCByC(aP,mGroundScale) + mGroundOffset;
}

double cDataRPC::NormZ(const double aZ,bool Direct)
{
    return Direct ?
           (aZ - mGroundOffset.z())/mGroundScale.z() : aZ*mGroundScale.z() + mGroundOffset.z();
}

cPt2dr cDataRPC::GroundToImage(const cPt3dr& aP)
{

    cPt3dr aPN  = NormGround(aP,true); // ground normalised
    cPt2dr aRes = NormIm(mInverseRPC->Val(aPN),false); // image unnormalised

    return aRes;
}

cPt3dr cDataRPC::ImageZToGround(const cPt2dr& aPIm,const double aZ)
{

    cPt2dr aPImN = NormIm(aPIm,true); // image normalised
    double aZN = NormZ(aZ,true); // norm Z
    cPt2dr aPGrN = mDirectRPC->Val(cPt3dr(aPImN.x(),aPImN.y(),aZN)); // ground normalised
    cPt3dr aRes = NormGround( cPt3dr(aPGrN.x(),aPGrN.y(),aZN) ,false); // ground unnormalised

    return aRes;
}

void cDataRPC::Exe()
{
    ReadXML(mNameRPC);
}

void cDataRPC::ReadXMLModel(const cSerialTree& aTree,const std::string& aPrefix,cRatioPolynXY * aModel)
{
    const cSerialTree * aDirect = aTree.GetUniqueDescFromName(aPrefix);

    aModel->Y().NumPoly().Initialise(*aDirect,"SAMP_NUM_COEFF_");
    aModel->Y().DenPoly().Initialise(*aDirect,"SAMP_DEN_COEFF_");

    aModel->X().NumPoly().Initialise(*aDirect,"LINE_NUM_COEFF_");
    aModel->X().DenPoly().Initialise(*aDirect,"LINE_DEN_COEFF_");

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

/** Bench the reprojection functions */

void TestRPCProjections(const std::string& aNameRPC1)
{
    // read
    cDataRPC aCam1(aNameRPC1);
    aCam1.Exe();

    // (latitude,longtitude,h)
    double aZ = 1188.1484901208684;
    cPt3dr aPtGround(20.7369382477,16.6170106276,aZ);

    // (j,i) ~ (Y, X) ~ (LINE,SAMPLE)
    cPt2dr aPtIm(5769.51863767362192,6188.93377727110783);

    StdOut() << "===== Ground to image" << std::endl;
    cPt2dr aPtImPred = aCam1.GroundToImage(aPtGround);
    StdOut() << aPtIm << " =? " << aPtImPred << ", " << std::endl;


    StdOut() << "===== Image to ground" << std::endl;
    cPt3dr aPtGroundPred = aCam1.ImageZToGround(aPtIm,aZ);

    StdOut() << aPtGround << " =? " << aPtGroundPred << std::endl;

}

void TestDataRPCReasXML(const std::string& aNameFile)
{
    cDataRPC aRPC(aNameFile);
    aRPC.Exe();
    aRPC.Show();
}



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
    //TestDataRPCReasXML(mNameSensorIn);
    TestRPCProjections(mNameSensorIn);

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
