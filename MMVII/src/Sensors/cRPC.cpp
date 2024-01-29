#include "StdAfx.h"
#include "V1VII.h"
#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "../Serial/Serial.h"

namespace MMVII
{

typedef double  tRPCCoeff[20];
class cDataRPC;
class cRatioPolynXY;
class cRPC_RatioPolyn;
class cRPC_Polyn ;


/**  A class containing a polynomial with RPC convention    */

class cRPC_Polyn : public cDataMapping<tREAL8,3,1>
{
    public:
        cRPC_Polyn(){};

        void Initialise(const cSerialTree &,const std::string&);
        void Show();

        const double * Coeffs() const {return mCoeffs;}
        double * Coeffs() {return mCoeffs;}
        double Val(const tRPCCoeff & aCoeffCub) const;

        static void  FillCubicCoeff(tRPCCoeff & aVCoeffs,const cPt3dr &) ;
	// cPt1dr Value(const cPt3dr &) const override; // Make the object a mapping, in case it is usefull
    private:
        double Val(const cPt3dr &) const;

        tRPCCoeff mCoeffs;
};

/**  A class containing a rational polynomial  */


class cRPC_RatioPolyn
{
    public:
        cRPC_RatioPolyn(){};

        const cRPC_Polyn & NumPoly() const {return mNumPoly;} // read access
        cRPC_Polyn & NumPoly() {return mNumPoly;}             // write access

        const cRPC_Polyn & DenPoly() const {return mDenPoly;} // read access
        cRPC_Polyn & DenPoly() {return mDenPoly;}             // write access

        void Show();
        double Val(const tRPCCoeff &) const;

    private:
        double Val(const cPt3dr &) const;
        cRPC_Polyn mNumPoly; // numerator polynomial
        cRPC_Polyn mDenPoly; // denominator polynomial


};

/**  A class containing two rational polynomials,
     each to predict one coordinate :
     mX(lat,lon) = j,     mY(lat,lon) = i or
     mX(j,i)     = lat,   mY(j,i)     = lon
*/

class cRatioPolynXY
{
    public:
        cRatioPolynXY(){};

        const cRPC_RatioPolyn & X() const {return mX;}
        cRPC_RatioPolyn & X() {return mX;}

        const cRPC_RatioPolyn & Y() const {return mY;}
        cRPC_RatioPolyn & Y() {return mY;}

        cPt2dr Val(const cPt3dr &) const;
        void   Show();

    private:
        cRPC_RatioPolyn mX;
        cRPC_RatioPolyn mY;

};


/**  A class containing RPCs  */

class cDataRPC // : public cSensorImage
{
    public:

         tSeg3dr  Image2Bundle(const cPt2dr &) const ;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const ;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const ;

         /// Will be implemanted later, for example using Genrate point 3D/2D -> Recompute RPC by least square
         cSensorImage * SensorChangSys(cDataInvertibleMapping<tREAL8,3> &) const ;
         ///  
         const cPixelDomain & PixelDomain() const ;

         // These 2 method are not meaningfull for RPC,  probably will have to redesign
         // the base class cSensorImage, waiting for that define fake functions
         ///  not implemented, error
         cPt3dr Ground2ImageAndDepth(const cPt3dr &) const ;
         /// not implemented, error
         cPt3dr ImageAndDepth2Ground(const cPt3dr &) const ;





        cDataRPC(const std::string&);
        void Dimap_ReadXML_Glob(const cSerialTree&);

        cPt3dr ImageZToGround(const cPt2dr&,const double) const;

        ~cDataRPC();

        void Show();

        const cRatioPolynXY& DirectRPC() {return *mDirectRPC;}
        const cRatioPolynXY& InverseRPC() {return *mInverseRPC;}

    private:
        const cPt2dr & ImOffset() const {return mImOffset;}
        cPt2dr & ImOffset() {return mImOffset;}

        const cPt2dr & ImScale() const {return mImScale;}
        cPt2dr & ImScale() {return mImScale;}

        const cPt3dr & GroundOffset() const {return mGroundOffset;}
        cPt3dr & GroundOffset() {return mGroundOffset;}

        const cPt3dr & GroundScale() const {return mGroundScale;}
        cPt3dr & GroundScale() {return mGroundScale;}

    private:


        cDataRPC(const cDataRPC&) = delete;
        void operator = (const cDataRPC&) = delete;


        cPt2dr NormIm(const cPt2dr &aP,bool) const;
        cPt3dr NormGround(const cPt3dr &aP,bool) const;
        double NormZ(const double,bool) const;

        double ReadXMLItem(const cSerialTree&,const std::string&);
        void   Dimap_ReadXMLModel(const cSerialTree&,const std::string&,cRatioPolynXY *);
        void   Dimap_ReadXMLNorms(const cSerialTree&);

        cRatioPolynXY * mDirectRPC; // rational polynomial for image to ground projection
        cRatioPolynXY * mInverseRPC; // rational polynomial for ground to image projection

        cPt2dr  IO_PtIm(const cPt2dr &) const;  // Id or SwapXY, depending mSwapIJImage
        cPt3dr  IO_PtGr(const cPt3dr &) const;  // Id or SwapXY, depending mSwapXYGround

        /// coordinate normalisation:  coord_norm = (coord - coord_offset) / coord_scale
        cPt2dr mImOffset; // line and sample offsets
        cPt2dr mImScale;  // line and sample scales

        cPt3dr mGroundOffset; // ground coordinate (e.g., lambda, phi, h) offets
        cPt3dr mGroundScale;  // ground coordinate (e.g., lambda, phi, h) scales

        std::string mNameRPC;

        // Internally the cDataRPC use dimap convention "Lat,long,H" and  "Line,Col";
        //  while MM-V2/V1  use XYZ direct (i.e Long,Lat,H) and    col,line
        //  Not sur what we will do in the future, so we 
        bool  mSwapXYGround;
        bool  mSwapIJImage;
        //  For Image 2 Bundle, we need to know how we generate 
        tREAL8 mAmplZB;
        //  cPixelDomain
        cDataPixelDomain  mDataPixDomain;
};



/* =============================================== */
/*                                                 */
/*                 cRPC_Polyn                      */
/*                                                 */
/* =============================================== */


void cRPC_Polyn::Initialise(const cSerialTree & aData,const std::string& aPrefix)
{
	// StdOut() << "cRPC_Polyn::InitialisecRPC_Polyn::Initialise  \n"; getchar();
    for (int aK=1; aK<21; aK++)
    {
        const cSerialTree * aItem = aData.GetUniqueDescFromName(aPrefix+ToString(aK));
        mCoeffs[aK-1] =  std::stod(aItem->UniqueSon().Value()); // coeffs
    }
}

double cRPC_Polyn::Val(const cPt3dr& aP) const
{
     static tRPCCoeff aCoeff;
     FillCubicCoeff(aCoeff,aP);
     return Val(aCoeff);

/*
    "Old" method, keep track
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
   */
}

double cRPC_Polyn::Val(const tRPCCoeff & aCoeffCub) const
{
   // to see if faster than for(int ...) 
   return   mCoeffs[0] *aCoeffCub[0]   + mCoeffs[1] *aCoeffCub[1]  + mCoeffs[2] *aCoeffCub[2]  + mCoeffs[3] *aCoeffCub[3]
          + mCoeffs[4] *aCoeffCub[4]   + mCoeffs[5] *aCoeffCub[5]  + mCoeffs[6] *aCoeffCub[6]  + mCoeffs[7] *aCoeffCub[7]
          + mCoeffs[8] *aCoeffCub[8]   + mCoeffs[9] *aCoeffCub[9]  + mCoeffs[10]*aCoeffCub[10] + mCoeffs[11]*aCoeffCub[11]
          + mCoeffs[12] *aCoeffCub[12] + mCoeffs[13]*aCoeffCub[13] + mCoeffs[14]*aCoeffCub[14] + mCoeffs[15]*aCoeffCub[15]
          + mCoeffs[16] *aCoeffCub[16] + mCoeffs[17]*aCoeffCub[17] + mCoeffs[18]*aCoeffCub[18] + mCoeffs[19]*aCoeffCub[19]
    ;
}

void  cRPC_Polyn::FillCubicCoeff(tRPCCoeff & aVCoeffs,const cPt3dr & aP) 
{
     aVCoeffs[0] = 1.0;
     aVCoeffs[5]  = aP.y() * aP.z();
     aVCoeffs[10] = aP.x() * aP.y() * aP.z();
     aVCoeffs[15] = aP.x() * aP.x() * aP.x();
     aVCoeffs[1]  = aP.y();
     aVCoeffs[6]  = aP.x() * aP.z();
     aVCoeffs[11] = aP.y() * aP.y() * aP.y();
     aVCoeffs[16] = aP.x() * aP.z() * aP.z();
     aVCoeffs[2]  = aP.x();
     aVCoeffs[7]  = aP.y() * aP.y();
     aVCoeffs[12] = aP.y() * aP.x() * aP.x();
     aVCoeffs[17] = aP.y() * aP.y() * aP.z();
     aVCoeffs[3]  = aP.z();
     aVCoeffs[8]  = aP.x() * aP.x();
     aVCoeffs[13] = aP.y() * aP.z() * aP.z();
     aVCoeffs[18] = aP.x() * aP.x() * aP.z();
     aVCoeffs[4]  = aP.y() * aP.x();
     aVCoeffs[9]  = aP.z() * aP.z();
     aVCoeffs[14] = aP.y() * aP.y() * aP.x();
     aVCoeffs[19] = aP.z() * aP.z() * aP.z();
}

	// cPt1dr Value(const cPt3dr &) const override;

void cRPC_Polyn::Show()
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


double cRPC_RatioPolyn::Val(const tRPCCoeff &aCoeff) const
{
    return mNumPoly.Val(aCoeff) / mDenPoly.Val(aCoeff);
}

double cRPC_RatioPolyn::Val(const cPt3dr &aP) const
{
    static tRPCCoeff aBuf;
    cRPC_Polyn::FillCubicCoeff(aBuf,aP);

    return Val(aBuf); 
    // return mNumPoly.Val(aBuf) / mDenPoly.Val(aBuf);
    // return mNumPoly.Val(aP) / mDenPoly.Val(aP);
}

void cRPC_RatioPolyn::Show()
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


cPt2dr cRatioPolynXY::Val(const cPt3dr &aP) const
{
    static tRPCCoeff aBuf;
    cRPC_Polyn::FillCubicCoeff(aBuf,aP);

    return cPt2dr(mX.Val(aBuf),mY.Val(aBuf));

    /*
    cPt2dr aRes;
    aRes.x() = mX.Val(aP);
    aRes.y() = mY.Val(aP);
    return aRes;
    */
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

     // ====================================================
     //     Construction & Destruction  & Show
     // ====================================================

cDataRPC::cDataRPC(const std::string& aNameRPC) :
    mDirectRPC       (nullptr),
    mInverseRPC      (nullptr),
    mNameRPC         (aNameRPC),
    mSwapXYGround    (true),
    mSwapIJImage     (true),
    mAmplZB          (1.0),
    mDataPixDomain   (cPt2di(1,1))
{
    //  Is it a xml file ?
    if (UCaseEqual(LastPostfix(aNameRPC),"xml"))
    {
        cSerialFileParser * aSFP = cSerialFileParser::Alloc(aNameRPC,eTypeSerial::exml);
        cSerialTree  aTree(*aSFP);
        // Is it a dimap tree
        if (!aTree.GetAllDescFromName("Dimap_Document").empty())
        {
           // if yes read the dimap-xml-tree and return
           Dimap_ReadXML_Glob(aTree);
           delete aSFP;
           return;
        }
        MMVII_UnclasseUsEr("RPC : Dont handle this xml file, for " + aNameRPC);
    }
    else
    {
        MMVII_UnclasseUsEr("RPC : Dont handle postfix for "+aNameRPC);
    }
}

cDataRPC::~cDataRPC()
{
    delete mDirectRPC;
    delete mInverseRPC;
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

     // ======================  Dimap creation ===========================

void cDataRPC::Dimap_ReadXMLModel(const cSerialTree& aTree,const std::string& aPrefix,cRatioPolynXY * aModel)
{
    const cSerialTree * aDirect = aTree.GetUniqueDescFromName(aPrefix);

    aModel->Y().NumPoly().Initialise(*aDirect,"SAMP_NUM_COEFF_");
    aModel->Y().DenPoly().Initialise(*aDirect,"SAMP_DEN_COEFF_");

    aModel->X().NumPoly().Initialise(*aDirect,"LINE_NUM_COEFF_");
    aModel->X().DenPoly().Initialise(*aDirect,"LINE_DEN_COEFF_");

}

void cDataRPC::Dimap_ReadXMLNorms(const cSerialTree& aTree)
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

    int anX = round_ni(ReadXMLItem(*aData,"LAST_COL"));
    int anY = round_ni(ReadXMLItem(*aData,"LAST_ROW"));
    mDataPixDomain =  cDataPixelDomain(cPt2di(anX,anY));
}

double cDataRPC::ReadXMLItem(const cSerialTree & aData,const std::string& aPrefix)
{
    return std::stod(aData.GetUniqueDescFromName(aPrefix)->UniqueSon().Value());
}

void cDataRPC::Dimap_ReadXML_Glob(const cSerialTree & aTree)
{
    // read the direct model
    mDirectRPC = new cRatioPolynXY();
    Dimap_ReadXMLModel(aTree,"Direct_Model",mDirectRPC);


    // read the inverse model
    mInverseRPC = new cRatioPolynXY();
    Dimap_ReadXMLModel(aTree,"Inverse_Model",mInverseRPC);


    // read the normalisation data (offset, scales)
    Dimap_ReadXMLNorms(aTree);
}

     // ====================================================
     //     Normalisation 
     // ====================================================

cPt2dr cDataRPC::NormIm(const cPt2dr &aP,bool Direct) const
{
    return  Direct ?
            DivCByC(aP - mImOffset,mImScale) : MulCByC(aP,mImScale) + mImOffset;
}

cPt3dr cDataRPC::NormGround(const cPt3dr &aP,bool Direct) const
{
    return  Direct ?
            DivCByC(aP - mGroundOffset,mGroundScale) : MulCByC(aP,mGroundScale) + mGroundOffset;
}

double cDataRPC::NormZ(const double aZ,bool Direct) const
{
    return Direct ?
           (aZ - mGroundOffset.z())/mGroundScale.z() : aZ*mGroundScale.z() + mGroundOffset.z();
}

     // ====================================================
     //     Image <-> Ground  transformation
     // ====================================================

cPt2dr cDataRPC::IO_PtIm(const cPt2dr&aPt) const {return mSwapIJImage?cPt2dr(aPt.y(),aPt.x()):aPt;}  
cPt3dr cDataRPC::IO_PtGr(const cPt3dr&aPt) const {return mSwapXYGround?cPt3dr(aPt.y(),aPt.x(),aPt.z()):aPt;} 

cPt2dr cDataRPC::Ground2Image(const cPt3dr& aP) const
{
    cPt3dr aPN  = NormGround(IO_PtGr(aP),true); // ground normalised
    cPt2dr aRes = NormIm(mInverseRPC->Val(aPN),false); // image unnormalised

    return IO_PtIm(aRes);
}

cPt3dr cDataRPC::ImageZToGround(const cPt2dr& aPIm,const double aZ) const
{

    cPt2dr aPImN = NormIm(IO_PtIm(aPIm),true); // image normalised
    double aZN = NormZ(aZ,true); // norm Z
    cPt2dr aPGrN = mDirectRPC->Val(cPt3dr(aPImN.x(),aPImN.y(),aZN)); // ground normalised
    cPt3dr aRes = NormGround( cPt3dr(aPGrN.x(),aPGrN.y(),aZN) ,false); // ground unnormalised

    return IO_PtGr(aRes);
}

cPt3dr cDataRPC::ImageAndZ2Ground(const cPt3dr& aP) const
{
    return ImageZToGround(cPt2dr(aP.x(),aP.y()),aP.z());
}

tSeg3dr  cDataRPC::Image2Bundle(const cPt2dr & aPtIm) const 
{
     tREAL8  aZ0 = mGroundOffset.z() - mAmplZB;
     tREAL8  aZ1 = mGroundOffset.z() + mAmplZB;
      
     cPt3dr  aPGr0 = ImageZToGround(aPtIm,aZ0);
     cPt3dr  aPGr1 = ImageZToGround(aPtIm,aZ1);

     return tSeg3dr(aPGr0,aPGr1);
}

     // ====================================================
     //     Not implemanted (not yet or never)
     // ====================================================

cPt3dr cDataRPC::Ground2ImageAndDepth(const cPt3dr &) const 
{
    MMVII_INTERNAL_ERROR("No cDataRPC::Ground2ImageAndDepth");
    return cPt3dr::Dummy();
}

cPt3dr cDataRPC::ImageAndDepth2Ground(const cPt3dr &) const 
{
    MMVII_INTERNAL_ERROR("No cDataRPC::ImageAndDepth2Ground");
    return cPt3dr::Dummy();
}

cSensorImage * cDataRPC::SensorChangSys(cDataInvertibleMapping<tREAL8,3> &) const 
{
    MMVII_INTERNAL_ERROR("cDataRPC::SensorChangSys not yet implemanted");
    return nullptr;
}

/* =============================================== */
/*                                                 */
/*                 cDataRPC                        */
/*                                                 */
/* =============================================== */




void Test2RPCProjections(const std::string& aNameIm)
{
    // cPhotogrammetricProject(cMMVII_Appli &);
    // int InitStandAloneAppli(const cSpecMMVII_Appli & aSpec, int argc, char*argv[]);


}

/** Bench the reprojection functions */

void TestRPCProjections(const std::string& aNameRPC1)
{
    // read
    cDataRPC aCam1(aNameRPC1);
    // aCam1.Exe();

    // (latitude,longtitude,h)
    double aZ = 1188.1484901208684;
    cPt3dr aPtGround(20.7369382477,16.6170106276,aZ);

    // (j,i) ~ (Y, X) ~ (LINE,SAMPLE)
    cPt2dr aPtIm(5769.51863767362192,6188.93377727110783);

    StdOut() << "===== Ground to image" << std::endl;
    cPt2dr aPtImPred = aCam1.Ground2Image(aPtGround);
    StdOut() << aPtIm << " =? " << aPtImPred << ", " << std::endl;


    StdOut() << "===== Image to ground" << std::endl;
    cPt3dr aPtGroundPred = aCam1.ImageZToGround(aPtIm,aZ);

    StdOut() << aPtGround << " =? " << aPtGroundPred << std::endl;

}

void TestDataRPCReasXML(const std::string& aNameFile)
{
    cDataRPC aRPC(aNameFile);
    // aRPC.Exe();
    aRPC.Show();
}

/* =============================================== */
/*                                                 */
/*                 cAppliTestImportSensors         */
/*                                                 */
/* =============================================== */

/**  A basic application for  */

class cAppliTestImportSensors : public cMMVII_Appli
{
     public :

        cAppliTestImportSensors(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        cPhotogrammetricProject  mPhProj;
        std::string              mNameImage;
        std::string              mNameRPC;
        bool                     mShowDetail;

};


cAppliTestImportSensors::cAppliTestImportSensors(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mShowDetail  (false)
{
}

cCollecSpecArg2007 & cAppliTestImportSensors::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             << Arg2007(mNameImage,"Name of input RPC", {eTA2007::FileDirProj})
             << Arg2007(mNameRPC,"Name of input RPC", {eTA2007::Orient})
             << mPhProj.DPPointsMeasures().ArgDirInMand()
      ;
}

cCollecSpecArg2007 & cAppliTestImportSensors::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << AOpt2007(mShowDetail,"ShowD","Show detail",{eTA2007::HDV})
            ;
}

int cAppliTestImportSensors::Exe()
{
    mPhProj.FinishInit();

    cSetMesImGCP aSetMes;
    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,mNameImage);

    cDataRPC aDataRPC(mNameRPC);

    cSet2D3D aSetM23;
    aSetMes.ExtractMes1Im(aSetM23,mNameImage);

    tREAL8 aSomCheckIm = 0.0;
    tREAL8 aSomConsistIm = 0.0;
    tREAL8 aSomConsistGr = 0.0;

    for (const auto & aPair : aSetM23.Pairs())
    {
         cPt3dr  aPGr = aPair.mP3;
         tREAL8  aZ   = aPGr.z();
         cPt2dr  aPIm = aDataRPC.Ground2Image(aPGr);

         cPt3dr  aPGr2 =  aDataRPC.ImageZToGround(aPIm,aZ);
         cPt2dr  aPIm2 = aDataRPC.Ground2Image(aPGr2);

         aSomCheckIm +=  Norm2(aPIm  - aPair.mP2);
         aSomConsistIm +=  Norm2(aPIm-aPIm2);
         aSomConsistGr +=  Norm2(aPGr-aPGr2);

         if (mShowDetail) 
         {
             StdOut()  << "ImGT=" <<  aPIm  - aPair.mP2
                       << "  GroundConsist=" << aPGr2-aPGr
                       << "  ImConsist=" << aPIm-aPIm2
                       << "\n";
         }
/*
         cPt3dr aPtGr3 = aDataRPC.ImageZToGround(aPIm2,aPtGr1.z());
         cPt2dr aPIm3  = aDataRPC.Ground2Image(aPtGr3);
         StdOut() << "  -- PGR =" <<  aPtGr3  << " DeltaGr=" <<   aPtGr3-aPtGr2 << " DeltaIm=" << aPIm2-aPIm3<< "\n";
*/

    }
    aSomCheckIm /= aSetM23.NbPair();
    aSomConsistIm  /= aSetM23.NbPair();
    aSomConsistGr  /= aSetM23.NbPair();

    StdOut()  << "  CheckIm=" <<  aSomCheckIm << "\n";
    StdOut()  << "  ConsistIm=" <<  aSomConsistIm << "\n";
    StdOut()  << "  ConsistGr=" <<  aSomConsistGr << "\n";

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestImportSensors(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliTestImportSensors(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecTestImportSensors
(
     "TestImportSensor",
      Alloc_TestImportSensors,
      "Test orientation functions with ground truth 2D/3D correspondance",
      {eApF::Ori},
      {eApDT::Ori,eApDT::GCP},
      {eApDT::Console},
      __FILE__
);


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
