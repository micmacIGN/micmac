#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "cExternalSensor.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_util_tpl.h"

namespace MMVII
{

/* =============================================== */

static constexpr int TheNbRPCoeff = 20;
typedef double  tRPCCoeff[TheNbRPCoeff];
class cRPCSens;
class cRatioPolynXY;
class cRPC_RatioPolyn;
class cRPC_Polyn ;


/**  A class containing a polynomial with RPC convention    */

class cRPC_Polyn : public cDataMapping<tREAL8,3,1>
{
    public:
        cRPC_Polyn(){};

        bool Initialise(const cSerialTree &,const std::string&);
        void VectInitialise(const cSerialTree &,const std::vector<std::string>&);
        void Show();

        const double * Coeffs() const {return mCoeffs;}
        double * Coeffs() {return mCoeffs;}
        double Val(const tRPCCoeff & aCoeffCub) const;

        static void  FillCubicCoeff(tRPCCoeff & aVCoeffs,const cPt3dr &) ;
	// cPt1dr Value(const cPt3dr &) const override; // Make the object a mapping, in case it is usefull
	//
	void PushCoeffs(std::vector<tREAL8>&) const;
	void SetCoeffs(const std::vector<tREAL8>&,size_t aK0);


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

	void PushCoeffs(std::vector<tREAL8>&) const;
	void SetCoeffs(const std::vector<tREAL8>&);
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

	void PushCoeffs(std::vector<tREAL8>&) const;

	//  Recompute a new RPC using correspondance
	void  InitFromSamples(const std::vector<cPt3dr> & aVIn,const std::vector<cPt3dr> & aVOut);
    private:
        cRPC_RatioPolyn mX;
        cRPC_RatioPolyn mY;

};


/**  A class containing RPCs  */
class cRPCSens : public cSensorImage
{
    public:

         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;
        /// Epsilon-coordinate, taking into account the heterogeneity
	cPt3dr  EpsDiffGround2Im(const cPt3dr & aPt) const override;

	 /// Compute analytical differential using gen-code
	 tProjImAndGrad  DiffGround2Im(const cPt3dr &) const override;


	/// Indicate how much a point belongs to sensor visibilty domain
         double DegreeVisibility(const cPt3dr &) const  override;
	 bool  HasIntervalZ()  const override;
         cPt2dr GetIntervalZ() const override;

         ///  
         const cPixelDomain & PixelDomain() const override;
         std::string  V_PrefixName() const  override;

         cRPCSens(const std::string& aNameImage);
	 void InitFromFile(const cAnalyseTSOF &);

         void Dimap_ReadXML_Glob(const cSerialTree&);

        cPt3dr ImageZToGround(const cPt2dr&,const double) const;

        ~cRPCSens();

        void Show();

        const cRatioPolynXY& DirectRPC() {return *mDirectRPC;}
        const cRatioPolynXY& InverseRPC() {return *mInverseRPC;}

	cPt3dr  PseudoCenterOfProj() const override;

	cRPCSens * RPCChangSys(cDataInvertibleMapping<tREAL8,3> &) const ;

    private:
        const cPt2dr & ImOffset() const {return mImOffset;}
        cPt2dr & ImOffset() {return mImOffset;}

        const cPt2dr & ImScale() const {return mImScale;}
        cPt2dr & ImScale() {return mImScale;}

        const cPt3dr & GroundOffset() const {return mGroundOffset;}
        cPt3dr & GroundOffset() {return mGroundOffset;}

        const cPt3dr & GroundScale() const {return mGroundScale;}
        cPt3dr & GroundScale() {return mGroundScale;}

	const cPt3dr * CenterOfFootPrint() const override;

    private:

         //  --------------- BEGIN NOT IMPLEMANTED -----------------------------------------------
 
         // standard declaration to forbid copy 
         cRPCSens(const cRPCSens&) = delete;
         void operator = (const cRPCSens&) = delete;

         // These  method are not meaningfull for RPC,  probably will have to redesign
         // the base class cSensorImage, waiting for that define fake functions
         ///  not implemented, error

         //  --------------- END  NOT IMPLEMANTED -----------------------------------------------


        cPt2dr NormIm(const cPt2dr &aP,bool) const;
        cPt3dr NormGround(const cPt3dr &aP,bool) const;
        double NormZ(const double,bool) const;

	std::string ReadXmlItem(const cSerialTree&,const std::string&);
        double ReadRealXmlItem(const cSerialTree&,const std::string&);

        void   Dimap_ReadXMLModel(const cSerialTree&,const std::vector<std::string>&,cRatioPolynXY *);
        void   Dimap_ReadXMLNorms(const cSerialTree&);

        cPt2dr  IO_PtIm(const cPt2dr &) const;  // Id or SwapXY, depending mSwapIJImage
        cPt3dr  IO_PtGr(const cPt3dr &) const;  // Id or SwapXY, depending mSwapXYGround

        cRatioPolynXY * mDirectRPC; // rational polynomial for image to ground projection
        cRatioPolynXY * mInverseRPC; // rational polynomial for ground to image projection


        /// coordinate normalisation:  coord_norm = (coord - coord_offset) / coord_scale
        cPt2dr mImOffset; // line and sample offsets
        cPt2dr mImScale;  // line and sample scales
        // Just Add Z to make it more homogeneaous with mGroundOffset/mGroundScale
        cPt3dr m3DImOffset; // line and sample offsets
        cPt3dr m3DImScale;  // line and sample scales

        cPt3dr mGroundOffset; // ground coordinate (e.g., lambda, phi, h) offets
        cPt3dr mGroundScale;  // ground coordinate (e.g., lambda, phi, h) scales
	cPt3dr mCenterOfFootPrint;  // +- mGroundOffset but in MM convention

        std::string mNameRPC;

        // Internally the cRPCSens use dimap convention "Lat,long,H" and  "Line,Col";
        //  while MM-V2/V1  use XYZ direct (i.e Long,Lat,H) and    col,line
        //  Not sur what we will do in the future, so we 
        bool  mSwapXYGround;
        bool  mSwapIJImage;
        //  For Image 2 Bundle, we need to know how we generate 
        tREAL8 mAmplZB;
        //  cPixelDomain
        cDataPixelDomain  mDataPixelDomain;
        cPixelDomain      mPixelDomain;
	cBox3dr           mBoxGround;

	cPt3dr            mEpsCoord; /// Pre-compute the "right" espislon value
};



/* =============================================== */
/*                                                 */
/*                 cRPC_Polyn                      */
/*                                                 */
/* =============================================== */


bool cRPC_Polyn::Initialise(const cSerialTree & aData,const std::string& aPrefix)
{
	// StdOut() << "cRPC_Polyn::InitialisecRPC_Polyn::Initialise  \n"; getchar();
    for (int aK=1; aK<21; aK++)
    {
        const cSerialTree * aItem = aData.GetUniqueDescFromName(aPrefix+std::to_string(aK),true);
        if (aItem==nullptr)
           return false;
        mCoeffs[aK-1] =  std::stod(aItem->UniqueSon().Value()); // coeffs
    }
    return true;
}

void cRPC_Polyn::VectInitialise(const cSerialTree & aData,const std::vector<std::string>& aVecPrefix)
{
    // Try diffent prefixes of tags
    for (const auto & aPrefix : aVecPrefix)
    {
        if (Initialise(aData,aPrefix))
           return;
    }
    MMVII_INTERNAL_ERROR("cRPCSens::VectInitialise : could not get any tag for tags " + aVecPrefix.at(0));
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

void cRPC_Polyn::PushCoeffs(std::vector<tREAL8>& aVObs) const
{
    for (size_t aK=0 ; aK<TheNbRPCoeff ; aK++)
        aVObs.push_back(mCoeffs[aK]);
}

void cRPC_Polyn::SetCoeffs(const std::vector<tREAL8>& aVC,size_t aK0)
{
    for (size_t aK=0 ; aK<TheNbRPCoeff ; aK++)
        mCoeffs[aK] =  aVC[aK+aK0];
}

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

void cRPC_RatioPolyn::PushCoeffs(std::vector<tREAL8>& aVObs) const
{
    mNumPoly.PushCoeffs(aVObs);
    mDenPoly.PushCoeffs(aVObs);
}

void cRPC_RatioPolyn::SetCoeffs(const std::vector<tREAL8>& aVC)
{
   mNumPoly.SetCoeffs(aVC,0);
   mDenPoly.SetCoeffs(aVC,TheNbRPCoeff);
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

void cRatioPolynXY::PushCoeffs(std::vector<tREAL8>& aVObs) const
{
    mX.PushCoeffs(aVObs);
    mY.PushCoeffs(aVObs);
}

     // ====================================================
     //     Construction & Destruction  & Show
     // ====================================================

cRPCSens::cRPCSens(const std::string& aNameImage) :
    cSensorImage       (aNameImage),
    mDirectRPC         (nullptr),
    mInverseRPC        (nullptr),
    mNameRPC           (MMVII_NONE),
    mSwapXYGround      (true),
    mSwapIJImage       (true),
    mAmplZB            (1.0),
    mDataPixelDomain   (cPt2di(1,1)),  // No default constructor
    mPixelDomain       (&mDataPixelDomain),
    mBoxGround         (cBox3dr::Empty())  // Empty box because no default init
{
    ///  For now assume RPC is WGS84Degree always, see later if we change that
    SetCoordinateSystem(E2Str(eSysCoGeo::eWGS84Degrees));
}

void cRPCSens::InitFromFile(const cAnalyseTSOF & anAnalyse)
{
   mNameRPC = anAnalyse.mData.mNameFileInit;
   MMVII_INTERNAL_ASSERT_strong(anAnalyse.mData.mType==eTypeSensor::eRPC,"Sensor is not RPC in cRPCSens"); 
   if (anAnalyse.mData.mFormat== eFormatSensor::eDimap_RPC)
   {
        Dimap_ReadXML_Glob(*anAnalyse.mSTree);
   }

   mCenterOfFootPrint =  IO_PtGr(mGroundOffset);

   //  For now trust the scale
   cPt3dr aScGrV2 = IO_PtGr(mGroundScale);
   cPt2dr aScGrIm = IO_PtIm(mImScale);

   tREAL8 aNbPixel = 5.0;
   //  Make the Epsilon so that a dif of 1 Epislon correspond ~ to aNbPixel
   mEpsCoord.x() = (aScGrV2.x() / aScGrIm.x()) * aNbPixel;
   mEpsCoord.y() = (aScGrV2.y() / aScGrIm.y()) * aNbPixel;

   mEpsCoord.z() =  1.0 * aNbPixel; // very rough
				    //
   // StdOut() << "EPSILON : " << mEpsCoord << "\n";
}

cRPCSens::~cRPCSens()
{
    delete mDirectRPC;
    delete mInverseRPC;
    mDirectRPC = nullptr;
    mInverseRPC = nullptr;
}

void cRPCSens::Show()
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

std::string  cRPCSens::V_PrefixName() const  
{
	return "RPC";
}

void  cRatioPolynXY::InitFromSamples(const std::vector<cPt3dr> & aVIn,const std::vector<cPt3dr> & aVOut)
{
    cBox3dr aBoxIn = cTplBoxOfPts<tREAL8,3>::FromVect(aVIn).CurBox();
    cBox3dr aBoxOut = cTplBoxOfPts<tREAL8,3>::FromVect(aVOut).CurBox();

   // static void  FillCubicCoeff(tRPCCoeff & aVCoeffs,const cPt3dr &) ;
    for (auto IsX : {true,false})
    {
        cLeasSqtAA<tREAL8>   aSys(2*TheNbRPCoeff);
        std::vector<tREAL8>  aVCoeff(2*TheNbRPCoeff);

        tRPCCoeff aVPolXYZ;	
        tREAL8 aSomC2 = 0.0;
	size_t IndNumCste = 0;
        for (size_t aKPt=0; aKPt<aVIn.size() ; aKPt++)
        {
            cPt3dr aPNormIn  = aBoxIn.ToNormaliseCoord(aVIn.at(aKPt));
	    cRPC_Polyn::FillCubicCoeff(aVPolXYZ,aPNormIn);

            cPt3dr aPNormOut = aBoxOut.ToNormaliseCoord(aVOut.at(aKPt));
            tREAL8 aCoord = IsX ? aPNormOut.x() : aPNormOut.y() ;
         
	    //   Num Coef - Den * Coef * I =0    , N [0] = 0
	    for (int aKC=0 ; aKC<TheNbRPCoeff ; aKC++)
	    {
                 aSomC2 += Square(aVPolXYZ[aKC]);
                 aVCoeff[aKC] = aVPolXYZ[aKC];
                 aVCoeff[aKC+TheNbRPCoeff] = -aVPolXYZ[aKC] * aCoord;
	    }
	    tREAL8 aValCste = aVCoeff[IndNumCste];
	    aVCoeff[IndNumCste] = 0;
	    aSys.PublicAddObservation(1.0,aVCoeff,-aValCste);
        }
	aSys.AddObsFixVar(std::sqrt(aSomC2),IndNumCste,1.0);
	std::vector<tREAL8> aSol = aSys.Solve().ToStdVect();

        if (IsX)
           mX.SetCoeffs(aSol);
	else
           mY.SetCoeffs(aSol);
    }
		   
}

//  Not finisehd
cRPCSens * cRPCSens::RPCChangSys(cDataInvertibleMapping<tREAL8,3> & aMap) const 
{
    cRPCSens * aRes = new cRPCSens(NameImage());
    cSet2D3D aSet = SyntheticsCorresp3D2D(30,5,mBoxGround.P0().z(),mBoxGround.P1().z(),false,0.0);

    std::vector<cPt3dr> aVIm;
    std::vector<cPt3dr> aVGr;

    for (const auto & aPair : aSet.Pairs())
    {
	 cPt3dr  aPGr = aMap.Value(aPair.mP3); 
	 cPt3dr  aPIm = TP3z(aPair.mP2,aPGr.z());

	 aVIm.push_back(aPIm);
	 aVGr.push_back(aPGr);
    }

    // InitFromSamples(aVIm,aVGr);
    // InitFromSamples(aVGr,aVIm);

    return aRes;
}

     // ======================  Dimap creation ===========================

void cRPCSens::Dimap_ReadXMLModel(const cSerialTree& aTreeGlob,const std::vector<std::string> & aVecPrefix,cRatioPolynXY * aModel)
{
    // Try possible tags 
    for (const auto & aPrefix : aVecPrefix)
    {
         // try to get the tree of a given tag
         const cSerialTree * aTree = aTreeGlob.GetUniqueDescFromName(aPrefix,SVP::Yes);

         if (aTree) // it got it : go
         {
            // Again try different tag
            aModel->Y().NumPoly().VectInitialise(*aTree,{"SAMP_NUM_COEFF_","LON_NUM_COEFF_"});  // LON_NUM_COEFF_
            aModel->Y().DenPoly().VectInitialise(*aTree,{"SAMP_DEN_COEFF_","LON_DEN_COEFF_"});  // LON_DEN_COEFF_

            aModel->X().NumPoly().VectInitialise(*aTree,{"LINE_NUM_COEFF_","LAT_NUM_COEFF_"});
            aModel->X().DenPoly().VectInitialise(*aTree,{"LINE_DEN_COEFF_","LAT_DEN_COEFF_"});

            return;
         }
    }
    // if no tag was success, we have a problem ....
    MMVII_INTERNAL_ERROR("cRPCSens::Dimap_ReadXMLModel : could not get any tag for tags " + aVecPrefix.at(0));
}

void cRPCSens::Dimap_ReadXMLNorms(const cSerialTree& aTree)
{
    // This multiplier was added to correct in quick&dirty way the problem of 
    // different dimension in coordinates
    tREAL8 aMul = 1;  // Someting like 1e5 to 
    if (aMul!=1)
    {
       MMVII_DEV_WARNING("LAT_SCALELONG_SCALE");
    }

    const cSerialTree * aData = aTree.GetUniqueDescFromName("RFM_Validity");

    mGroundOffset.x() = ReadRealXmlItem(*aData,"LAT_OFF");
    mGroundOffset.y() = ReadRealXmlItem(*aData,"LONG_OFF");
    m3DImOffset.z() = mGroundOffset.z() = ReadRealXmlItem(*aData,"HEIGHT_OFF");

    mGroundScale.x() = ReadRealXmlItem(*aData,"LAT_SCALE") * aMul;   //  MULTIPLY
    mGroundScale.y() = ReadRealXmlItem(*aData,"LONG_SCALE") * aMul;   //  MULTIPLY
    m3DImScale.z() = mGroundScale.z() = ReadRealXmlItem(*aData,"HEIGHT_SCALE");

    m3DImScale.x() = mImScale.x() = ReadRealXmlItem(*aData,"LINE_SCALE");
    m3DImScale.y() = mImScale.y() = ReadRealXmlItem(*aData,"SAMP_SCALE");

    m3DImOffset.x() = mImOffset.x() = ReadRealXmlItem(*aData,"LINE_OFF");
    m3DImOffset.y() = mImOffset.y() = ReadRealXmlItem(*aData,"SAMP_OFF");

    int anX = round_ni(ReadRealXmlItem(*aData,"LAST_COL"));
    int anY = round_ni(ReadRealXmlItem(*aData,"LAST_ROW"));
    mDataPixelDomain =  cDataPixelDomain(cPt2di(anX,anY));

    // compute the validity of bounding box
    cPt3dr aP0Gr;
    aP0Gr.x() = ReadRealXmlItem(*aData,"FIRST_LAT");
    aP0Gr.y() = ReadRealXmlItem(*aData,"FIRST_LON");
    aP0Gr.z() =   mGroundOffset.z() -  mGroundScale.z();

    cPt3dr aP1Gr;
    aP1Gr.x() = ReadRealXmlItem(*aData,"LAST_LAT");
    aP1Gr.y() = ReadRealXmlItem(*aData,"LAST_LON");
    aP1Gr.z() =   mGroundOffset.z() +  mGroundScale.z();

    //  ===========   MULTIPLY  ============
    cPt3dr aMil = (aP1Gr+aP0Gr)/2.0;
    cPt3dr aAmpl = MulCByC((aP1Gr-aP0Gr),cPt3dr(aMul,aMul,1));
    aP0Gr = aMil-aAmpl;
    aP1Gr = aMil+aAmpl;


    // StdOut() << "BBBBBB " << aP0Gr << " " << aP1Gr << "\n";
    mBoxGround = cBox3dr(IO_PtGr(aP0Gr),IO_PtGr(aP1Gr));
}

bool  cRPCSens::HasIntervalZ()  const {return true;}
cPt2dr cRPCSens::GetIntervalZ() const 
{
	return cPt2dr(mBoxGround.P0().z(),mBoxGround.P1().z());
}

double cRPCSens::ReadRealXmlItem(const cSerialTree & aData,const std::string& aPrefix)
{
    return std::stod(ReadXmlItem(aData,aPrefix));
}

std::string cRPCSens::ReadXmlItem(const cSerialTree & aData,const std::string& aPrefix)
{
    return aData.GetUniqueDescFromName(aPrefix)->UniqueSon().Value();
}

void cRPCSens::Dimap_ReadXML_Glob(const cSerialTree & aTree)
{
	/*
{
      MMVII_DEV_WARNING("Dimap_ReadXML_GlobDimap_ReadXML_Glob  Image2Bundle");
      cMMVII_Ofs anOfs("toto.xml",eFileModeOut::CreateText);
      aTree.Xml_PrettyPrint(anOfs);
}
*/

    // read the direct model
    mDirectRPC = new cRatioPolynXY();
    Dimap_ReadXMLModel(aTree,{"Direct_Model","ImagetoGround_Values"},mDirectRPC);
    // ImagetoGround_Values


    // read the inverse model
    mInverseRPC = new cRatioPolynXY();
    Dimap_ReadXMLModel(aTree,{"Inverse_Model","GroundtoImage_Values"},mInverseRPC);


    // read the normalisation data (offset, scales)
    Dimap_ReadXMLNorms(aTree);
}



     // ====================================================
     //     Normalisation 
     // ====================================================

cPt2dr cRPCSens::NormIm(const cPt2dr &aP,bool Direct) const
{
    return  Direct ?
            DivCByC(aP - mImOffset,mImScale) : MulCByC(aP,mImScale) + mImOffset;
}

cPt3dr cRPCSens::NormGround(const cPt3dr &aP,bool Direct) const
{
    return  Direct ?
            DivCByC(aP - mGroundOffset,mGroundScale) : MulCByC(aP,mGroundScale) + mGroundOffset;
}

double cRPCSens::NormZ(const double aZ,bool Direct) const
{
    return Direct ?
           (aZ - mGroundOffset.z())/mGroundScale.z() : aZ*mGroundScale.z() + mGroundOffset.z();
}

     // ====================================================
     //     Image <-> Ground  transformation
     // ====================================================

// cPt2dr cRPCSens::IO_PtIm(const cPt2dr&aPt) const {return mSwapIJImage?cPt2dr(aPt.y(),aPt.x()):aPt;}  
// cPt3dr cRPCSens::IO_PtGr(const cPt3dr&aPt) const {return mSwapXYGround?cPt3dr(aPt.y(),aPt.x(),aPt.z()):aPt;} 

cPt2dr cRPCSens::IO_PtIm(const cPt2dr&aPt) const {return mSwapIJImage?PSymXY(aPt):aPt;}  
cPt3dr cRPCSens::IO_PtGr(const cPt3dr&aPt) const {return mSwapXYGround?PSymXY(aPt):aPt;} 

cPt2dr cRPCSens::Ground2Image(const cPt3dr& aP) const
{
    cPt3dr aPN  = NormGround(IO_PtGr(aP),true); // ground normalised
    cPt2dr aRes = NormIm(mInverseRPC->Val(aPN),false); // image unnormalised

    return IO_PtIm(aRes);
}

tProjImAndGrad  cRPCSens::DiffGround2Im(const cPt3dr & aP) const 
{
    // return     DiffG2IByFiniteDiff(aP);

    // extract the object given access to generated code
    static cCalculator<double> * aCalc = RPC_Proj(true /* With Derivative*/,1,true/*ReUse = its create only once*/);
    //  Vector of observation, use static for recycling memory
    static std::vector<double> aVObs;
    aVObs.clear();

    //  Push all the normalisation data in the vector of observation
    mGroundOffset.PushInStdVector(aVObs);   // Push mGroundOffset.x() mGroundOffset.y() ....
    mGroundScale.PushInStdVector(aVObs);    // Puxh mGroundScale.x() ...
    m3DImOffset.PushInStdVector(aVObs);
    m3DImScale.PushInStdVector(aVObs);

    mInverseRPC->PushCoeffs(aVObs);  // push the 80 coefficients of RPC

    //  execute the computation of value & derivatives
    aCalc->DoOneEval(IO_PtGr(aP).ToStdVector(),aVObs);

    // Un-mangle the data : 
    // Theoretically,  it is possible to make several computation (for parallezation), the first indexe
    // K0 indicate which computation is used, here K0=0 always
    //     ValComp(K0,I)  => extract the  Ith value , here 0 or 1 for  I or J
    //     DerComp(K0,I,V) => extract  the diffrential of I relatively to Vth value (here 0,1,2 for "x,y,z")
    //
    //  Also note IO_PtIm, IO_PtGr and mSwapIJImage due difference of convention between standard RPC and
    //  MicMac
    //

    tProjImAndGrad aRes;
    aRes.mPIJ = IO_PtIm(cPt2dr(aCalc->ValComp(0,0),aCalc->ValComp(0,1)));

    // tProjImAndGrad aDifF = DiffG2IByFiniteDiff(aP);

    aRes.mGradI = IO_PtGr(cPt3dr(aCalc->DerComp(0,0,0),aCalc->DerComp(0,0,1),aCalc->DerComp(0,0,2)));
    aRes.mGradJ = IO_PtGr(cPt3dr(aCalc->DerComp(0,1,0),aCalc->DerComp(0,1,1),aCalc->DerComp(0,1,2)));

    if (mSwapIJImage)  
       std::swap(aRes.mGradI,aRes.mGradJ);

    return aRes;
    /*

    StdOut() << "PGRound "<< aP << " PROJ " <<  aDifF.mPIJ << aRes.mPIJ << std::endl;
    StdOut() << " GRADI "<<  aDifF.mGradI  << aRes.mGradI << std::endl;
    StdOut() << " GRADJ "<<  aDifF.mGradJ  << aRes.mGradJ << std::endl;
    getchar();

    return aDifF;
    */
}

cPt3dr  cRPCSens::EpsDiffGround2Im(const cPt3dr & ) const {return mEpsCoord;}

cPt3dr cRPCSens::ImageZToGround(const cPt2dr& aPIm,const double aZ) const
{

    cPt2dr aPImN = NormIm(IO_PtIm(aPIm),true); // image normalised
    double aZN = NormZ(aZ,true); // norm Z
    cPt2dr aPGrN = mDirectRPC->Val(cPt3dr(aPImN.x(),aPImN.y(),aZN)); // ground normalised
    cPt3dr aRes = NormGround( cPt3dr(aPGrN.x(),aPGrN.y(),aZN) ,false); // ground unnormalised

    return IO_PtGr(aRes);
}

cPt3dr cRPCSens::ImageAndZ2Ground(const cPt3dr& aP) const
{
    return ImageZToGround(cPt2dr(aP.x(),aP.y()),aP.z());
}

tSeg3dr  cRPCSens::Image2Bundle(const cPt2dr & aPtIm) const 
{
     tREAL8  aZ0 = mGroundOffset.z() - mAmplZB;
     tREAL8  aZ1 = mGroundOffset.z() + mAmplZB;
      
     cPt3dr  aPGr0 = ImageZToGround(aPtIm,aZ0);
     cPt3dr  aPGr1 = ImageZToGround(aPtIm,aZ1);

     return tSeg3dr(aPGr0,aPGr1);
}

const cPt3dr * cRPCSens::CenterOfFootPrint() const { return & mCenterOfFootPrint; }

const cPixelDomain & cRPCSens::PixelDomain() const  {return mPixelDomain;}


double cRPCSens::DegreeVisibility(const cPt3dr & aP) const
{
     // To see, but there is probably a unity problem, maybe to it with normalized coordinat theb
     // multiply by "pseudo" focal to have convention similar to central perspective

     return mBoxGround.Insideness(aP);
}

     // ====================================================
     //     Not implemanted (not yet or never)
     // ====================================================
cPt3dr  cRPCSens::PseudoCenterOfProj() const
{
    MMVII_INTERNAL_ERROR("cRPCSens::PseudoCenterOfProj =>  2 Implement");

    return cPt3dr::Dummy();
}

cSensorImage *  AllocRPCDimap(const cAnalyseTSOF & anAnalyse,const std::string & aNameImage)
{
    cRPCSens* aRes = new cRPCSens(aNameImage);
    aRes->InitFromFile(anAnalyse);

    return aRes;
}

};
