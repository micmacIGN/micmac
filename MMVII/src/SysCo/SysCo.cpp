#include "MMVII_SysCo.h"

#include <proj.h>
#include "MMVII_Geom3D.h"


namespace MMVII
{

PJ* cSysCo::PJ_GeoC2Geog = nullptr;

const std::string SysCoDefSep = "*";

PJ* createCRS2CRS(const std::string &def_from, const std::string &def_to); //< returns nullptr if error

PJ_COORD toPjCoord(const tPt3dr &aPt)
{
    PJ_COORD aPtPJ;
    aPtPJ.xyzt.x = aPt.x();
    aPtPJ.xyzt.y = aPt.y();
    aPtPJ.xyzt.z = aPt.z();
    aPtPJ.xyzt.t = 0.;
    return aPtPJ;
}

tPt3dr fromPjCoord(const PJ_COORD &aPtPJ)
{
    return tPt3dr(aPtPJ.xyz.x, aPtPJ.xyz.y, aPtPJ.xyz.z);
}


//---------------------------------------------------

void cSysCoData::AddData(const  cAuxAr2007 & anAuxInit)
{
    cAuxAr2007 anAux("SysCoData",anAuxInit);
    MMVII::AddData(cAuxAr2007("Def",anAux),mDef);
}

void AddData(const cAuxAr2007 & anAux, cSysCoData & aSysCoData)
{
    aSysCoData.AddData(anAux);
}



//---------------------------------------------------

/**
 * @brief cSysCoLocal: for for local coordinates.
 *
 * Value() and Inverse() make an error if those fonctions are used
 * because there is no info to convert local into georeferenced coordinates.
 */
class cSysCoLocal : public cSysCo
{
    friend class cSysCo;
public :
    tPt Value(const tPt & in)   const override; //< to GeoC: error
    tPt Inverse(const tPt & in) const override; //< to GeoC: error
protected:
    cSysCoLocal(const std::string & def);
};


/**
 * @brief cSysCoGeoC: for geocentric coordinates
 *
 * Value() and Inverse() are identity.
 */
class cSysCoGeoC : public cSysCo
{
    friend class cSysCo;
public :
    tPt Value(const tPt & in)   const override { return in; } //< to GeoC
    tPt Inverse(const tPt & in) const override { return in; } //< from GeoC
protected:
    cSysCoGeoC(const std::string & def);
};

/**
 * @brief cSysCoProj: for any libproj CRS (including geographic)
 */
class cSysCoProj : public cSysCo
{
    friend class cSysCo;
public :
    virtual ~cSysCoProj();

    // do not copy and move because of PJ* (not needed via tPtrSysCo)
    cSysCoProj(const cSysCoProj &other) = delete;
    cSysCoProj(cSysCoProj &&other) = delete;
    cSysCoProj& operator=(const cSysCoProj& other) = delete;
    cSysCoProj& operator=(cSysCoProj&& other) = delete;

    tPt Value(const tPt &)   const override; //< to GeoC
    tPt Inverse(const tPt &) const override; //< from GeoC

protected:
    cSysCoProj(const std::string & def);
    PJ* mPJ_Proj2GeoC;
};

/**
 * @brief cSysCoLGeo: for any local euclidian frame that
 * have a matrix transfrom into geocentric
 */
class cSysCoLGeo : public cSysCo
{
    friend class cSysCo;
public :
    typedef cIsometry3D<tREAL8> tPoseR;
    virtual ~cSysCoLGeo();

    // do not copy and move because of PJ* (not needed via tPtrSysCo)
    cSysCoLGeo(const cSysCoLGeo &other) = delete;
    cSysCoLGeo(cSysCoLGeo &&other) = delete;
    cSysCoLGeo& operator=(const cSysCoLGeo& other) = delete;
    cSysCoLGeo& operator=(cSysCoLGeo&& other) = delete;

    tPt Value(const tPt &)   const override; //< to GeoC
    tPt Inverse(const tPt &) const override; //< from GeoC

    const tPoseR& getTranfo2GeoC() const { return mTranfo2GeoC; }
protected:
    cSysCoLGeo(const std::string & def); //< construct from a definition, starting with LGeo
    cSysCoLGeo(); //< default constructor for derived classes
    tPoseR mTranfo2GeoC; //< the transfo to geocentric
    tREAL8 mCenterLatRad, mCenterLongRad;
};

/**
 * @brief cSysCoRTL: A special case of cSysCoLGeo, where
 * X = easting, Y = northing, Z = up at origin point
 * RTL = local tangent frame
 */
class cSysCoRTL : public cSysCoLGeo
{
    friend class cSysCo;
public :
    typedef cIsometry3D<tREAL8> tPoseR;
    virtual ~cSysCoRTL();

    // do not copy and move because of PJ* (not needed via tPtrSysCo)
    cSysCoRTL(const cSysCoRTL &other) = delete;
    cSysCoRTL(cSysCoRTL &&other) = delete;
    cSysCoRTL& operator=(const cSysCoRTL& other) = delete;
    cSysCoRTL& operator=(cSysCoRTL&& other) = delete;

    virtual cRotation3D<tREAL8> getVertical(const tPt &aPtIn) const override; //< get rotation from SysCo origin to vertical at this point

protected:
    cSysCoRTL(const std::string & def); //< construct from a definition, starting with RTL
    cSysCoRTL(tPt anOrigin, std::string aInDef); //< construct for RTL
    bool computeRTL(tPt anOrigin, std::string aInDef); //< init mTranfo2GeoC for RTL case
};
//------------------------------------------------------------

cSysCo::cSysCo() :
    mDef(), mType(eSysCo::eLocalSys)
{
    if (!PJ_GeoC2Geog)
        PJ_GeoC2Geog = createCRS2CRS(MMVII_SysCoDefGeoC, MMVII_SysCoDefLatLong);
}

cSysCo::cSysCo(const std::string &aDef) :
    mDef(aDef), mType(eSysCo::eLocalSys)
{
}

cSysCo::~cSysCo()
{
}

bool cSysCo::isEuclidian() const
{
    switch (mType) {
    case eSysCo::eGeoC:
    case eSysCo::eLocalSys:
    case eSysCo::eLGeo:
    case eSysCo::eRTL:
        return true;
    case eSysCo::eProj:
    case eSysCo::eNbVals:
        return false;
    }
    return false;
}

cSysCoData cSysCo::toSysCoData()
{
    return {mDef};
}

cRotation3D<tREAL8> cSysCo::getVertical(const tPt &)   const
{
    MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eBadSysCo,
                               std::string("Error: getVertical() not defined for SysCo type ") + E2Str(mType));
    return cRotation3D<tREAL8>::Identity();
}

tREAL8 cSysCo::getRadiusApprox(const tPt &in) const
{
    auto inGeoc = Value(in);
    PJ_COORD pj_geoc = toPjCoord(inGeoc);
    PJ_COORD pj_geog = proj_trans(PJ_GeoC2Geog, PJ_FWD, pj_geoc);
    tREAL8 lat = pj_geog.lp.phi/AngleInRad(eTyUnitAngle::eUA_degree);

    //GRS80
    const tREAL8 semi_axis = 6378137;
    const tREAL8 e2        = 0.00669438;
    return semi_axis*sqrt(1-e2)/(1-e2*Square(sin(lat)));//total curvature sphere
}

tREAL8 cSysCo::getDistHzApprox(const tPt & aPtA, const tPt & aPtB) const
{
    auto aPtAGeoc = Value(aPtA);
    auto aPtAgeog = fromPjCoord(proj_trans(PJ_GeoC2Geog, PJ_FWD, toPjCoord(aPtAGeoc)));
    auto aPtBGeoc = Value(aPtB);

    tREAL8 cosAlpha = Scal(aPtAGeoc,aPtBGeoc)/(Norm2(aPtAGeoc)*Norm2(aPtBGeoc));
    tREAL8 alpha = acos(cosAlpha);

    tREAL8 radius = getRadiusApprox(aPtA);

    return alpha*(radius + aPtAgeog.z());
}

tPtrSysCo cSysCo::MakeSysCo(const std::string &aDef)
{
    if (starts_with(aDef,MMVII_SysCoLocal))
    {
        return tPtrSysCo(new cSysCoLocal(aDef));
    }
    else if (starts_with(aDef,MMVII_SysCoGeoC))
    {
        return tPtrSysCo(new cSysCoGeoC(aDef));
    }
    else if (starts_with(aDef,MMVII_SysCoLGeo))
    {
        return tPtrSysCo(new cSysCoLGeo(aDef));
    }
    else if (starts_with(aDef,MMVII_SysCoRTL))
    {
        return tPtrSysCo(new cSysCoRTL(aDef));
    }
    else // def is supposed to be a libproj definition
    {
        return tPtrSysCo(new cSysCoProj(aDef));
    }
}

tPtrSysCo cSysCo::makeRTL(const cPt3dr & anOrigin, const std::string & aSysCoInDef)
{
    tPtrSysCo aSysCoFrom;
    if (ExistFile(aSysCoInDef))
        aSysCoFrom = cSysCo::FromFile(aSysCoInDef);
    else
        aSysCoFrom = cSysCo::MakeSysCo(aSysCoInDef);

    std::ostringstream oss;
    oss.precision(8);
    oss<<std::fixed;
    oss<<MMVII_SysCoRTL<<SysCoDefSep<<anOrigin.x()<<SysCoDefSep<<anOrigin.y()
       <<SysCoDefSep<<anOrigin.z()<<SysCoDefSep<<aSysCoFrom->Def();

    return MakeSysCo(oss.str());
}

tPtrSysCo cSysCo::FromFile(const std::string &aNameFile)
{
    cSysCoData aSysCoDataTmp;
    ReadFromFile(aSysCoDataTmp,aNameFile);
    return MakeSysCo(aSysCoDataTmp.mDef);
}

//------------------------------------------------------------

cSysCoLocal::cSysCoLocal(const std::string &aDef) :
    cSysCo(aDef)
{
    mType = eSysCo::eLocalSys;
}

tPt3dr cSysCoLocal::Value(const tPt & in)   const  //< to GeoC
{
    MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eBadSysCo,
                               "Can not convert SysCoLocal to Geocentric")
    return {};
}

tPt3dr cSysCoLocal::Inverse(const tPt & in) const //< from GeoC
{
    MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eBadSysCo,
                               "Can not convert SysCoLocal from Geocentric")
    return {};
}

//------------------------------------------------------------


cSysCoGeoC::cSysCoGeoC(const std::string &aDef) :
    cSysCo(aDef)
{
    mType = eSysCo::eGeoC;
}

//------------------------------------------------------------

cSysCoLGeo::cSysCoLGeo(const std::string &aDef) :
    cSysCo(aDef), mTranfo2GeoC({}, cRotation3D<tREAL8>::Identity()),
    mCenterLatRad(NAN), mCenterLongRad(NAN)
{
    mType = eSysCo::eLGeo;

    auto tokens = SplitString(mDef, SysCoDefSep);
    MMVII_INTERNAL_ASSERT_User(tokens.size()>0, eTyUEr::eInsufNbParam,
                               "Error in LGeo definition format: \""+mDef+"\"")
    if (tokens[0]==MMVII_SysCoLGeo)
    {
        MMVII_INTERNAL_ASSERT_User(tokens.size()==7, eTyUEr::eInsufNbParam,
                                   "Error in LGeo definition format: \""+mDef+"\"")
        mTranfo2GeoC.Tr() = {std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3])};
        tPt aOmegaPhiKappa(std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]));
        mTranfo2GeoC.SetRotation(cRotation3D<tREAL8>::RotFromWPK(aOmegaPhiKappa));

        PJ_COORD to = proj_trans(PJ_GeoC2Geog, PJ_FWD, toPjCoord(mTranfo2GeoC.Tr()));
        if (proj_errno(PJ_GeoC2Geog))
        {
            StdOut()<<"Error with PJ_GeoC2Geog: "<<
                      proj_errno_string(proj_errno(PJ_GeoC2Geog))<<"\n";
            MMVII_INTERNAL_ASSERT_medium(false, "SysCo proj error")
        }
        mCenterLatRad = to.lp.phi/AngleInRad(eTyUnitAngle::eUA_degree);
        mCenterLongRad = to.lp.lam/AngleInRad(eTyUnitAngle::eUA_degree);
    }
    else
    {
        MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError,
                                   "Error in LGeo definition format: \""+mDef+"\"")
    }
}

cSysCoLGeo::cSysCoLGeo() :
    cSysCo(), mTranfo2GeoC({}, cRotation3D<tREAL8>::Identity()),
    mCenterLatRad(NAN), mCenterLongRad(NAN)
{
    mType = eSysCo::eLGeo;
}

cSysCoLGeo::~cSysCoLGeo()
{
}


tPt3dr cSysCoLGeo::Value(const tPt & in)   const  //< to GeoC
{
    return getTranfo2GeoC().Rot().Mat() * in + getTranfo2GeoC().Tr();
}

tPt3dr cSysCoLGeo::Inverse(const tPt & in) const //< from GeoC
{
    return getTranfo2GeoC().Rot().Mat().Transpose() * (in - getTranfo2GeoC().Tr());
}


//------------------------------------------------------------

cSysCoRTL::cSysCoRTL(const std::string &aDef) :
    cSysCoLGeo()
{
    mType = eSysCo::eRTL;
    mDef = aDef;

    auto tokens = SplitString(mDef, SysCoDefSep);
    MMVII_INTERNAL_ASSERT_User(tokens.size()>0, eTyUEr::eInsufNbParam,
                               "Error in RTL definition format: \""+mDef+"\"")
    if (tokens[0]==MMVII_SysCoRTL)
    {
        MMVII_INTERNAL_ASSERT_User(tokens.size()==5, eTyUEr::eInsufNbParam,
                                   "Error in RTL definition format: \""+mDef+"\"")
        tPt anOrigin(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
        std::string aInDef = tokens[4];
        computeRTL(anOrigin, aInDef);
    }
    else
    {
        MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError,
                                   "Error in RTL definition format: \""+mDef+"\"")
    }
}

cSysCoRTL::cSysCoRTL(tPt anOrigin, std::string aInDef) :
    cSysCoLGeo()
{
    mType = eSysCo::eRTL;
    std::ostringstream oss;
    oss.precision(8);
    oss<<std::fixed;
    oss<<MMVII_SysCoRTL<<SysCoDefSep<<anOrigin.x()<<SysCoDefSep<<anOrigin.y()
       <<SysCoDefSep<<anOrigin.z()<<SysCoDefSep<<aInDef;
    mDef = oss.str();
    computeRTL(anOrigin,aInDef);
}

cSysCoRTL::~cSysCoRTL()
{
}

bool cSysCoRTL::computeRTL(tPt anOrigin, std::string aInDef)
{
    PJ_COORD from, to;
    from = toPjCoord(anOrigin);

    PJ* pj_in2latlong = createCRS2CRS(aInDef, MMVII_SysCoDefLatLong);
    PJ* pj_in2geocent = createCRS2CRS(aInDef, MMVII_SysCoDefGeoC);
    to = proj_trans(pj_in2geocent, PJ_FWD, from);
    if (proj_errno(pj_in2geocent))
    {
        StdOut()<<"Error with proj "<<aInDef<<" "<<MMVII_SysCoDefGeoC<<": "<<
                                  proj_errno_string(proj_errno(pj_in2geocent))<<"\n";
        MMVII_INTERNAL_ASSERT_medium(false, "SysCo proj error")
    }
    mTranfo2GeoC.Tr() = fromPjCoord(to);
    to = proj_trans(pj_in2latlong, PJ_FWD, from);
    if (proj_errno(pj_in2latlong))
    {
        StdOut()<<"Error with proj "<<aInDef<<" "<<MMVII_SysCoDefLatLong<<": "<<
                                  proj_errno_string(proj_errno(pj_in2latlong))<<"\n";
        MMVII_INTERNAL_ASSERT_medium(false, "SysCo proj error")
    }

    mCenterLatRad = to.lp.phi/AngleInRad(eTyUnitAngle::eUA_degree);
    mCenterLongRad = to.lp.lam/AngleInRad(eTyUnitAngle::eUA_degree);
    auto Rz = cRotation3D<tREAL8>::RotKappa(-mCenterLongRad);
    auto Ry = cRotation3D<tREAL8>::RotPhi( mCenterLatRad - M_PI/2.);
    auto Rz2 = cRotation3D<tREAL8>::RotFromCanonicalAxes("-jik").Mat();
            //            cRotation3D<tREAL8>::RotKappa(-M_PI/2);
    mTranfo2GeoC.Rot() = cRotation3D<tREAL8>((Rz2*Ry*Rz).Transpose(),false);

    proj_destroy(pj_in2latlong);
    proj_destroy(pj_in2geocent);

    if (0) // just to test, save RTL as generic LGeo
    {
        std::ostringstream oss;
        oss.precision(8);
        oss<<std::fixed;
        tPt aOmegaPhiKappa = mTranfo2GeoC.Rot().ToWPK();
        oss<<MMVII_SysCoLGeo<<SysCoDefSep
          <<mTranfo2GeoC.Tr().x()<<SysCoDefSep<<mTranfo2GeoC.Tr().y()<<SysCoDefSep<<mTranfo2GeoC.Tr().z()<<SysCoDefSep
          <<aOmegaPhiKappa.x()<<SysCoDefSep<<aOmegaPhiKappa.y()<<SysCoDefSep<<aOmegaPhiKappa.z();
        mDef = oss.str();
    }

    return true;
}


cRotation3D<tREAL8> cSysCoRTL::getVertical(const tPt & aPtIn)  const
{
    tPt ptGeoC = Value(aPtIn);
    auto anOtherRTL = cSysCoLGeo::makeRTL(ptGeoC, MMVII_SysCoDefGeoC);
    auto anOtherRTL_asRTL = static_cast<cSysCoRTL*>(anOtherRTL.get());
    // TODO: add vertical deflection
    return cRotation3D(anOtherRTL_asRTL->getTranfo2GeoC().Rot().Mat().Transpose(),false) * getTranfo2GeoC().Rot();
}

//------------------------------------------------------------

cSysCoProj::cSysCoProj(const std::string &aDef) :
    cSysCo(aDef), mPJ_Proj2GeoC(nullptr)
{
    mType = eSysCo::eProj;
    mPJ_Proj2GeoC = createCRS2CRS(mDef, MMVII_SysCoDefGeoC);
}

cSysCoProj::~cSysCoProj()
{
    proj_destroy(mPJ_Proj2GeoC);
}

tPt3dr cSysCoProj::Value(const tPt & in)   const  //< to GeoC
{
    PJ_COORD pj_in, pj_out;
    pj_in = proj_coord(in.x(), in.y(), in.z(), 0.);
    pj_out = proj_trans(mPJ_Proj2GeoC, PJ_FWD, pj_in);
    if (proj_errno(mPJ_Proj2GeoC))
    {
        StdOut()<<"Error with proj "<<mDef<<" "<<MMVII_SysCoDefGeoC<<": "<<
                                  proj_errno_string(proj_errno(mPJ_Proj2GeoC))<<"\n";
        MMVII_INTERNAL_ASSERT_medium(false, "SysCo proj error")
    }
    return {pj_out.xyz.x, pj_out.xyz.y, pj_out.xyz.z};
}

tPt3dr cSysCoProj::Inverse(const tPt & in) const //< from GeoC
{
    PJ_COORD pj_in, pj_out;
    pj_in = proj_coord(in.x(), in.y(), in.z(), 0.);
    pj_out = proj_trans(mPJ_Proj2GeoC, PJ_INV, pj_in);
    if (proj_errno(mPJ_Proj2GeoC))
    {
        StdOut()<<"Error with proj "<<mDef<<" "<<MMVII_SysCoDefGeoC<<": "<<
                                  proj_errno_string(proj_errno(mPJ_Proj2GeoC))<<"\n";
        MMVII_INTERNAL_ASSERT_medium(false, "SysCo proj error")
    }
    return {pj_out.xyz.x, pj_out.xyz.y, pj_out.xyz.z};
}

//------------------------------------------------------------

PJ* createCRS2CRS(const std::string &def_from, const std::string &def_to)
{
    PJ* aPJ = proj_create_crs_to_crs(nullptr, def_from.c_str(), def_to.c_str(), nullptr);
    if (!aPJ)
    {
        StdOut() << "Error: init proj \""<<def_from<<"\" to \""<<def_to
                 <<"\":\n"<<proj_errno_string(proj_errno(aPJ))<<"\n";
        MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eBadSysCo,
                                   std::string("Error in SysCo definition for \"")
                                   +def_from+"\" to \""+def_to+"\": "+proj_errno_string(proj_errno(aPJ)))
    }
    return aPJ;
}

//------------------------------------------------------------

cChangeSysCo::cChangeSysCo():
    cDataInvertibleMapping<tREAL8,3> (cPt3dr::PCste(1.0)),
    mSysCoOrigin(nullptr),mSysCoTarget(nullptr),mIsNull(true)
{
}


cChangeSysCo::cChangeSysCo(tPtrSysCo aSysCoOrigin, tPtrSysCo aSysCoTarget, tREAL8  aEpsDeriv):
    cDataInvertibleMapping<tREAL8,3> (cPt3dr::PCste(aEpsDeriv)),
    mSysCoOrigin(aSysCoOrigin),mSysCoTarget(aSysCoTarget),mIsNull(mSysCoOrigin->Def()==mSysCoTarget->Def())
{
}

cChangeSysCo::cChangeSysCo(const cChangeSysCo &other):
    cDataInvertibleMapping<tREAL8,3> (other.EpsJac()),
    mSysCoOrigin(other.mSysCoOrigin),mSysCoTarget(other.mSysCoTarget),mIsNull(mSysCoOrigin->Def()==mSysCoTarget->Def())
{
}


tPt3dr cChangeSysCo::Value(const tPt &in)   const
{
    if (mIsNull)
        return in;
    else
    {
        tPt aPtGeoC = mSysCoOrigin->Value(in);
        return mSysCoTarget->Inverse(aPtGeoC);
    }
}

tPt3dr cChangeSysCo::Inverse(const tPt &in)   const
{
    if (mIsNull)
        return in;
    else
    {
        tPt aPtGeoC = mSysCoTarget->Value(in);
        return mSysCoOrigin->Inverse(aPtGeoC);
    }
}

//------------------------------------------------------------

void GenSpec_SysCo(const std::string & aDir)
{
    SpecificationSaveInFile<cSysCoData>(aDir+"SysCo.xml");
}

//------------------------------------------------------------

void BenchSysCo(cParamExeBench & aParam)
{
    if (! aParam.NewBench("SysCo")) return;

    // basic libProj conversion
    std::string L93Def ="IGNF:LAMB93";
    std::string latlongDef ="+proj=latlong";
    PJ* pj_L932latlong = createCRS2CRS(L93Def, latlongDef);

    MMVII_INTERNAL_ASSERT_bench(pj_L932latlong,"SysCo create crs to crs");

    PJ_COORD a, b;
    a = proj_coord(657730, 6860675, 50, 0);
    b = proj_trans(pj_L932latlong, PJ_FWD, a);

    MMVII_INTERNAL_ASSERT_bench(fabs(b.lpz.lam - 2.42403442)<0.00001,"SysCo coords");
    MMVII_INTERNAL_ASSERT_bench(fabs(b.lpz.phi - 48.84473383)<0.00001,"SysCo coords");
    MMVII_INTERNAL_ASSERT_bench(fabs(b.lpz.z -50.)<0.00001,"SysCo coords");

    proj_destroy(pj_L932latlong);

    // RTL creation
    tPtrSysCo aSysCoRTL = cSysCoLGeo::makeRTL({657723.000000,6860710.000000,0.}, "IGNF:LAMB93");
    auto aSysCoRTL_asRTL = static_cast<cSysCoRTL *>(aSysCoRTL.get());
    //             -0.042293037933441094  -0.7522588760680056    0.6575088458106564              4201661.926785135
    // Geocentr =   0.9991052491816668    -0.031843805452299215  0.027832950112332798  * RTL +    177860.1878016033
    //              0                      0.6580976792477065    0.7529325630949846              4779236.016271434

    auto aPose = aSysCoRTL_asRTL->getTranfo2GeoC();
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Tr()-tPt3dr(4201661.926785135,177860.1878016033,4779236.016271434))<0.00001,"SysCo RTL");
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Rot().AxeI()-tPt3dr(-0.042293037933441094,0.9991052491816668,0.))<0.00001,"SysCo RTL");
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Rot().AxeJ()-tPt3dr(-0.7522588760680056,-0.031843805452299215,0.6580976792477065))<0.00001,"SysCo RTL");
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Rot().AxeK()-tPt3dr(0.6575088458106564,0.027832950112332798,0.7529325630949846))<0.00001,"SysCo RTL");


    // RTL to GeoC
    tPtrSysCo aSysCoGeoC = cSysCo::MakeSysCo("GeoC");
    cChangeSysCo aRTL2GeoC(aSysCoRTL, aSysCoGeoC);
    tPt3dr aPtGeoC = aSysCoRTL_asRTL->getTranfo2GeoC().Tr();
    MMVII_INTERNAL_ASSERT_bench(Norm2(aRTL2GeoC.Inverse(aPtGeoC)-tPt3dr(0.,0.,0.))<0.00001,"SysCo RTL2GeoC");

    tPt3dr aPtRTL = {100.,10,1.};
    aPtGeoC = aRTL2GeoC.Value(aPtRTL);
    tPt3dr aPtRTL2 = aRTL2GeoC.Inverse(aPtGeoC);
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPtRTL-aPtRTL2)<0.00001,"SysCo RTL2GeoC");

    // RTL to Proj
    tPtrSysCo aSysCoL93 = cSysCo::MakeSysCo("IGNF:LAMB93");
    cChangeSysCo aRTL2L93(aSysCoRTL, aSysCoL93);
    aPtRTL = {100.,10,1.};
    tPt3dr aPtL93 = aRTL2L93.Value(aPtRTL);
    aPtRTL2 = aRTL2L93.Inverse(aPtL93);
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPtRTL-aPtRTL2)<0.00001,"SysCo RTL2Proj");

    // Proj to RTL
    tPtrSysCo aSysCoRTL_bis = cSysCo::MakeSysCo("RTL*0.67451979*45.18899334*0.00000000*EPSG:4326");
    cChangeSysCo aL932RTL(aSysCoL93, aSysCoRTL_bis);
    aPtL93 = {521565.580,6459960.990,252.830};//{657723.000000+100,6860710.000000+10,1.};
    aPtRTL = aL932RTL.Value(aPtL93);
    tPt3dr aPtL93_bis = aL932RTL.Inverse(aPtRTL);
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPtL93-aPtL93_bis)<0.00001,"SysCo Proj2RTL");

    // RTL to RTL
    cChangeSysCo aRTL2RTL(aSysCoRTL, aSysCoRTL_bis);
    aPtRTL = {100.,10,1.};
    tPt3dr aPtRTL_bis = aRTL2RTL.Value(aPtRTL);
    aPtRTL2 = aRTL2RTL.Inverse(aPtRTL_bis);
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPtRTL-aPtRTL2)<0.00001,"SysCo RTL2RTL");

    // Ellipsoid normal
    tPtrSysCo aSysCoGeog = cSysCo::MakeSysCo(MMVII_SysCoDefLatLong);
    cChangeSysCo aGeoC2Geog(aSysCoGeoC, aSysCoGeog);
    tPt3dr aPtAGeog = {2.4240, 48.8447, 100.};
    tPt3dr aPtAGeoC = aGeoC2Geog.Inverse(aPtAGeog);
    tREAL8 lambda = aPtAGeog.x()/AngleInRad(eTyUnitAngle::eUA_degree);
    tREAL8 phi = aPtAGeog.y()/AngleInRad(eTyUnitAngle::eUA_degree);

    tREAL8 aDistN = 1000.;
    tPt3dr aVectN = { cos(lambda)*cos(phi), sin(lambda)*cos(phi), sin(phi) };
    tPt3dr aPtBGeoC = aPtAGeoC + aVectN * aDistN;
    tPt3dr aPtBGeog = aGeoC2Geog.Value(aPtBGeoC) - tPt3dr(0.,0.,aDistN) ;
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPtAGeog-aPtBGeog)<0.001,"SysCo Ellipsoid normal");



    // Ellipsoid normals
    const tREAL8 testDist = 1000.;
    std::vector<std::string> allOrigins = {"0*0", "55*0", "0*55", "55*55", "-55*0",  "0*-55", "-55*-55"};
    std::vector<tPt3dr> allPtRTL = { {RandInInterval(-100.,100.),RandInInterval(-100.,100.),RandInInterval(-100.,100.)},
                                     {testDist,RandInInterval(-100.,100.),RandInInterval(-100.,100.)},
                                     {RandInInterval(-100.,100.),testDist,RandInInterval(-100.,100.)},
                                     {testDist,testDist,RandInInterval(-100.,100.)} };
    for (auto & aOrigin: allOrigins)
    {
        for (auto & aPtRTL: allPtRTL)
        {
            tPtrSysCo aSysCoRTL_ter = cSysCo::MakeSysCo(std::string("RTL*")+aOrigin+"*0*"+MMVII_SysCoDefLatLong);
            cSysCoLGeo * aSysCoRTLasRTL = static_cast<cSysCoRTL *>(aSysCoRTL_ter.get());
            MMVII_INTERNAL_ASSERT_bench(aSysCoRTLasRTL,"SysCo RTL as RTL");

            // check if possible to find the point above aPtRTL in RTL
            tPtrSysCo aSysCoGeog = cSysCo::MakeSysCo(MMVII_SysCoDefLatLong);
            cChangeSysCo aRTL2Geog(aSysCoRTL, aSysCoGeog);
            auto aMatVertical = aSysCoRTLasRTL->getVertical( aPtRTL );
            tPt3dr aPtGeog = aRTL2Geog.Value(aPtRTL);
            tPt3dr aVectUp = {0.,0.,testDist};
            tPt3dr aPtGeogUp = aPtGeog + aVectUp;
            tPt3dr aPtRTLUp = aPtRTL + aMatVertical.Inverse(aVectUp);
            tPt3dr aPtRTLUp_check = aRTL2Geog.Inverse(aPtGeogUp);
            //std::cout<<std::setprecision(10);
            //std::cout<<"at "<<aOrigin<<", aPtRTL: "<<aPtRTL<<"\n";
            //std::cout<<"aPtRTLUp-aPtRTLUp_check: "<<aPtRTLUp-aPtRTLUp_check<<"\n";
            MMVII_INTERNAL_ASSERT_bench(Norm2(aPtRTLUp-aPtRTLUp_check)<0.001,"SysCo RTL vert");
        }
    }


    aParam.EndBench();
    return;
}


}; // MMVII

