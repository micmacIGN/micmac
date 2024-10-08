#ifndef  _MMVII_SYSCO_H_
#define  _MMVII_SYSCO_H_

#include "MMVII_DeclareCste.h"
#include "MMVII_Mappings.h"
#include "MMVII_AllClassDeclare.h"
#include "MMVII_Geom3D.h"
#include <proj.h>

//struct pj_ctx;
//typedef struct pj_ctx PJ_CONTEXT;
struct PJconsts;
typedef struct PJconsts PJ; //< libproj conversion between 2 CRS

namespace MMVII
{

/**
 * @brief The cSysCoData class is only used for cSysCo serialization
 */
class cSysCoData
{
public :
    std::string mDef; //< definition
    void AddData(const  cAuxAr2007 & anAuxInit);
};

void AddData(const cAuxAr2007 & anAux, cSysCoData & aSysCoData);

/**
 * @brief The cSysCo class records one coordinate system.
 * Derived classes are used for different types of coordinates systems.
 *
 * Value() and Inverse() to convert to/from geocentric
 *
 * SysCo definitions are like this:
 * "type*param1*param2*..."
 * They entirely define a SysCo and are interpreted by the factory MakeSysCo() to create an object of the corresponding concrete class.
 * The definitions are retrieved by command line argument or by deserialization of a cSysCoData.
 */
class cSysCo : public cDataInvertibleMapping<tREAL8,3>
{
public :
    typedef cIsometry3D<tREAL8> tPoseR;
    virtual ~cSysCo();

    // do not copy and move because of PJ* (not needed via tPtrSysCo)
    cSysCo(const cSysCo &other) = delete;
    cSysCo(cSysCo &&other) = delete;
    cSysCo& operator=(const cSysCo& other) = delete;
    cSysCo& operator=(cSysCo&& other) = delete;

    virtual tPt Value(const tPt &)   const override = 0; //< to GeoC
    virtual tPt Inverse(const tPt &) const override = 0; //< from GeoC


    tPt toGeoG(const tPt & aPtIn) const; //< uses mPJ_GeoC2Geog
    tPt fromGeoG(const tPt &aPtInGeoG) const; //< uses mPJ_GeoC2Geog

    virtual cRotation3D<tREAL8> getRot2Vertical(const tPt &)   const; //< get rotation from SysCo origin to vertical at point
    virtual tREAL8 getRadiusApprox(const tPt &in) const; //< approximate earth total curvature radius at a point
    virtual tREAL8 getDistHzApprox(const tPt & aPtA, const tPt & aPtB) const; //< approximate horizontal distance (along ellipsoid) from one point to an other
    virtual const tPoseR *getTranfo2GeoC() const;

    static tPtrSysCo MakeSysCo(const std::string &aDef, bool aDebug=false); //< factory from a SysCo definition
    static tPtrSysCo makeRTL(const cPt3dr & anOrigin, const std::string & aSysCoInDef);
    static tPtrSysCo FromFile(const std::string &aNameFile, bool aDebug=false);

    std::string Def() const { return mDef; }
    cSysCoData toSysCoData();

    eSysCo getType() const { return mType; }
    bool isEuclidian() const;

    tREAL8 getEllipsoid_a() const {return semi_axis;}
    tREAL8 getEllipsoid_e2() const {return e2;}
    tREAL8 getEllipsoid_b() const {return b;}
    bool isReady() const {return mIsReady; }
protected :
    cSysCo(bool aDebug);
    cSysCo(const std::string & def, bool aDebug);
    std::string mDef; //< definition
    eSysCo mType;
    PJ_CONTEXT* mPJContext;
    PJ* mPJ_GeoC2Geog; //< for generic use
    bool mDebug; //< show debug messages
    bool mIsReady = true; // to be able to compute transfo later

    //GRS80
    const tREAL8 semi_axis = 6378137;
    const tREAL8 e2        = 0.00669438;
    const tREAL8 b = semi_axis * sqrt(1.-e2);
};

inline bool operator==(const cSysCo& lhs, const cSysCo& rhs) { return lhs.Def() == rhs.Def(); }
inline bool operator!=(const cSysCo& lhs, const cSysCo& rhs) { return !(lhs == rhs); }

//------------------------------------------------------------

/**
 * @brief Coordinate Reference System transformation
 *
 * Value() goes from mSysCoInit to mSysCoTarget
 * Inverse() goes from mSysCoTarget to mSysCoInit
 *
 * It works with cSysCoLocal only if both have the same definition.
 * */
class cChangeSysCo : public cDataInvertibleMapping<tREAL8,3>
{
public:
    cChangeSysCo(); //< never do anything
    cChangeSysCo(tPtrSysCo aSysCoOrigin, tPtrSysCo aSysCoTarget, tREAL8  aEpsDeriv = 1.0);
    cChangeSysCo(const cChangeSysCo &other);

    tPt Value(const tPt &)   const override;
    tPt Inverse(const tPt &) const override;

    tPtrSysCo  SysOrigin() const { return mSysCoOrigin; };     ///< Accessor
    tPtrSysCo  SysTarget() const { return mSysCoTarget; };   ///< Accessor

    void setOriginSysCo(tPtrSysCo aSysCo);
    void setTargetsysCo(tPtrSysCo aSysCo);

private:
    tPtrSysCo mSysCoOrigin;
    tPtrSysCo mSysCoTarget;
    bool mIsNull; //< if nothing to do
};

//------------------------------------------------------------



}; // MMVII

#endif  //  _MMVII_SYSCO_H_
