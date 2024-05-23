#ifndef  _MMVII_SYSCO_H_
#define  _MMVII_SYSCO_H_

#include "MMVII_DeclareCste.h"
#include "MMVII_Mappings.h"
#include "MMVII_AllClassDeclare.h"

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
    cSysCo(const cSysCo &other) = delete;
    virtual ~cSysCo();

    virtual tPt Value(const tPt &)   const override = 0; //< to GeoC
    virtual tPt Inverse(const tPt &) const override = 0; //< from GeoC

    virtual cRotation3D<tREAL8> getVertical(const tPt &)   const; //< get rotation from SysCo origin to vertical at point
    virtual tREAL8 getRadiusApprox(const tPt &in) const; //< approximate earth total curvature radius at a point
    virtual tREAL8 getDistHzApprox(const tPt & aPtA, const tPt & aPtB) const; //< approximate horizontal distance (along ellipsoid) from one point to an other

    static tPtrSysCo MakeSysCo(const std::string &aDef); //< factory from a SysCo definition
    static tPtrSysCo makeRTL(const cPt3dr & anOrigin, const std::string & aSysCoInDef);
    static tPtrSysCo FromFile(const std::string &aNameFile);

    std::string Def() const { return mDef; }
    cSysCoData toSysCoData();

    eSysCo getType() const { return mType; }
    bool isEuclidian() const;
protected :
    cSysCo();
    cSysCo(const std::string & def);
    std::string mDef; //< definition
    eSysCo mType;
    static PJ* PJ_GeoC2Geog; //< for generic use
};

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

private:
    tPtrSysCo mSysCoOrigin;
    tPtrSysCo mSysCoTarget;
    bool mIsNull; //< if nothing to do
};

//------------------------------------------------------------



}; // MMVII

#endif  //  _MMVII_SYSCO_H_
