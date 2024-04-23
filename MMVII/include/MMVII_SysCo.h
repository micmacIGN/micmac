#ifndef  _MMVII_SYSCO_H_
#define  _MMVII_SYSCO_H_

#include "MMVII_DeclareCste.h"
#include "MMVII_Mappings.h"
#include "MMVII_AllClassDeclare.h"


namespace MMVII
{

/**
 * @brief The cSysCoData class is only used for cSysCo serialization
 */
class cSysCoData
{
public :
    std::string mName; //< name / def
    void AddData(const  cAuxAr2007 & anAuxInit);
};

void AddData(const cAuxAr2007 & anAux, cSysCoData & aSysCoData);

/**
 * @brief The cSysCo class records one coordinate system.
 * Derived classes are used for different types of coordinates systems.
 */
class cSysCo : public cDataInvertibleMapping<tREAL8,3>
{
public :
    cSysCo(const cSysCo &other) = delete;
    virtual ~cSysCo();

    virtual tPt Value(const tPt &)   const override = 0; //< to GeoC
    virtual tPt Inverse(const tPt &) const override = 0; //< from GeoC

    virtual cRotation3D<tREAL8> getVertical(const tPt &)   const; //< get rotation from SysCo origin to vertical at this point

    static tPtrSysCo MakeSysCo(const std::string &aDef); //< factory
    static tPtrSysCo makeRTL(const cPt3dr & anOrigin, const std::string & aSysCoNameIn);
    static tPtrSysCo FromFile(const std::string &aNameFile);

    std::string Name() const { return mName; }
    cSysCoData toSysCoData();

    eSysCo getType() const { return mType; }
    bool isEuclidian() const;
protected :
    cSysCo();
    cSysCo(const std::string & def);
    std::string mName; //< name / def
    eSysCo mType;
};

//------------------------------------------------------------

/**
 * @brief Coordinate Reference System transformation
 *
 * Value() goes from mSysCoInit to mSysCoTarget
 * Inverse() goes from mSysCoTarget to mSysCoInit
 *
 * It works with local cSysCo only if both have the same name.
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
