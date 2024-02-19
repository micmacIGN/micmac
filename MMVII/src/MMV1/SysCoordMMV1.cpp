#include "V1VII.h"
#include "MMVII_util.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"


#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

// WGS84Degre
// WGS84

namespace MMVII
{

/*********************************************/
/*                                           */
/*             cSysCoUsingV1                 */
/*                                           */
/*********************************************/

class cSysCoUsingV1 : public cSysCoordV2
{
	public :
           //cSysCoUsingV1(cSysCoord *,eSysCoGeo,const std::vector<double>& aVAttrD,const std::vector<std::string>& aVAttrS);
	   cSysCoUsingV1(eSysCoGeo,const std::map<std::string,std::string> &);

           void AddData(const  cAuxAr2007 & anAuxInit) ;
           void ToFile(const std::string &) const override;
	   static cSysCoUsingV1 * FromFile(const std::string &);

           tPt ToGeoC(const tPt &) const override;
           tPt FromGeoC(const tPt &) const  override;

	   cSysCoUsingV1();
           ~cSysCoUsingV1();
           void  InterpretAttr();

	   static cSysCoUsingV1*   RTL(const cPt3dr & aPt,const std::string & aSysRef);


	   /// Conventional value for string that has been consumed
	   static const std::string SConsumed;

	   static const std::string KeyRTLSys;
	   static const std::string KeyRTL_x0;
	   static const std::string KeyRTL_y0;
	   static const std::string KeyRTL_z0;
	   static const std::string KeyNameLocal;

        private :
	   static std::string  GetAttr(std::map<std::string,std::string> & aMap,const std::string & aKey);
	   static void  SetAttr(std::map<std::string,std::string> & aMap,const std::string & aKey,const std::string & aVal);

	   void Init(eSysCoGeo,std::map<std::string,std::string> &);

	   cSysCoord *                        mSV1;
	   eSysCoGeo                          mType;
	   std::map<std::string,std::string>  mAttribs;
};

// Any value that will never be a valide value
const std::string cSysCoUsingV1::SConsumed = "@#%Zy_!";
const std::string cSysCoUsingV1::KeyRTLSys = "Sys_RTL";
const std::string cSysCoUsingV1::KeyRTL_x0 = "x0_RTL";
const std::string cSysCoUsingV1::KeyRTL_y0 = "y0_RTL";
const std::string cSysCoUsingV1::KeyRTL_z0 = "z0_RTL";
const std::string cSysCoUsingV1::KeyNameLocal = "NameLocalSys";

std::string  cSysCoUsingV1::GetAttr(std::map<std::string,std::string> & aMap,const std::string & aKey)
{
     auto anIter = aMap.find(aKey);
     if (anIter == aMap.end())
         MMVII_UnclasseUsEr("Key not found in cSysCoUsingV1::GetAttr : " + aKey);

     if (anIter->second == SConsumed)
         MMVII_UnclasseUsEr("Key multiple consumed in cSysCoUsingV1::GetAttr : " + aKey);

     std::string aRes = anIter->second;
     anIter->second = SConsumed;

     return aRes;
}

void  cSysCoUsingV1::SetAttr(std::map<std::string,std::string> & aMap,const std::string & aKey,const std::string & aVal)
{
     auto anIter = aMap.find(aKey);
     if (anIter != aMap.end())
         MMVII_UnclasseUsEr("Key multiple found in cSysCoUsingV1::GetAttr : " + aKey);

     aMap[aKey] = aVal;
}


cSysCoUsingV1::cSysCoUsingV1() :
  mSV1   (nullptr),
  mType  (eSysCoGeo::eNbVals)
{
}

cSysCoUsingV1::cSysCoUsingV1(eSysCoGeo aType,const std::map<std::string,std::string> & aAttribs) :
    mSV1      (nullptr),
    mType     (aType),
    mAttribs  (aAttribs)
{
    InterpretAttr();
}

void cSysCoUsingV1::AddData(const  cAuxAr2007 & anAuxInit) 
{
     cAuxAr2007 anAux("SysCoGeo",anAuxInit);

     MMVII::EnumAddData(anAux,mType,"Type");
     MMVII::AddData(cAuxAr2007("Attributes",anAux),mAttribs);
}

void AddData(const  cAuxAr2007 & anAux,cSysCoUsingV1 & aSysC)
{
    aSysC.AddData(anAux);
}

void cSysCoUsingV1::ToFile(const std::string & aNameFile) const 
{
   SaveInFile(*this,aNameFile);
}

void cSysCoUsingV1::Init(eSysCoGeo aType,std::map<std::string,std::string> & aMap)
{
     if  (aType == eSysCoGeo::eLambert93)
     {
         mPtEpsDeriv = cPt3dr(1.0,1.0,1.0);
         mSV1 = cProj4::Lambert93();
     }
     else if  (aType == eSysCoGeo::eGeoC)
     {
         mPtEpsDeriv = cPt3dr(1.0,1.0,1.0);
         mSV1 = cSysCoord::GeoC();
     }
     else if  (aType == eSysCoGeo::eWGS84Degrees)
     {
         // Rules 40000 Km for the earth perimeter
         tREAL8 aEpsXYZ =  4e7 / 360.0;
         mPtEpsDeriv = cPt3dr(aEpsXYZ,aEpsXYZ,1.0);
         mSV1 = cSysCoord::WGS84Degre();
     }
     else if  (aType == eSysCoGeo::eWGS84Rads)
     {
         tREAL8 aEpsXYZ =  4e7 / 6.28;
         mPtEpsDeriv = cPt3dr(aEpsXYZ,aEpsXYZ,1.0);
         mSV1 = cSysCoord::WGS84();
     }
     else if  (aType == eSysCoGeo::eLocalSys)
     {
     }
     else
     {
         MMVII_UnclasseUsEr("Unhandled sys-co in read");
     }
}


cSysCoUsingV1 * cSysCoUsingV1::FromFile(const std::string & aNameFile)
{
     cSysCoUsingV1 * aRes = new cSysCoUsingV1;
     ReadFromFile(*aRes,aNameFile);
     aRes->InterpretAttr();

     return aRes;
}

void  cSysCoUsingV1::InterpretAttr()
{
     std::map<std::string,std::string>  aCpAttr = mAttribs;

     if (mType  != eSysCoGeo::eRTL)
     {
         Init(mType,aCpAttr);
     }
     else
     {
          std::string aStrSysOri =  GetAttr(aCpAttr,KeyRTLSys);  // extract the string of system storing ori
	  eSysCoGeo  aTypeSysOri = Str2E<eSysCoGeo>(aStrSysOri); // convert it to enum
	  cSysCoUsingV1 aSysOri;
	  aSysOri.Init(aTypeSysOri,aCpAttr);  // initialize a system

	  tREAL8 aXOri = cStrIO<tREAL8>::FromStr(GetAttr(aCpAttr,KeyRTL_x0));
	  tREAL8 aYOri = cStrIO<tREAL8>::FromStr(GetAttr(aCpAttr,KeyRTL_y0));
	  tREAL8 aZOri = cStrIO<tREAL8>::FromStr(GetAttr(aCpAttr,KeyRTL_z0));

	  cPt3dr  aOriSys(aXOri,aYOri,aZOri);
	  aOriSys = aSysOri.ToGeoC(aOriSys);

	  mSV1 = cSysCoord::RTL(ToMMV1(aOriSys));
     }

}

cSysCoUsingV1*   cSysCoUsingV1::RTL(const cPt3dr & aPt,const std::string & aSysRef)
{
    cSysCoUsingV1 *  aSys = cSysCoUsingV1::FromFile(aSysRef);

    std::map<std::string,std::string> aAttr = aSys->mAttribs;

    SetAttr(aAttr,KeyRTLSys,E2Str(aSys->mType));
    SetAttr(aAttr,KeyRTL_x0,ToStr(aPt.x()));
    SetAttr(aAttr,KeyRTL_y0,ToStr(aPt.y()));
    SetAttr(aAttr,KeyRTL_z0,ToStr(aPt.z()));

    cSysCoUsingV1* aRes = new cSysCoUsingV1(eSysCoGeo::eRTL,aAttr);

    delete aSys;

    return aRes;
}

cPt3dr cSysCoUsingV1::ToGeoC(const cPt3dr & aP) const
{
    MMVII_INTERNAL_ASSERT_strong(mSV1!=nullptr,"No cSysCoUsingV1::ToGeoC");
    return ToMMVII(mSV1->ToGeoC(ToMMV1(aP)));
}
cPt3dr cSysCoUsingV1::FromGeoC(const cPt3dr & aP) const
{
    MMVII_INTERNAL_ASSERT_strong(mSV1!=nullptr,"No cSysCoUsingV1::FromGeoC");
    return ToMMVII(mSV1->FromGeoC(ToMMV1(aP)));
}

cSysCoUsingV1::~cSysCoUsingV1()
{
    // delete mSV1;
}

void GenSpec_SysCoordV1(const std::string & aDir)
{
    SpecificationSaveInFile<cSysCoUsingV1>(aDir+"SysCoordV1.xml");
}

/*********************************************/
/*                                           */
/*             cSysCoordLocal                */
/*                                           */
/*********************************************/

/*
class cSysCoordLocal : public cSysCoordV2
{
    public :
           //void ToFile(const std::string &) const override;
	   //static cSysCoUsingV1 * FromFile(const std::string &);

           tPt ToGeoC(const tPt &) const override;
           tPt FromGeoC(const tPt &) const  override;
    private :
};

tPt cSysCoordLocal::ToGeoC(const tPt &) const
{
    MMVII_INTERNAL_ERROR("No cSysCoordLocal::ToGeoC");
    return tPt::PCste(0.0);
}

tPt cSysCoordLocal::FromGeoC(const tPt &) const
{
    MMVII_INTERNAL_ERROR("No cSysCoordLocal::ToGeoC");
    return tPt::PCste(0.0);
}
*/


/*********************************************/
/*                                           */
/*             cSysCoordV2                   */
/*                                           */
/*********************************************/

cSysCoordV2::cSysCoordV2(tREAL8  aEpsDeriv) :
    cDataInvertibleMapping<tREAL8,3>(tPt::PCste(aEpsDeriv)),
    mPtEpsDeriv  (tPt::PCste(aEpsDeriv))
{
}

cSysCoordV2::~cSysCoordV2() {}

cPt3dr  cSysCoordV2::Value(const cPt3dr &aP) const
{
	return ToGeoC(aP);
}
cPt3dr  cSysCoordV2::Inverse(const cPt3dr &aP) const
{
	return FromGeoC(aP);
}

tPtrSysCo cSysCoordV2::Lambert93()
{
   return tPtrSysCo(new cSysCoUsingV1(eSysCoGeo::eLambert93,{}));
}

tPtrSysCo cSysCoordV2::GeoC()
{
   return tPtrSysCo(new cSysCoUsingV1(eSysCoGeo::eGeoC,{}));
}

tPtrSysCo cSysCoordV2::LocalSystem(const  std::string & aName)
{
   std::map<std::string,std::string> aMap;
   aMap[cSysCoUsingV1::KeyNameLocal] = aName;
   return tPtrSysCo(new cSysCoUsingV1(eSysCoGeo::eLocalSys,aMap));
}

tPtrSysCo cSysCoordV2::FromFile(const std::string & aName)
{
    return tPtrSysCo(cSysCoUsingV1::FromFile(aName));
}

tPtrSysCo cSysCoordV2::RTL(const cPt3dr & anOriInit,const std::string & aName)
{
     return tPtrSysCo(cSysCoUsingV1::RTL(anOriInit,aName));
}


/*********************************************/
/*                                           */
/*            cChangSysCoordV2               */
/*                                           */
/*********************************************/


cChangSysCoordV2::cChangSysCoordV2(tPtrSysCo aSysInit,tPtrSysCo aSysTarget,tREAL8  aEpsDeriv)  :
    cDataInvertibleMapping<tREAL8,3> (cPt3dr::PCste(aEpsDeriv)),
    mIdent        (false),
    mSysInit      (aSysInit),
    mSysTarget    (aSysTarget)
{
}

bool   cChangSysCoordV2::IsIdent() const {return mIdent;}


cChangSysCoordV2::cChangSysCoordV2 () :
     cDataInvertibleMapping<tREAL8,3> (cPt3dr::PCste(1.0)),
     mIdent     (true),
     mSysInit   (nullptr),
     mSysTarget (nullptr)
{
}

cChangSysCoordV2::cChangSysCoordV2 (tPtrSysCo aSysInOut) :
    cDataInvertibleMapping<tREAL8,3> (cPt3dr::PCste(1.0)),  // normally no influence
    mIdent        (true),
    mSysInit      (aSysInOut),
    mSysTarget    (aSysInOut)
{
}

tPtrSysCo cChangSysCoordV2::SysInit()   {return mSysInit;}
tPtrSysCo cChangSysCoordV2::SysTarget() {return mSysTarget;}


cChangSysCoordV2::~cChangSysCoordV2() {}

cPt3dr cChangSysCoordV2::Value(const cPt3dr & aPInit) const 
{
        if (mIdent)  return aPInit;
	return mSysTarget->FromGeoC(mSysInit->ToGeoC(aPInit));
}

cPt3dr cChangSysCoordV2::Inverse(const cPt3dr & aPInit) const 
{
        if (mIdent)  return aPInit;
	return mSysInit->FromGeoC(mSysTarget->ToGeoC(aPInit));
}

#if (0)

#endif


};
