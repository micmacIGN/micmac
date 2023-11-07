#include "V1VII.h"
#include "MMVII_util.h"

namespace MMVII
{


/*********************************************/
/*                                           */
/*             cSysCoordV1                   */
/*                                           */
/*********************************************/

class cSysCoordV1 : public cSysCoordV2
{
	public :
           cSysCoordV1(cSysCoord *);
	private :

           tPt ToGeoC(const tPt &) const override;
           tPt FromGeoC(const tPt &) const  override;

	   cSysCoord *  mSV1;
};


cSysCoordV1::cSysCoordV1 (cSysCoord * aSV1) :
   cSysCoordV2  (0.1),
   mSV1  (aSV1)
{
}

cPt3dr cSysCoordV1::ToGeoC(const cPt3dr & aP) const
{
    return ToMMVII(mSV1->ToGeoC(ToMMV1(aP)));
}
cPt3dr cSysCoordV1::FromGeoC(const cPt3dr & aP) const
{
    return ToMMVII(mSV1->FromGeoC(ToMMV1(aP)));
}

/*********************************************/
/*                                           */
/*             cSysCoordV2                   */
/*                                           */
/*********************************************/

cSysCoordV2::cSysCoordV2(tREAL8  aEpsDeriv) :
    cDataInvertibleMapping<tREAL8,3>(tPt::PCste(aEpsDeriv))
{
}

cPt3dr  cSysCoordV2::Value(const cPt3dr &aP) const
{
	return ToGeoC(aP);
}
cPt3dr  cSysCoordV2::Inverse(const cPt3dr &aP) const
{
	return FromGeoC(aP);
}

cSysCoordV2 * cSysCoordV2::Lambert93()
{
   return new cSysCoordV1(cProj4::Lambert93());
}

cSysCoordV2 * cSysCoordV2::RTL(const cPt3dr & anOriInit,cSysCoordV2* aSysCoordPt)
{
     cPt3dr anOri =   (aSysCoordPt==0) ? anOriInit : aSysCoordPt->ToGeoC(anOriInit);

     return new cSysCoordV1(cSysCoord::RTL(ToMMV1(anOri)));
}

/*********************************************/
/*                                           */
/*            cChangSysCoordV2               */
/*                                           */
/*********************************************/



cChangSysCoordV2::cChangSysCoordV2(cSysCoordV2 * aSysInit,cSysCoordV2 * aSysTarget,tREAL8  aEpsDeriv)  :
    cDataInvertibleMapping<tREAL8,3> (cPt3dr::PCste(aEpsDeriv)),
    mSysInit      (aSysInit),
    mSysTarget    (aSysTarget)
{
}

cPt3dr cChangSysCoordV2::Value(const cPt3dr & aPInit) const 
{
	return mSysTarget->FromGeoC(mSysInit->ToGeoC(aPInit));
}

cPt3dr cChangSysCoordV2::Inverse(const cPt3dr & aPInit) const 
{
	return mSysInit->FromGeoC(mSysTarget->ToGeoC(aPInit));
}




};
