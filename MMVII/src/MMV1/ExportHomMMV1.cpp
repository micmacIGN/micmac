#include "V1VII.h"

#include "MMVII_util.h"
#include "MMVII_MeasuresIm.h"

namespace MMVII
{

#if (MMVII_KEEP_LIBRARY_MMV1)
cHomogCpleIm  ToMMVII(const cNupletPtsHomologues &  aNUp) 
{

	return  cHomogCpleIm(ToMMVII(aNUp.P1()),ToMMVII(aNUp.P2()));
}

/*   ************************************************* */
/*                                                     */
/*         cConvertHomV1                               */
/*                                                     */
/*   ************************************************* */

class cImportHomV1 : public cInterfImportHom
{
      public :
          cImportHomV1(const std::string & aDir,const std::string & aSubDir,const std::string & aExt="dat");

	  void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const override;
          bool   HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const override;


      private :

          std::string NameHom(const std::string & aNameIm1,const std::string & aNameIm2) const;
          std::string  mDir;
          std::string  mSubDir;
          std::string  mExt;
          std::string  mKHIn;
          cElemAppliSetFile                 mEASF;
          cInterfChantierNameManipulateur * mICNM ;
};


cImportHomV1::cImportHomV1(const std::string & aDir,const std::string & aSubDir,const std::string & aExt) :
   mDir     (aDir) ,
   mSubDir  (aSubDir) ,
   mExt     (aExt),
   mKHIn    (std::string("NKS-Assoc-CplIm2Hom@") + mSubDir  +  std::string("@") +  mExt),
   mEASF    (mDir),
   mICNM    (mEASF.mICNM)
{
}


std::string cImportHomV1::NameHom(const std::string & aNameIm1,const std::string & aNameIm2) const
{
    return mDir +  mICNM->Assoc1To2(mKHIn,aNameIm1,aNameIm2,true);
}

bool   cImportHomV1::HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const 
{
    return ExistFile(NameHom(aNameIm1,aNameIm2));
}

void  cImportHomV1::GetHom(cSetHomogCpleIm & aPackV2,const std::string & aNameIm1,const std::string & aNameIm2) const
{
     aPackV2.Clear();
     ElPackHomologue aPackV1 = ElPackHomologue::FromFile(NameHom(aNameIm1,aNameIm2));

     for (ElPackHomologue::tIter aItV1 = aPackV1.begin() ; aItV1!=aPackV1.end() ; aItV1++)
     {
         aPackV2.Add(ToMMVII(*aItV1));
     }
}

/*   ************************************************* */
/*                                                     */
/*                 cInterfImportHom                    */
/*                                                     */
/*   ************************************************* */

cInterfImportHom * cInterfImportHom::CreateImportV1(const std::string&aDir,const std::string&aSubDir,const std::string&aExt)
{
	return new cImportHomV1(aDir,aSubDir,aExt);
}
#else // MMVII_KEEP_LIBRARY_MMV1

cInterfImportHom * cInterfImportHom::CreateImportV1(const std::string&aDir,const std::string&aSubDir,const std::string&aExt)
{
        MMVII_INTERNAL_ERROR("No CreateImportV1 ");
	return nullptr;
}
#endif // MMVII_KEEP_LIBRARY_MMV1


cInterfImportHom::~cInterfImportHom()
{
}





};
