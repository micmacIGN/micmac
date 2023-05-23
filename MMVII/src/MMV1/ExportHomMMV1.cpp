#include "V1VII.h"
#include "MMVII_util.h"

namespace MMVII
{

/*   ************************************************* */
/*                                                     */
/*         cConvertHomV1                               */
/*                                                     */
/*   ************************************************* */

/*
class cConvertHomV1
{
      public :
          cConvertHomV1(const std::string & aDir,const std::string & aSubDir,const std::string & anExt);

      private :

          std::string NameHom(const std::string & aNameIm1,const std::string & aNameIm2) const;

          std::string  mDir;
          std::string  mSubDir;
          std::string  mExt;
          std::string  mKHIn;
          cElemAppliSetFile                 mEASF;
          cInterfChantierNameManipulateur * mICNM ;
};
*/


#if (0)
cConvertHomV1::cConvertHomV1(const std::string & aDir,const std::string & aSubDir,const std::string & anExt) :
   mDir     (aDir) ,
   mSubDir  (aSubDir) ,
   mExt     (anExt) ,
   mKHIn    (std::string("NKS-Assoc-CplIm2Hom@") +  mSubDir +  std::string("@") +  mExt),
   mEASF    (mDir),
   mICNM    (mEASF.mICNM)
{
}


std::string cConvertHomV1::NameHom(const std::string & aNameIm1,const std::string & aNameIm2) const
{
    return mDir +  mICNM->Assoc1To2(mKHIn,aNameIm1,aNameIm2,true);
}
#endif



};
