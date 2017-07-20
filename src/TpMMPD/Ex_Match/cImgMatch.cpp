#include "cLSQTemplate.h"

cImgMatch::cImgMatch(string aName, cInterfChantierNameManipulateur *aICNM):
   mName (aName),
   mICNM (aICNM),
   mTif  (Tiff_Im::UnivConvStd(mICNM->Dir() + mName)),
   mIm2D (1,1),
   mTIm2D(mIm2D)

{
    // Charge l'image
    mIm2D.Resize(mTif.sz());
    ELISE_COPY(mIm2D.all_pts(),mTif.in(),mIm2D.out());
}

