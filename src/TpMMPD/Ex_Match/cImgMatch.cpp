#include "cLSQTemplate.h"

cImgMatch::cImgMatch(string aName, cInterfChantierNameManipulateur *aICNM):
   mName (aName),
   mICNM (aICNM),
   mTif  (Tiff_Im::UnivConvStd(mICNM->Dir() + mName)),
   mSzIm (Pt2dr(mTif.sz())),
   mIm2D (1,1),
   mTIm2D(mIm2D),
   mCurImgetIm2D (1,1),
   mCurImgetTIm2D (mCurImgetIm2D)
{
    // Charge l'image
    //ELISE_COPY(mIm2D.all_pts(),mTif.in(),mIm2D.out());    //pas la peine de lire tout image
}

bool cImgMatch::GetImget(Pt2dr aP, Pt2dr aSzW)
{
    cout<<"Get Imget : "<<aP<<" "<<aSzW<<endl;
    //mCurImgetIm2D.Resize(Pt2di(0,0));   // est qu'on peut utilise ca pout supprimer Im2D ?
    Pt2dr aRab(2.0,2.0);
    Pt2dr aPtInf = aP - aSzW - aRab;
    Pt2dr aPtSup = aP + aSzW + aRab;
    Pt2dr aSzModif = aPtSup - aPtInf;
    cout<<aPtInf<<aPtSup<<endl;
    if (mIm2D.Inside(round_down(aPtInf)) && mIm2D.Inside(round_up(aPtSup)))
    {
        cout<<"Inside"<<endl;
        mCurImgetIm2D.Resize(round_up(aSzModif));
        ELISE_COPY(
                    mCurImgetIm2D.all_pts(),
                    trans(mTif.in(0),round_down(aP)),
                    mCurImgetIm2D.out()
                  );  // charger imaget avec un decal (de origin à la centre)
        return true;
    }
    else
        return false;
}
