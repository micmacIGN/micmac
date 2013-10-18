/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"

#define DEF_OFSET -12349876

class cCEM_OneIm;
class cCoherEpi_main;


class cCEM_OneIm
{ 
     public :
          cCEM_OneIm (cCoherEpi_main * ,const std::string &,const Box2di & aBox,bool Visu);
          Box2dr BoxIm2();
          void SetConj(cCEM_OneIm *);

          Pt2dr ToIm2(const Pt2dr & aP)
          {
                 return RoughToIm2(aP+mRP0,mTPx.getprojR(aP))- mConj->mRP0;
          }
          Pt2dr AllerRetour(const Pt2dr & aP)
          {
                return mConj->ToIm2(ToIm2(aP));
          }
          Im2D_U_INT1  ImAR(double aMul);

     private :
          virtual  Pt2dr  RoughToIm2(const Pt2dr & aP,const double & aPx)
          {
             return Pt2dr(aP.x+aPx,aP.y);
          }

          Output VGray() {return mW ?  mW->ogray() : Output::onul(1) ;}

          cCoherEpi_main * mCoher;
          cCpleEpip *      mCple;
          std::string      mDir;
          std::string      mName;
          std::string      mNameEpi;
          Tiff_Im          mTifIm;
          Box2di           mBox;
          Pt2di            mSz;
          Pt2di            mP0;
          Pt2dr            mRP0;
          Im2D_U_INT2      mIm;
          std::string      mNamePx;
          Tiff_Im          mTifPx;
          Im2D_REAL4       mImPx;
          TIm2D<REAL4,REAL8> mTPx;
          std::string      mNameMasq;
          Tiff_Im          mTifMasq;
          Im2D_Bits<1>     mImMasq;
          TIm2DBits<1>     mTMasq;
          Video_Win *      mW;
          cCEM_OneIm *     mConj;
};




class cCoherEpi_main
{
     public :
        friend class cCEM_OneIm;
        cCoherEpi_main (int argc,char ** argv);

        

     private  :
        Box2di       mBoxIm1;
        Pt2di        mIntY1;
        std::string  mNameIm1;
        std::string  mNameIm2;
        std::string  mOri;
        std::string  mDir;
        cCpleEpip *  mCple;
        cCEM_OneIm  * mIm1; 
        cCEM_OneIm  * mIm2; 

        int           mDeZoom;
        int           mNumPx;
        int           mNumMasq;
        bool          mVisu;
        double        mSigmaP;
};

/*******************************************************************/
/*                                                                 */
/*                cCEM_OneIm                                       */
/*                                                                 */
/*******************************************************************/

cCEM_OneIm::cCEM_OneIm
(
    cCoherEpi_main *       aCoher,
    const std::string &    aName,
    const Box2di      &    aBox,
    bool                   aVisu
)  :
   mCoher     (aCoher),
   mCple      (mCoher->mCple),
   mDir       (mCoher->mDir),
   mName      (aName),
   mNameEpi   (mDir+ mCple->LocNameImEpi(mName)),
   mTifIm     (mNameEpi.c_str()),
   mBox       (Inf(aBox,Box2di(Pt2di(0,0),mTifIm.sz()))),
   mSz        (mBox.sz()),
   mP0        (mBox._p0),
   mRP0       (mP0),
   mIm        (mSz.x,mSz.y),
   mNamePx    (mDir+mCple->LocPxFileMatch(mName,mCoher->mNumPx,mCoher->mDeZoom)),
   mTifPx     (mNamePx.c_str()),
   mImPx      (mSz.x,mSz.y),
   mTPx       (mImPx),
   mNameMasq  (mDir+mCple->LocMasqFileMatch(mName,mCoher->mNumMasq)),
   mTifMasq   (mNameMasq.c_str()),
   mImMasq    (mSz.x,mSz.y),
   mTMasq     (mImMasq),
   mW         (aVisu ? Video_Win::PtrWStd(mSz) : 0),
   mConj      (0)
   
{
    ELISE_COPY ( mIm.all_pts(),trans(mTifPx.in(),mP0),mImPx.out());
    ELISE_COPY ( mIm.all_pts(),trans(mTifMasq.in(),mP0),mImMasq.out());
    ELISE_COPY ( mIm.all_pts(),trans(mTifIm.in(),mP0),mIm.out() | VGray());
}

void cCEM_OneIm::SetConj(cCEM_OneIm * aConj)
{
   mConj = aConj;
   mConj->mConj = this;
}

Box2dr cCEM_OneIm::BoxIm2()
{
   Pt2dr aP0(1e9,1e9);
   Pt2dr aP1(-1e9,-1e9);

   Pt2di aPIm;
   for (aPIm.x=0 ; aPIm.x<mSz.x ; aPIm.x++)
   {
       for (aPIm.y=0 ; aPIm.y<mSz.y ; aPIm.y++)
       {
          if (mTMasq.get(aPIm))
          {
             Pt2dr aPIm2 = RoughToIm2(Pt2dr(aPIm+mP0),mTPx.get(aPIm));
             aP0.SetInf(aPIm2);
             aP1.SetSup(aPIm2);
          }
       }
   }

   return Box2dr(aP0,aP1);
}

Im2D_U_INT1  cCEM_OneIm::ImAR(double aMul)
{
   Im2D_U_INT1 aRes(mSz.x,mSz.y);
   TIm2D<U_INT1,INT> aTRes(aRes);

   double aSigma = mCoher->mSigmaP;
   Pt2di aPIm;
   for (aPIm.x=0 ; aPIm.x<mSz.x ; aPIm.x++)
   {
       for (aPIm.y=0 ; aPIm.y<mSz.y ; aPIm.y++)
       {
           Pt2dr aQ = AllerRetour(Pt2dr(aPIm));
           double aDist = euclid(Pt2dr(aPIm)-aQ);
           double aPds = (aSigma/(aSigma+aDist)) * 255;
           aTRes.oset(aPIm,ElMin(255,round_ni(aPds)));
       }
   }
   if (mW) 
   {
      ELISE_COPY(aRes.all_pts(),aRes.in(),VGray());
   }


   return aRes;
}

/*******************************************************************/
/*                                                                 */
/*                cCoherEpi_main                                   */
/*                                                                 */
/*******************************************************************/

cCoherEpi_main::cCoherEpi_main (int argc,char ** argv) :
    mDir      ("./"),
    mDeZoom   (1),
    mNumPx    (9),
    mNumMasq  (8),
    mVisu     (false),
    mSigmaP   (1.5)
    
{
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(mNameIm1,"Name Im1") 
                    << EAMC(mNameIm2,"Name Im2") 
                    << EAMC(mOri,"Orientation") ,
	LArgMain()  << EAM(mDir,"Dir",true)
                    << EAM(mBoxIm1,"Box",true)
                    << EAM(mIntY1,"YBox",true)
                    << EAM(mVisu,"Visu",true)
    );	

    mCple = StdCpleEpip(mDir,mOri,mNameIm1,mNameIm2);

   std::cout << "Name EPI1 " << mCple->LocDirMatch(mNameIm1) << "\n";
   std::cout << "Name EPI2 " << mCple->LocDirMatch(mNameIm2) << "\n";

   std::string aNameEpi1 = mDir+ mCple->LocNameImEpi(mNameIm1);
   Tiff_Im aTF1(aNameEpi1.c_str());
   Pt2di aSz1 = aTF1.sz();

   if (!EAMIsInit(&mBoxIm1))
   {
         if (EAMIsInit(&mIntY1))
         {
             mBoxIm1 = Box2di(Pt2di(0,mIntY1.x),Pt2di(aSz1.x,mIntY1.y));
         }
         else
         {
             mBoxIm1 = Box2di(Pt2di(0,0),aSz1);
         }
   }
   mIm1 = new cCEM_OneIm(this,mNameIm1,mBoxIm1,mVisu);
   Box2di aBoxIm2 = R2ISup(mIm1->BoxIm2());
   mIm2 = new cCEM_OneIm(this,mNameIm2,aBoxIm2,mVisu);
   std::cout << "Box2 " <<aBoxIm2._p0 << " " << aBoxIm2._p1 << "\n";
   mIm1->SetConj(mIm2);

   Im2D_U_INT1 anAR1 = mIm1->ImAR(20.0);
   Tiff_Im::Create8BFromFonc(mDir+"AR1.tif",anAR1.sz(),anAR1.in());
}


int CoherEpi_main(int argc,char ** argv)
{
    cCoherEpi_main aCEM(argc,argv);

    return 0;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
