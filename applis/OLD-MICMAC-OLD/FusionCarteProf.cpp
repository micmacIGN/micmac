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
#include "general/all.h"
#include "private/all.h"
#include "MICMAC.h"
#include "ext_stl/numeric.h"
#include "im_tpl/algo_dist32.h"

using namespace NS_ParamMICMAC;
using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

/*********************************************************************/
/*                                                                   */
/*                       cFusionCarteProf                            */
/*                                                                   */
/*********************************************************************/

template <class Type> class  cLoadedCP;
template <class Type> class cFusionCarteProf;

class cElPile;

    //=================================

class cElPile
{
    public :
        cElPile (double aZ,double aPds) :
           mZ  (aZ),
           mP  (aPds)
        {
        }

        const double & Z() const {return mZ;}
        const double & P() const {return mP;}
    private :
        double mZ;
        double mP;
};

bool operator < (const cElPile & aP1, const cElPile & aP2) {return aP1.Z() < aP2.Z();}

class PPile
{
    public :
       double operator ()(const cElPile & aPile) const {return aPile.P();}
};
static PPile ThePPile;
class ZPile
{
    public :
       typedef double tValue;
       double operator ()(const cElPile & aPile) const {return aPile.Z();}
};
static ZPile TheZPile;





template <class Type> class  cLoadedCP
{
    public :

        typedef  Type  tNum;
        typedef  typename El_CTypeTraits<tNum>::tBase  tNBase;

        cLoadedCP(cFusionCarteProf<Type> &, const std::string & anId,const std::string & aFus);
        const cXML_ParamNuage3DMaille & Nuage() {return mNuage;}

        cElPile  ElPile(const Pt2dr &) const;

        void  SetSz(const Pt2di & aSz);
        bool  ReLoad(const Box2dr & aBoxTer) ;

    private :

        cFusionCarteProf<Type>  & mFCP;
        const cParamFusionMNT & mParam;
        cInterfChantierNameManipulateur * mICNM;

        std::string   mNameNuage;
        cXML_ParamNuage3DMaille  mNuage;
        cImage_Profondeur        mIP;
        ElAffin2D                mAfM2CGlob;
        ElAffin2D                mAfC2MGlob;
        ElAffin2D                mAfM2CCur;
        ElAffin2D                mAfC2MCur;
        std::string              mDirNuage;

        std::string        mNameCP;
        Tiff_Im            mTifCP;
        Pt2di              mSzGlob;
        Box2di             mBoxImGlob;
        Pt2di              mSzCur;
        Box2di             mBoxImCur;
        Im2D<tNum,tNBase>  mImCP;
        TIm2D<tNum,tNBase> mTImCP;

        std::string        mNameMasq;
        Tiff_Im            mTifMasq;
        Im2D_Bits<1>       mImMasq;
        TIm2DBits<1>       mTImMasq;

        bool               mHasCorrel;
        std::string        mNameCorrel;
        Im2D_U_INT1        mImCorrel;
        TIm2D<U_INT1,INT>  mTImCorrel;

};




template <class Type> class cFusionCarteProf
{
     public :
          typedef  Type  tNum;
          typedef  typename El_CTypeTraits<tNum>::tBase  tNBase;

          cFusionCarteProf(const cResultSubstAndStdGetFile<cParamFusionMNT>  & aP);
          const cParamFusionMNT & Param() {return mParam;}
          cInterfChantierNameManipulateur *ICNM() {return mICNM;}
     private :

          void DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut);
          void DoOneFusion(const std::string &);
          void DoCalc();
          void DoMake();

          cParamFusionMNT mParam;
          double          mPerc;
          cInterfChantierNameManipulateur * mICNM;
          std::vector<std::string>          mGenRes;
          bool                              mCalledByMkf;
          std::vector<cLoadedCP<Type> *>          mVC;
          std::vector<cLoadedCP<Type> *>          mVCL;
          cXML_ParamNuage3DMaille                 mNuage;
          cImage_Profondeur *                     mIP;
          ElAffin2D                               mAfM2CGlob;
          ElAffin2D                               mAfC2MGlob;
          ElAffin2D                               mAfM2CCur;
          ElAffin2D                               mAfC2MCur;
          Pt2di                                   mSzGlob;
          Pt2di                                   mSzCur;
          std::string                             mNameTif;
          std::string                             mNameMasq;

          
};


/**********************************************************************/
/*                                                                    */
/*                      cLoadedCP                                     */
/*                                                                    */
/**********************************************************************/

template <class Type>  cLoadedCP<Type>::cLoadedCP(cFusionCarteProf<Type> & aFCP, const std::string & anId,const std::string & aFus) :
  mFCP     (aFCP),
  mParam   (aFCP.Param()),
  mICNM    (aFCP.ICNM()),

  mNameNuage  (mICNM->Dir()+mICNM->Assoc1To2(mParam.KeyNuage(),anId,aFus,true)),
  mNuage      (StdGetObjFromFile<cXML_ParamNuage3DMaille>
                 (
                     mNameNuage,
                     StdGetFileXMLSpec("SuperposImage.xml"),
                     "XML_ParamNuage3DMaille",
                     "XML_ParamNuage3DMaille"
                 )
              ),
  mIP         (mNuage.PN3M_Nuage().Image_Profondeur().Val()),
  mAfM2CGlob  (Xml2EL(mNuage.Orientation().OrIntImaM2C())),
  mAfC2MGlob  (mAfM2CGlob.inv()),
  mDirNuage   (DirOfFile(mNameNuage)),

  mNameCP  (mDirNuage+mIP.Image()),
  mTifCP   (Tiff_Im::StdConv(mNameCP)),
  mSzGlob    (mTifCP.sz()),
  mBoxImGlob (Pt2di(0,0),mSzGlob),
  mImCP    (1,1),
  mTImCP   (mImCP),

  mNameMasq (mDirNuage+mIP.Masq()),
  mTifMasq  (Tiff_Im::StdConv(mNameMasq)),
  mImMasq   (1,1),
  mTImMasq   (mImMasq),
  mHasCorrel (mIP.Correl().IsInit()),
  mNameCorrel (mHasCorrel ? mDirNuage+mIP.Correl().Val() : ""),
  mImCorrel   (1,1),
  mTImCorrel  (mImCorrel)

{
}
/*
   mImCP.Resize(mSz);
   mTImCP = TIm2D<tNum,tNBase>(mImCP);
   ELISE_COPY(mImCP.all_pts(),mTifCP.in(),mImCP.out());

   mImMasq = Im2D_Bits<1>(mSz.x,mSz.y);
   ELISE_COPY(mImMasq.all_pts(),mTifMasq.in(),mImMasq.out());
   mTImMasq = TIm2DBits<1>(mImMasq);

   if (mHasCorrel)
   {
       Tiff_Im aTifCorrel = Tiff_Im::StdConv(mNameCorrel);
       mImCorrel.Resize(mSz);
       ELISE_COPY(mImCorrel.all_pts(),aTifCorrel.in(),mImCorrel.out());
       mTImCorrel = Im2D<U_INT1,INT>(mImCorrel);
   }
   std::cout << mNameCP << " " << mSz << "\n";
*/


template <class Type> void  cLoadedCP<Type>::SetSz(const Pt2di & aSz)
{
   mSzCur = aSz;

   mImCP.Resize(mSzCur);
   mTImCP = TIm2D<tNum,tNBase>(mImCP);

   mImMasq = Im2D_Bits<1>(mSzCur.x,mSzCur.y);
   mTImMasq = TIm2DBits<1>(mImMasq);

   if (mHasCorrel)
   {
       mImCorrel.Resize(mSzCur);
       mTImCorrel = Im2D<U_INT1,INT>(mImCorrel);
   }

}

template <class Type> bool  cLoadedCP<Type>::ReLoad(const Box2dr & aBoxTer) 
{
   // Box2dr aRBoxImCur = aBoxTer.BoxImage(mAfM2CGlob);
   mBoxImCur =  R2I(aBoxTer.BoxImage(mAfM2CGlob));
   if (InterVide(mBoxImCur,mBoxImGlob))
   {
       SetSz(Pt2di(1,1));
       return false;
   }

   mBoxImCur = Inf(mBoxImCur,mBoxImGlob);
   SetSz(mBoxImCur.sz());

   mAfM2CCur =  ElAffin2D::trans(-Pt2dr(mBoxImCur._p0)) * mAfM2CGlob;
   mAfC2MCur = mAfM2CCur.inv();

   ELISE_COPY(mImCP.all_pts(),trans(mTifCP.in(),mBoxImCur._p0),mImCP.out());
   ELISE_COPY(mImMasq.all_pts(),trans(mTifMasq.in(),mBoxImCur._p0),mImMasq.out());
   if (mHasCorrel)
   {
       Tiff_Im aTifCorrel = Tiff_Im::StdConv(mNameCorrel);
       ELISE_COPY(mImCorrel.all_pts(),trans(aTifCorrel.in(),mBoxImCur._p0),mImCorrel.out());
   }
   return true;
}

template <class Type> cElPile  cLoadedCP<Type>::ElPile(const Pt2dr & aPTer) const
{
   Pt2dr aPIm = mAfM2CCur(aPTer);
   double aPds = mTImMasq.get(round_ni(aPIm),0);
   double aZ = 0;
   if (aPds > 0)
   {
       if (mHasCorrel)
       {
           aPds = mTImCorrel.getprojR(aPIm);
           aPds = (aPds -128.0) / 128.0;

           if (aPds<0) aPds = (aPds+1) / 100.0;
           else if (aPds<0.5)
               aPds = 0.01 + aPds;
           else 
               aPds = 0.51 + (aPds-0.5) * 10;
           aPds = ElMax(0.0,aPds);
           aPds = pow(aPds,2);

           if (aPds < 1e-4) aPds = 0.0;
       }
   }

   if (aPds > 0)
   {
       aZ = mIP.OrigineAlti() +  mTImCP.getprojR(aPIm) * mIP.ResolutionAlti();
   }
   return cElPile(aZ,aPds);
}
/*
*/



/**********************************************************************/
/*                                                                    */
/*                      cFusionCarteProf                              */
/*                                                                    */
/**********************************************************************/

template <class Type> void cFusionCarteProf<Type>::DoOneFusion(const std::string & anId)
{
    std::string aNameNuage = mICNM->Dir() + mICNM->Assoc1To1(mParam.KeyResult(),anId,true);

    mNameTif = StdPrefix(aNameNuage)+ ".tif";
    std::cout << anId  << "=> " << mNameTif<< "\n";
    mNameMasq = StdPrefix(aNameNuage)+ "_Masq.tif";


    std::vector<std::string> aStrFus = GetStrFromGenStrRel(mICNM,mParam.GenereInput(),anId);

    if (aStrFus.size() == 0)
    {
        std::cout << "FOR ID = " << anId  << "\n";
        ELISE_ASSERT(false,"No data in DoOneFusion");
    }

    for (int aK=0 ; aK<int(aStrFus.size()) ; aK++)
    {
          mVC.push_back(new cLoadedCP<Type>(*this,anId,aStrFus[aK]));
    }

    if (mParam.ModeleNuageResult().IsInit())
    {
       mNuage = StdGetObjFromFile<cXML_ParamNuage3DMaille>
                (
                     mParam.ModeleNuageResult().Val(),
                     StdGetFileXMLSpec("SuperposImage.xml"),
                     "XML_ParamNuage3DMaille",
                     "XML_ParamNuage3DMaille"
                );
    }
    else
    {
         mNuage = mVC[0]->Nuage();
    }
    mIP = &(mNuage.Image_Profondeur().Val());
    mIP->Image() = NameWithoutDir(mNameTif) ;
    mIP->Masq() =  NameWithoutDir(mNameMasq);
    mIP->Correl().SetNoInit();
    
    mAfM2CGlob  = Xml2EL(mNuage.Orientation().OrIntImaM2C());
    mAfC2MGlob = mAfM2CGlob.inv();
    mSzGlob = mNuage.NbPixel();


   cDecoupageInterv2D aDecoup = cDecoupageInterv2D::SimpleDec
                                (
                                     mSzGlob,
                                     mParam.SzDalles().Val(),
                                     mParam.RecouvrtDalles().Val()
                                );

   for (int aKI=0 ; aKI<aDecoup.NbInterv() ; aKI++)
   {
       DoOneBloc
       (
           aDecoup.NbInterv()-aKI,
           aDecoup.KthIntervIn(aKI),
           aDecoup.KthIntervOut(aKI)
       );
   }

   MakeFileXML(mNuage,aNameNuage);
   DeleteAndClear(mVC);
}

template <class Type> void cFusionCarteProf<Type>::DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut)
{
   std::cout << "RESTE " << aKB <<   " BLOCS \n";
   mAfM2CCur =  ElAffin2D::trans(-Pt2dr(aBoxIn._p0)) * mAfM2CGlob ;
   mAfC2MCur = mAfM2CCur.inv();

   mSzCur = aBoxIn.sz();

   Im2D<tNum,tNBase>  aImFus(mSzCur.x,mSzCur.y);
   TIm2D<tNum,tNBase> aTImFus(aImFus);


   Im2D_Bits<1>       aImMasq(mSzCur.x,mSzCur.y);
   TIm2DBits<1>       aTImMasq(aImMasq);

   Box2di aBoxInLoc(Pt2di(0,0),mSzCur);
   Box2dr aBoxTer = aBoxInLoc.BoxImage(mAfC2MCur);


   mVCL.clear();
   for (int aK=0 ; aK<int(mVC.size()) ; aK++)
   {
       if (mVC[aK]->ReLoad(aBoxTer))
          mVCL.push_back(mVC[aK]);
   }

   std::vector<cElPile> aPCel;
   Pt2di aQ0;
   for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
   {
        for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
        {
            Pt2dr aT0 = mAfC2MCur(Pt2dr(aQ0));
            aPCel.clear();
            for (int aKI=0 ; aKI<int(mVCL.size()); aKI++)
            {
                cElPile anEl = mVCL[aKI]->ElPile(aT0);
                if (anEl.P()>0)
                {
                   aPCel.push_back(anEl);
                }
            }
            int Ok= 0;
            double aZ=0;
            if (aPCel.size() >0)
            {
                std::sort(aPCel.begin(),aPCel.end());

                double aSomP = SomPerc(aPCel,ThePPile);
                aZ = GenValPdsPercentile(aPCel,mPerc,TheZPile,ThePPile,aSomP);
                aZ = (aZ -mIP->OrigineAlti()) / mIP->ResolutionAlti();
                Ok=1;
            }
            else
            {
            }
            aTImFus.oset(aQ0,aZ);
            aTImMasq.oset(aQ0,Ok);
        }
   }

   if (1)
   {
        Im2D_Bits<1>       aIm1(mSzCur.x,mSzCur.y,1);
        TIm2DBits<1>       aTIm1(aIm1);
        ComplKLipsParLBas ( aTIm1, aTImMasq, aTImFus);
   }

   bool IsModified;
   Tiff_Im  aTifFusion = Tiff_Im::CreateIfNeeded
                         (
                             IsModified,
                             mNameTif,
                             mSzGlob,
                             aImFus.TypeEl(),
                             Tiff_Im::No_Compr,
                             Tiff_Im::BlackIsZero
                         );


   ELISE_COPY
   (
       rectangle(aBoxOut._p0,aBoxOut._p1),
       trans(aImFus.in(),-aBoxIn._p0),
       aTifFusion.out()
   );

   Tiff_Im  aTifMasq = Tiff_Im::CreateIfNeeded
                         (
                             IsModified,
                             mNameMasq,
                             mSzGlob,
                             aImMasq.TypeEl(),
                             Tiff_Im::No_Compr,
                             Tiff_Im::BlackIsZero
                         );
   ELISE_COPY
   (
       rectangle(aBoxOut._p0,aBoxOut._p1),
       trans(aImMasq.in(),-aBoxIn._p0),
       aTifMasq.out()
   );

   // std::cout << "ENnnndd \n"; getchar();
}
/*
{

    Im2D<tNum,tNBase>  aImFus(mSz.x,mSz.y);
    TIm2D<tNum,tNBase> aTImFus(aImFus);

    Im2D_Bits<1>       aImMasq(mSz.x,mSz.y);
    TIm2DBits<1>       aTImMasq(aImMasq);
   // Prepare le decoupage par bloc 
    mVCL = mVC;

    std::vector<cElPile> aPCel;
    Pt2di aQ0;
    for (aQ0.y = 0 ; aQ0.y < mSz.y; aQ0.y++)
    {
        if ((aQ0.y%100)==0) 
             std::cout << "Reste " << mSz.y - aQ0.y << "\n";
        for (aQ0.x = 0 ; aQ0.x < mSz.x; aQ0.x++)
        {
            Pt2dr aT0 = mAfC2M(Pt2dr(aQ0));
            aPCel.clear();
            for (int aKI=0 ; aKI<int(mVCL.size()); aKI++)
            {
                cElPile anEl = mVCL[aKI]->ElPile(aT0);
                if (anEl.P()>0)
                {
                   aPCel.push_back(anEl);
                }
            }
            int Ok= 0;
            double aZ=0;
            if (aPCel.size() >0)
            {
                std::sort(aPCel.begin(),aPCel.end());

                double aSomP = SomPerc(aPCel,ThePPile);
                aZ = GenValPdsPercentile(aPCel,50.0,TheZPile,ThePPile,aSomP);
                aZ = (aZ -mIP->OrigineAlti()) / mIP->ResolutionAlti();
                Ok=1;
            }
            else
            {
            }
            aTImFus.oset(aQ0,aZ);
            aTImMasq.oset(aQ0,Ok);
        }
    }

    Tiff_Im::CreateFromIm(aImFus,aNameTif);
    Tiff_Im::CreateFromIm(aImMasq,mNameMasq);


}
*/

template <class Type> void cFusionCarteProf<Type>::DoCalc()
{
   for (int aKS=0 ; aKS<int(mGenRes.size()) ; aKS++)
   {
       DoOneFusion(mGenRes[aKS]);
   }
    
}

template <class Type> void cFusionCarteProf<Type>::DoMake()
{
}



template <class Type> cFusionCarteProf<Type>::cFusionCarteProf
(
       const cResultSubstAndStdGetFile<cParamFusionMNT>  & aParam
)  :
     mParam   (*(aParam.mObj)),
     mPerc    (mParam.PercFusion().Val()),
     mICNM    (aParam.mICNM),
     mGenRes  (GetStrFromGenStr(mICNM,mParam.GenereRes())),
     mCalledByMkf  (mParam.InterneParalDoOnlyOneRes().Val() != "")
{
    if (mCalledByMkf)
    {
         mGenRes.clear();
         mGenRes.push_back(mParam.InterneParalDoOnlyOneRes().Val());
    }
    if ((mParam.ParalMkF().Val() != "") && (! mCalledByMkf))
    {
        DoMake();
    }
    else
    {
        DoCalc();
    }
}




/*
*/

#if ELISE_windows
int __cdecl main(int argc,char ** argv)
#else
int main(int argc,char ** argv)
#endif 
{
  ELISE_ASSERT(argc>=2,"Not Enough args to FusionMNT.cpp");
  MMD_InitArgcArgv(argc,argv);

  Tiff_Im::SetDefTileFile(50000);


  cResultSubstAndStdGetFile<cParamFusionMNT> aP2
                                           (
                                              argc-2,argv+2,
                                              argv[1],
                                              StdGetFileXMLSpec("SuperposImage.xml"),
                                              "ParamFusionMNT",
                                              "ParamFusionMNT",
                                              "WorkDir",
                                              "FileChantierNameDescripteur"
                                           );

  cFusionCarteProf<INT2>  aFCP(aP2);
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
