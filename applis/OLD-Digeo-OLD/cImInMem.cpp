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
#include "Digeo.h"

namespace NS_ParamDigeo
{

/****************************************/
/*                                      */
/*             cTplImInMem               */
/*                                      */
/****************************************/

template <class Type> 
cTplImInMem<Type>::cTplImInMem
(
        cImDigeo & aIGlob,
        const Pt2di & aSz,
        GenIm::type_el aType,
        cTplOctDig<Type>  & anOct,
        double aResolOctaveBase,
        int    aKInOct,
        int    IndexSigma
) :
     cImInMem(aIGlob,aSz,aType,anOct,aResolOctaveBase,aKInOct,IndexSigma),
     mTOct (anOct),
     mIm(1,1),
     mTIm (mIm),
     mTMere (0),
     mTFille (0),
     mOrigOct (0),
     mData   (0)
{
    Resize(aSz+Pt2di(PackTranspo,0)); 
    Resize(aSz); 
}

template <class Type> void cTplImInMem<Type>::Resize(const Pt2di & aSz)
{
   mIm.Resize(aSz);
   mTIm = mIm;
   mSz = aSz;
   mData = mIm.data();
   // ELISE_COPY(mIm.all_pts(),1,mIm.out());
// std::cout << "SZ = " << aSz << " " << (void *)mIm.data_lin() << "\n";
}

template <class Type> bool cTplImInMem<Type>::InitRandom()
{
   if (! mAppli.SectionTest().IsInit())
      return false;
   const cSectionTest & aST = mAppli.SectionTest().Val();
   if (! aST.GenereAllRandom().IsInit())
      return false;


    const cGenereAllRandom & aGAR = aST.GenereAllRandom().Val();

   ELISE_COPY
   (
      mIm.all_pts(),
      255*(gauss_noise_1(aGAR.SzFilter())>0),
      mIm.out()
   );

   return true;
}
 
template <class Type> void cTplImInMem<Type>::LoadFile(Tiff_Im aFile,const Box2di & aBox)
{
    Resize(aBox.sz());
    ELISE_COPY(mIm.all_pts(), trans(aFile.in(),aBox._p0), mIm.out());
    if (
              mAppli.MaximDyn().ValWithDef(sizeof(Type)<=2) 
           && type_im_integral(mType) 
           && (!signed_type_num(mType))
       )
    {
       int aMinT,aMaxT;
       min_max_type_num(mType,aMinT,aMaxT);
       aMaxT = ElMin(aMaxT-1,1<<19);  // !!! LIES A NbShift ds PyramideGaussienne
       tBase aMul = 0;

       if(mAppli.ValMaxForDyn().IsInit())
       {
           tBase aMaxTm1 = aMaxT-1;
           aMul = round_ni(aMaxTm1/mAppli.ValMaxForDyn().Val()) ;
            
           for (int aY=0 ; aY<mSz.y ; aY++)
           {
               Type * aL = mData[aY];
               for (int aX=0 ; aX<mSz.x ; aX++)
                   aL[aX] = ElMin(aMaxTm1,tBase(aL[aX]*aMul));
           }
              std::cout << " Multiplieur in : " << aMul  << "\n";
       }
       else
       {
           Type aMaxV = aMinT;
           for (int aY=0 ; aY<mSz.y ; aY++)
           {
              Type * aL = mData[aY];
              for (int aX=0 ; aX<mSz.x ; aX++)
                  ElSetMax(aMaxV,aL[aX]);
           }
           aMul = (aMaxT-1) / aMaxV;
           if (mAppli.ShowTimes().Val())
           {
              std::cout << " Multiplieur in : " << aMul 
                        << " MaxVal " << tBase(aMaxV )
                        << " MaxType " << aMaxT 
                        << "\n";
           }
           if (aMul > 1)
           {
              for (int aY=0 ; aY<mSz.y ; aY++)
              {
                 Type * aL = mData[aY];
                 for (int aX=0 ; aX<mSz.x ; aX++)
                     aL[aX] *= aMul;
              }
              SauvIm("ReDyn_");
           }
       }
       mImGlob.SetDyn(aMul);

    }

   if (mAppli.SectionTest().IsInit())
   {
      const cSectionTest & aST = mAppli.SectionTest().Val();
      if (aST.GenereRandomRect().IsInit())
      {
         cGenereRandomRect aGRR = aST.GenereRandomRect().Val();
         ELISE_COPY(mIm.all_pts(),0, mIm.out());
         for (int aK= 0 ; aK < aGRR.NbRect() ; aK++)
         {
             Pt2di aP = aBox.RandomlyGenereInside() - aBox._p0;
             int aL = 1 +NRrandom3(aGRR.SzRect());
             int aH = 1 +NRrandom3(aGRR.SzRect());
             ELISE_COPY
             (
                   rectangle(aP-Pt2di(aL,aH),aP+Pt2di(aL,aH)),
                   255*NRrandom3(),
                   mIm.out()
             );
         }
      }
      if (aST.GenereCarroyage().IsInit())
      {
         cGenereCarroyage aGC = aST.GenereCarroyage().Val();
         ELISE_COPY(mIm.all_pts(),255*((FX/aGC.PerX()+FY/aGC.PerY())%2), mIm.out());
      }

      InitRandom();
   }
}

template <class Type> Im2DGen cTplImInMem<Type>::Im()
{
   return TIm();
}

template <class Type>  void  cTplImInMem<Type>::SetOrigOct(cTplImInMem<Type> * anOrig)
{
    mOrigOct = anOrig;
}

template <class Type>  void  cTplImInMem<Type>::SetMereSameDZ(cTplImInMem<Type> * aTMere)
{
    
/*
    ELISE_ASSERT((mType==aMere->TypeEl()),"cImInMem::SetMere type mismatch");
    tTImMem * aTMere = static_cast<tTImMem *>(aMere);
*/

    ELISE_ASSERT((mTMere==0) && (aTMere->mTFille==0),"cImInMem::SetMere");
    mTMere = aTMere;
    aTMere->mTFille = this;
}



template <class Type>  double  cTplImInMem<Type>::CalcGrad2Moy()
{
    double aRes = 0;
    for (int anY = 0 ; anY< mSz.y-1 ; anY++)
    {
        Type * aL0 = mData[anY];
        Type * aL1 = mData[anY+1];
        for (int anX = 0 ; anX< mSz.x-1 ; anX++)
        {
            aRes += ElSquare(double(aL0[anX]-aL0[anX+1]))+ElSquare(double(aL0[anX]-aL1[anX]));
        }
    }

    return aRes/((mSz.y-1)*(mSz.x-1));
}

// Force instanciation

InstantiateClassTplDigeo(cTplImInMem)






/****************************************/
/*                                      */
/*             cImInMem                 */
/*                                      */
/****************************************/

cImInMem::cImInMem
(
       cImDigeo & aIGlob,
       const Pt2di & aSz,
       GenIm::type_el aType,
       cOctaveDigeo & anOct,
       double aResolOctaveBase,
       int    aKInOct,
       int    IndexSigma
) :
    mAppli            (aIGlob.Appli()),
    mImGlob           (aIGlob),
    mOct              (anOct),
    mSz               (aSz),
    mType             (aType),
    mResolGlob        (anOct.Niv()),
    mResolOctaveBase  (aResolOctaveBase),
    mKInOct           (aKInOct),
    mIndexSigma       (IndexSigma),
    mMere             (0),
    mFille            (0),
    mKernelTot        (1,1.0)
{
 
}




void cImInMem::SetMere(cImInMem * aMere)
{
    ELISE_ASSERT((mMere==0) && (aMere->mFille==0),"cImInMem::SetMere");
    mMere = aMere;
    aMere->mFille = this;
}



void cImInMem::SauvIm(const std::string & aAdd)
{
   if (! mAppli.SauvPyram().IsInit())
      return;

   const cTypePyramide & aTP = mAppli.TypePyramide();
   cSauvPyram aSP = mAppli.SauvPyram().Val();
   std::string aDir =  mAppli.DC() + aSP.Dir().Val();
   ELISE_fp::MkDirSvp(aDir);

   std::string aNRes = ToString(mResolGlob);
   if (aTP.PyramideGaussienne().IsInit())
   {
       aNRes = aNRes + "_Sigma" + ToString(round_ni(100*mResolOctaveBase));
   }

   std::string aName =     aDir
                         + aAdd
                         + mAppli.ICNM()->Assoc1To2
                           (
                                aSP.Key().Val(),
                                mImGlob.Name(),
                                aNRes,
                                true
                           );

   if ((! ELISE_fp::exist_file(aName)) || (aSP.CreateFileWhenExist().Val()))
   {
      L_Arg_Opt_Tiff aLArgTiff = Tiff_Im::Empty_ARG;
      int aStrip = aSP.StripTifFile().Val();
      if (aStrip>0)
          aLArgTiff = aLArgTiff +  Arg_Tiff(Tiff_Im::AStrip(aStrip));

      if (aSP.Force8B().Val())
      {
          Tiff_Im::Create8BFromFonc(aName,mSz,Min(255,Im().in()*aSP.Dyn().Val()));
      }
      else
      {
          Tiff_Im::CreateFromIm(Im(),aName,aLArgTiff);
      }
   }

}

    // ACCESSOR 

GenIm::type_el  cImInMem::TypeEl() const { return mType; }
Pt2di cImInMem::Sz() const {return mSz;}
int cImInMem::RGlob() const {return mResolGlob;}
double cImInMem::ROct() const {return mResolOctaveBase;}
cImInMem *  cImInMem::Mere() {return mMere;}
cOctaveDigeo &  cImInMem::Oct() {return mOct;}



};



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
