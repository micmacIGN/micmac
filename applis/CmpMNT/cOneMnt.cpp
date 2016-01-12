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
// #include "anag_all.h"

#include "CmpMNT.h"

namespace NS_CmpMNT
{

cOneMnt::cOneMnt
(
        const cAppliCmpMnt& anAppli,
        const cMNT2Cmp & anArg,
        int aNum,
        const cCompareMNT & anArgGlob,
        cOneMnt * aPrec
)   :
   mAppli   (anAppli),
   mArgM    (anArg),
   mNum     (aNum),
   mArgG    (anArgGlob),
   mW       (0),
   mIm      (1,1),
   mTIm     (mIm),
   mNIm     (mArgM.NameIm()),
   mTif     (Tiff_Im::StdConv(mNIm)),
   mSzFile  (mTif.sz()),
   mNameXml (mArgM.NameXml().ValWithDef(StdPrefix(mNIm)+".xml")),
   mFOM     (StdGetObjFromFile<cFileOriMnt2>
                 (
		      mNameXml,
		      "include/XML_GEN/ParamChantierPhotogram.xml",
		      "FileOriMnt",
		      "FileOriMnt2"
		 )
            )
{
   if (aPrec)
   {
      ELISE_ASSERT
      (
             (mFOM.OriginePlani() ==  aPrec->mFOM.OriginePlani())
          && (mFOM.ResolutionPlani() == aPrec->mFOM.ResolutionPlani())
          && (mFOM.Geometrie() == aPrec->mFOM.Geometrie()),
	  "MNT INCOHERENTS"
      );
   }
}
Fonc_Num cOneMnt::Grad()
{
   Pt2dr aResol = mArgG.ResolutionPlaniTerrain();
   return deriche(mIm.in_proj(),1.0) / Virgule(aResol.x,aResol.y);
}

int cOneMnt::IdRef() const
{
  return mArgM.IdIsRef().Val();
}

Pt2di cOneMnt::SzFile() const
{
   return mSzFile;
}


void cOneMnt::Load(const cOneZoneMnt & aZone)
{
   mCurZone = & aZone;
   Box2di aBox = aZone.Box();
   

   mIm  =  Im2D_REAL4(aBox.sz().x,aBox.sz().y);
   mTIm = mIm;
   ELISE_COPY
   (
       mIm.all_pts(),
       mFOM.OrigineAlti()+trans(mTif.in(),aBox._p0)*mFOM.ResolutionAlti(),
       mIm.out()
   );

   if (mArgG.VisuInter().Val())
   {
      mW = Video_Win::PtrWStd(aBox.sz());
      ELISE_COPY
      (
          mIm.all_pts(),
	  mIm.in(),
	  mW->ocirc()
      );
      ELISE_COPY
      (
          select(mIm.all_pts(),! mCurZone->Masq().in()),
	  128,
	  mW->ogray()
      );
      std::string aTit = ShortName();
      mW->set_title(aTit.c_str());

   }
}

void cOneMnt::CalcVMoy(double & aZMoy,double & aPenteMoy)
{
    double aS1;

    Pt2dr aResol = mArgG.ResolutionPlaniTerrain();
    ELISE_COPY
    (
        mIm.all_pts(),
          Virgule(1,mIm.in(),polar(Grad(),0).v0())
	* mCurZone->Masq().in(),
	Virgule
	(
            sigma(aS1),
            sigma(aZMoy),
            sigma(aPenteMoy)
        )
    );
    aZMoy /= aS1;
    aPenteMoy /= aS1;
}

std::string cOneMnt::ShortName()
{
   return  mArgM.ShorName().ValWithDef("Num " + ToString(mNum));
}



Video_Win *   cOneMnt::W()
{
   return mW;
}


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
