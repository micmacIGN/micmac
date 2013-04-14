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

/******************************************/
/*                                        */
/*         cVisuCaracDigeo                */
/*                                        */
/******************************************/


cVisuCaracDigeo::cVisuCaracDigeo
(
     cAppliDigeo & anAppli,
     Pt2di aSz,
     int aZ,
     Fonc_Num aF,
     const cParamVisuCarac & aParam
) :
    mAppli (anAppli),
    mSz1   (aSz),
    mZ     (aZ),
    mSzZ   (mSz1*aZ),
    mIm    (mSzZ.x,mSzZ.y),
    mTIm   (mIm),
    mParam (aParam),
    mDynG  (aParam.DynGray().Val())
{


   ELISE_COPY
   (
       mIm.all_pts(),
       Max(0,Min(mDynG-1,(aF[Virgule(FX/aZ,FY/aZ)]*double(mDynG))/255)),
       mIm.out() 
   );
}
 

void cVisuCaracDigeo::Save(const std::string& aName)
{
  std::vector<Elise_colour> aVCol;

  for (int aK=0 ; aK<mDynG ; aK++)
  {
      double aGr = aK/double(mDynG-1);
      aVCol.push_back(Elise_colour::rgb(aGr,aGr,aGr));
  }

 aVCol.push_back(Elise_colour::rgb(0,1,0)); // eTES_instable
 aVCol.push_back(Elise_colour::rgb(0.7,0.5,0)); // eTES_TropAllonge
 aVCol.push_back(Elise_colour::rgb(1,1,0));  // eTES_GradFaible


  int aK0 = mDynG + eTES_Ok ;
  for (int aK=aK0  ; aK<256 ; aK++)
  {
      aVCol.push_back(((aK-aK0)%2) ? Elise_colour::red : Elise_colour::blue );
  }


  Disc_Pal aPal(&(aVCol[0]),aVCol.size());

  std::string aDir =  mAppli.DC() + mParam.Dir()  ;
  ELISE_fp::MkDirSvp(aDir);
  std::string aFullName = aDir+mParam.Prefix().Val()+ aName;


  Tiff_Im   aTif(
                 aFullName.c_str(),
                 mIm.sz(),
                 GenIm::u_int1,
                 Tiff_Im::No_Compr,
                 aPal
            );

  ELISE_COPY(mIm.all_pts(),mIm.in(),aTif.out()); 
}


void cVisuCaracDigeo::SetPtsCarac
     (
         const Pt2dr & aP,
         bool aMax,
         double aSigma,
         int  aIndSigma,
         eTypeExtreSift aType
     )
{
    int anIndCoul =  mDynG;

    if (aType==eTES_Ok)
    {
         anIndCoul += (int)eTES_Ok+ 2*aIndSigma + (aMax?1:0);
    }
    else
    {
        if (!mParam.ShowCaracEchec().Val())
           return;
        anIndCoul += (int) aType;
    }


    for (int anX= 0 ; anX<mZ ; anX++)
    {
       for (int anY=0 ; anY<mZ ; anY++)
       {
            if ((anX==(mZ/2)) && (anY==(mZ/2)))
            {
                mTIm.oset_svp(Pt2di(anX,anY)+aP,anIndCoul);
            }
       }
    }
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
