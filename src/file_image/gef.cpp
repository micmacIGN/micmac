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



/**************************************************************/
/**************************************************************/
/*****                                                     ****/
/*****         ElDataGenFileIm                             ****/
/*****                                                     ****/
/**************************************************************/
/**************************************************************/


ElDataGenFileIm::ElDataGenFileIm(){}

ElDataGenFileIm::~ElDataGenFileIm()
{
     DELETE_VECTOR(_sz     ,0);
     DELETE_VECTOR(_sz_tile,0);
}


void ElDataGenFileIm::init
(
    int              dim,
    const int *      sz,
    INT              nb_channel,
    bool             signedtype,
    bool             integral,
    int              nbbits,
    const int *      sz_tile,
    bool             compressed
)
{
    _dim         = dim;
    _sz          = dup(sz,dim);
    _nb_channel  = nb_channel;
    _signedtype  = signedtype;
    _integral    = integral;
    _nbbits      = nbbits;
    _sz_tile     = dup(sz_tile,dim);
    _compressed  = compressed;
}

/**************************************************************/
/**************************************************************/
/*****                                                     ****/
/*****         ElGenFileIm                                 ****/
/*****                                                     ****/
/**************************************************************/
/**************************************************************/


ElDataGenFileIm * ElGenFileIm::edgfi()
{
   return SAFE_DYNC(ElDataGenFileIm *,_ptr);
}

const ElDataGenFileIm * ElGenFileIm::edgfi() const
{
   return SAFE_DYNC(const ElDataGenFileIm *,_ptr);
}

INT          ElGenFileIm::Dim()             const    {return edgfi()->_dim;}
const INT *  ElGenFileIm::Sz()              const    {return edgfi()->_sz;}
INT          ElGenFileIm::NbChannel()       const    {return edgfi()->_nb_channel;}
bool         ElGenFileIm::SigneType()       const    {return edgfi()->_signedtype;}
bool         ElGenFileIm::IntegralType()    const    {return edgfi()->_integral;}
int          ElGenFileIm::NbBits()          const    {return edgfi()->_nbbits;}
const int *  ElGenFileIm::SzTile()          const    {return edgfi()->_sz_tile;}
bool         ElGenFileIm::Compressed()      const    {return edgfi()->_compressed;}


GenIm::type_el ElGenFileIm::type_el()
{
	return type_im(IntegralType(),NbBits(),SigneType(),true);
}


ElGenFileIm::ElGenFileIm(ElDataGenFileIm * EDGFI) :
  PRC0(EDGFI)
{
}



Elise_Rect ElGenFileIm::box() const
{
   return Elise_Rect(PTS_00000000000000,Sz(),Dim());
}

ElGenFileIm::~ElGenFileIm() {}

Fonc_Num ElGenFileIm::in()
{
    return edgfi()->in();
}
Fonc_Num ElGenFileIm::in(REAL val)
{
    return edgfi()->in(val);
}
Output   ElGenFileIm::out()
{
    return edgfi()->out();
}

Pt2di ElGenFileIm::Sz2() const
{
	ELISE_ASSERT(Dim()==2,"ElGenFileIm::Sz2");
	return Pt2di(Sz()[0],Sz()[1]);
}

template <class Type>  Im2D<Type,typename El_CTypeTraits<Type>::tBase> LoadFileIm(ElGenFileIm aFile,Type *)
{
   Pt2di aSz = aFile.Sz2();
   Im2D<Type,typename El_CTypeTraits<Type>::tBase>  aRes(aSz.x,aSz.y);

   ELISE_COPY(aRes.all_pts(),aFile.in(),aRes.out());

   return aRes;
}

template Im2D<U_INT1,INT> LoadFileIm(ElGenFileIm aFile,U_INT1 *);






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
