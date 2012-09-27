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


/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
/*******         Color_Pallete                                             *****/
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*****************************************************************/
/*                                                               */
/*                  Data_Col_Pal                                 */
/*                                                               */
/*****************************************************************/


INT Data_Col_Pal::_number_tot = 0;



Data_Col_Pal::Data_Col_Pal(Elise_Palette pal,INT c0,INT c1,INT c2) :
      _pal (pal)
{
   _c[0] = c0;
   _c[1] = c1;
   _c[2] = c2;
   _num =  Data_Col_Pal::_number_tot++;
}

Data_Col_Pal::Data_Col_Pal() :
      _pal (Elise_Palette(0))
{
    _c[0] = _c[1] = _c[2] = -1;
    _num = -1;
}



INT Data_Col_Pal::get_index_col(Data_Disp_Set_Of_Pal * ddsop)
{
   Data_Elise_Palette * dep = _pal.dep();
   Data_Disp_Pallete * ddp = ddsop->ddp_of_dep(dep,true);
   ASSERT_TJS_USER
   (
        ddp !=0,
        "Use of palette not loaded in display"
   );

   return dep->ilutage(ddsop->derd(),ddp->lut_compr(),_c);
}



/*****************************************************************/
/*                                                               */
/*                  Col_Pal                                      */
/*                                                               */
/*****************************************************************/


Col_Pal::Col_Pal(Elise_Palette pal,INT c0)  :
    PRC0 (new Data_Col_Pal(pal,c0))
{
}


Col_Pal::Col_Pal(Elise_Palette pal,INT c0,INT c1)  :
    PRC0 (new Data_Col_Pal(pal,c0,c1))
{
}

Col_Pal::Col_Pal(Elise_Palette pal,INT c0,INT c1,INT c2)  :
    PRC0 (new Data_Col_Pal(pal,c0,c1,c2))
{
}



/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
/*******         Line Styles                                               *****/
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*****************************************************************/
/*                                                               */
/*                  Data_Line_St                                 */
/*                                                               */
/*****************************************************************/


Data_Line_St::Data_Line_St(Col_Pal COL,REAL WIDTH) :
    _col   (COL),
    _width (WIDTH)
{
    ASSERT_TJS_USER(WIDTH>0,"witdh line <= 0");
}

/*****************************************************************/
/*                                                               */
/*                       Line_St                                 */
/*                                                               */
/*****************************************************************/

Line_St::Line_St(Col_Pal pal) :
    PRC0(new Data_Line_St(pal,1))
{
}

Line_St::Line_St(Col_Pal pal, REAL width) :
    PRC0(new Data_Line_St(pal,width))
{
}
Col_Pal Line_St::col() const { return dlst()->col();}



/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
/*******          Fill_Styles                                              *****/
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/



/*****************************************************************/
/*                                                               */
/*                  Data_Fill_St                                 */
/*                                                               */
/*****************************************************************/


Data_Fill_St::Data_Fill_St(Col_Pal COL) :
    _col   (COL)
{
}

/*****************************************************************/
/*                                                               */
/*                       Line_St                                 */
/*                                                               */
/*****************************************************************/

Fill_St::Fill_St(Col_Pal pal) :
    PRC0 (new Data_Fill_St(pal))
{
}

Col_Pal Fill_St::col() const { return dfst()->col();}






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
