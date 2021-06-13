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


/*********************************************************************/
/*                                                                   */
/*    Flux_Concat_Comp                                               */
/*                                                                   */
/*********************************************************************/

class  Flux_Concat_Comp  : public Flux_Pts_Computed
{
       public :
          
           // suppose they have already been reduced to the same type

           Flux_Concat_Comp(Flux_Pts_Computed *,Flux_Pts_Computed *);
           virtual ~Flux_Concat_Comp()
           {
               delete _flx[1];
               delete _flx[0];
           }

       private :

           const Pack_Of_Pts * next()
           {
               for (;;)
               {
                   const Pack_Of_Pts * pack = _flx[_ifl]->next();
                   if (pack)
                      return pack;
                   _ifl++;
                   if (_ifl == 2)
                      return 0;
               }
           }
 
           Flux_Pts_Computed * _flx[2];
           INT                 _ifl;
};


Flux_Concat_Comp::Flux_Concat_Comp
(
       Flux_Pts_Computed * flx1,
       Flux_Pts_Computed * flx2
)  :
   Flux_Pts_Computed
   (
          flx1->dim(),
          flx1->type(),
          ElMax(flx1->sz_buf(),flx2->sz_buf())
    )
{
    ASSERT_INTERNAL
    (
          flx1->type()==flx2->type(),
          "Incorrect type reduction in Flux_Concat_Comp"
    );
    ASSERT_TJS_USER
    (
          flx1->dim()==flx2->dim(),
          "Incompatible dimension in concatenation of flux"
    );

    _flx[0] = flx1;
    _flx[1] = flx2;
    _ifl = 0;
}

/*********************************************************************/
/*                                                                   */
/*    Flux_Concat_Not_Comp                                           */
/*                                                                   */
/*********************************************************************/

class Flux_Concat_Not_Comp : public Flux_Pts_Not_Comp
{

      public :

          Flux_Concat_Not_Comp(Flux_Pts,Flux_Pts);

      private :

         virtual Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg)
         {
                Flux_Pts_Computed * flc1 = _flx1.compute(arg); 
                Flux_Pts_Computed * flc2 = _flx2.compute(arg); 

                Flux_Pts_Computed::type_common(&flc1,&flc2);
                return new Flux_Concat_Comp(flc1,flc2);
         }

         Flux_Pts _flx1;
         Flux_Pts _flx2;
};


Flux_Concat_Not_Comp::Flux_Concat_Not_Comp
(
      Flux_Pts flx1,
      Flux_Pts flx2
)  :
   _flx1 (flx1),
   _flx2 (flx2)
{
}


Flux_Pts Flux_Pts::operator || (Flux_Pts flx)
{
      return new  Flux_Concat_Not_Comp (*this,flx);
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
