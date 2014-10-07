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
/*         Image_Lut_1D_Compile                                      */
/*                                                                   */
/*********************************************************************/

Image_Lut_1D_Compile::Image_Lut_1D_Compile(Im1D<INT4,INT> I) :
   _l     (I.data()),
   _b1    (0),
   _b2    (I.tx())
{
}


/*********************************************************************/
/*                                                                   */
/*         Sel_Func_Neigh_Rel_Comp                                   */
/*                                                                   */
/*********************************************************************/

class B2d_Spec_Neigh_Comp : public  Neigh_Rel_Compute
{
  public :

      virtual void set_reflexif(bool refl){ _reflexif = refl;}
      B2d_Spec_Neigh_Comp
      (
           const Arg_Neigh_Rel_Comp &,
           DataGenIm                *,
           Data_Neighbourood        *,
           Image_Lut_1D_Compile     func_sel,
           Image_Lut_1D_Compile     func_upd

      );

  private :

      virtual const Pack_Of_Pts * neigh_in_num_dir
                           ( const Pack_Of_Pts * pack_0,
                             char ** is_neigh,
                             INT & num_dir)
      {
           _gi->q_dilate
           (
                SAFE_DYNC(Std_Pack_Of_Pts<INT> *,_pack),
                is_neigh,
                SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(pack_0)),
                _neigh->coord(),
                _neigh->nb_neigh()+_reflexif,
                _func_sel,
                _func_upd
           );
           num_dir = _neigh->nb_neigh()+_reflexif;
           return _pack;
      }
 
      DataGenIm              *   _gi;
      Data_Neighbourood        *   _neigh;
      Image_Lut_1D_Compile         _func_sel;
      Image_Lut_1D_Compile         _func_upd;
      bool                         _reflexif;

};

B2d_Spec_Neigh_Comp::B2d_Spec_Neigh_Comp
(
           const Arg_Neigh_Rel_Comp & arg,
           DataGenIm              * gi,
           Data_Neighbourood        * neigh,
           Image_Lut_1D_Compile     func_sel,
           Image_Lut_1D_Compile     func_upd

) :
   Neigh_Rel_Compute
   (   arg,
       neigh,
       Pack_Of_Pts::integer,
       arg.flux()->sz_buf()*(neigh->nb_neigh() +arg.reflexif())
   ),
  _gi (gi),
  _neigh (neigh),
   _func_sel (func_sel),
   _func_upd (func_upd),
    _reflexif (arg.reflexif())
{
}

/*********************************************************************/
/*                                                                   */
/*         B2d_Spec_Neigh_Not_Comp                                   */
/*                                                                   */
/*********************************************************************/


class B2d_Spec_Neigh_Not_Comp : public  Neigh_Rel_Not_Comp
{
   public :

       B2d_Spec_Neigh_Not_Comp
       (
            Neighbourhood,
            Im2DGen,
            Im1D_INT4,
            Im1D_INT4
           
       );


        Neigh_Rel_Compute * compute(const Arg_Neigh_Rel_Comp & arg) 
        {

           Flux_Pts_Computed * flx = arg.flux();

           ASSERT_TJS_USER(flx->dim() == 2,"Cannot handle dilate spec for dim != 2");
           ASSERT_TJS_USER
          (        flx->type() == Pack_Of_Pts::integer,
                  "Cannot only handle  dilate spec for integer Pack of Points "
           );

           return new B2d_Spec_Neigh_Comp
                      (arg,_im.data_im(),_neigh.data_n(),_isel,_iupd);
        };

   private :


       Neighbourhood  _neigh;
       Im2DGen        _im;
       Im1D_INT4      _isel;
       Im1D_INT4      _iupd;
       
};

B2d_Spec_Neigh_Not_Comp::B2d_Spec_Neigh_Not_Comp
      (
            Neighbourhood   neigh,
            Im2DGen         im,
            Im1D_INT4       isel,
            Im1D_INT4       iupd

       ):

       _neigh       (neigh),
       _im          (im),
       _isel        (isel),
       _iupd        (iupd)
{
}

Neigh_Rel Im2DGen::neigh_test_and_set(Neighbourhood neigh,Im1D<INT4,INT> isel,Im1D<INT4,INT> iupd)
{
       return new B2d_Spec_Neigh_Not_Comp(neigh,*this,isel,iupd);
}


Neigh_Rel Im2DGen::neigh_test_and_set(Neighbourhood neigh,ElList<Pt2di> l,INT v_max)
{
    Im1D<INT4,INT> isel(v_max,0);
    Im1D<INT4,INT> iupd(v_max,0);


    while (! l.empty())
    {
       Pt2di p = l.pop();
       if (
            !(
                  (p.x>=0) && (p.x < v_max)
               && (p.y>=0) && (p.y < v_max)
             )
          )
       {
          std::cout << "Sel " << p.x << " Set " << p.y << " VM " << v_max<<"\n";
          ASSERT_TJS_USER
          ( 
               false,
               "incoherence in neigh_test_and_set"
          );
       }

       isel.data()[p.x] = 1;
       iupd.data()[p.x] = p.y;
    }

    return neigh_test_and_set(neigh,isel,iupd);
}

Neigh_Rel Im2DGen::neigh_test_and_set(Neighbourhood neigh,INT csel,INT cupdate,INT v_max)
{
     return neigh_test_and_set(neigh, ElList<Pt2di>()+Pt2di(csel,cupdate),v_max);
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
