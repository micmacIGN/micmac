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



/***********************************************************************/
/*                                                                     */
/*       ImOutRLE_Comp <TypeIm>                                        */
/*                                                                     */
/***********************************************************************/

template <class TypeIm>  
          void ImOutRLE_Comp<TypeIm>::update
               (const Pack_Of_Pts * pts,const Pack_Of_Pts * v)
{
    if (int nb = pts->nb())
    {
        const Std_Pack_Of_Pts<TypeIm> * vals = 
                  SAFE_DYNC(const Std_Pack_Of_Pts<TypeIm> *,v);

        const RLE_Pack_Of_Pts * rle_pts =
                  SAFE_DYNC(const RLE_Pack_Of_Pts *,pts);


        if (! rle_pts->inside(_gi->p0(),_gi->p1()))
           El_User_Dyn.ElAssert 
           (
              false,
              EEM0 
                << "   BITMAP :  out of domain while writing (RLE mode)\n"
                << "  pts : " 
                << ElEM(rle_pts, rle_pts->ind_outside(_gi->p0(),_gi->p1()))
                << ", Bitmap limits : " 
                << ElEM(_gi->p1(),rle_pts->dim())
                << "\n"
           );

       if (El_User_Dyn.active())
          _gi->verif_in_range_type(vals->_pts[0],pts);

       _gi->out_rle
       (
            _gi->calc_adr_seg(rle_pts->pt0()),
            nb,
            vals->_pts[0],
            rle_pts->vx0()
       );
    }
}

template <class TypeIm>  
          ImOutRLE_Comp<TypeIm>::ImOutRLE_Comp(DataGenIm *gi) :
                 GenImOutRLE_Comp(gi) 
{
}

template <class TypeIm>  
         ImOutRLE_Comp<TypeIm>::~ImOutRLE_Comp()
{}

      //----------------------------------------------
      //----------------------------------------------



/***********************************************************************/
/*                                                                     */
/*       ImInRLE_Comp <TypeBase>                                       */
/*                                                                     */
/***********************************************************************/


template <class TypeBase> ImInRLE_Comp<TypeBase>::ImInRLE_Comp
         (  
            const Arg_Fonc_Num_Comp & arg,
            Flux_Pts_Computed * flux,
            DataGenIm *gi,
            bool      with_def_value
         ) :

         Fonc_Num_Comp_TPL<TypeBase>  (arg,1,flux),
         _gi                          (gi),
         _with_def_value              (with_def_value)
{
}



template <class TypeBase> const Pack_Of_Pts * 
      ImInRLE_Comp<TypeBase>::values(const Pack_Of_Pts * pts)
{
   INT nb_tot = pts->nb();
   RLE_Pack_Of_Pts * rle_pts = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pts));

 
   if(! _with_def_value)
   {
       if (!rle_pts->inside(_gi->p0(),_gi->p1()))
           El_User_Dyn.ElAssert 
           (
               false,
               EEM0 
                 << "   BITMAP :  out of domain while reading (RLE mode)\n"
                 << "  pts : " 
                 << ElEM(rle_pts, rle_pts->ind_outside(_gi->p0(),_gi->p1()))
                 << ", Bitmap limits : " 
                 << ElEM(_gi->p1(),rle_pts->dim())
                 << "\n"
           );
    }

   if (nb_tot)
      _gi->void_input_rle
      (
             this->_pack_out->_pts[0],
             nb_tot,
             _gi->calc_adr_seg(rle_pts->pt0()),rle_pts->vx0()
      );
   this->_pack_out->set_nb(nb_tot);
   return this->_pack_out;
}

/***********************************************************************/
/*                                                                     */
/*        ImOutInteger <TypeBase>                                      */
/*                                                                     */
/***********************************************************************/
template <class TypeBase> ImOutInteger<TypeBase>::~ImOutInteger(){};

template <class TypeBase> ImOutInteger<TypeBase>::ImOutInteger
         (const Arg_Output_Comp &,DataGenIm *gi, bool auto_clip) :
                Output_Computed(1)
{
     _gi = gi;
     _auto_clip = auto_clip;
}


template <class TypeBase> void ImOutInteger<TypeBase>::update
       (const Pack_Of_Pts * pts,const Pack_Of_Pts * vals)
{

     const Std_Pack_Of_Pts<TypeBase> * pvals = 
           SAFE_DYNC(const Std_Pack_Of_Pts<TypeBase> *,vals);
     const Std_Pack_Of_Pts<INT> *      ppts 
            =  SAFE_DYNC(const  Std_Pack_Of_Pts<INT> *,pts);

     if ((!_auto_clip) && El_User_Dyn.active())
        ppts->verif_inside(_gi->p0(),_gi->p1(),0,0);

     if (El_User_Dyn.active())
        _gi->verif_in_range_type(pvals->_pts[0],pts);


    _gi->out_pts_integer
    (   ppts->_pts,
        ppts->nb(),
        pvals->_pts[0]
    );
} 

/***********************************************************************/
/*                                                                     */
/*        ImInInteger <TypeBase>                                       */
/*                                                                     */
/***********************************************************************/

template <class TypeBase>  ImInInteger<TypeBase>::ImInInteger
                           (
                              const Arg_Fonc_Num_Comp & arg,
                              DataGenIm *gi,
                              bool              with_def_value 
                           ) :

                  Fonc_Num_Comp_TPL<TypeBase> (arg,1,arg.flux()),
                 _gi                          (gi),
                 _with_def_value              (with_def_value)
{
}



template <class TypeBase> const Pack_Of_Pts *  
                 ImInInteger<TypeBase>::values(const Pack_Of_Pts * gen_pts)
{
    const Std_Pack_Of_Pts<INT> * pts = 
           SAFE_DYNC(const Std_Pack_Of_Pts<INT> *,gen_pts);

    INT nb_tot = pts->nb();
    INT ** coord = pts->_pts;

    if ((! _with_def_value) && El_User_Dyn.active())
        pts->verif_inside(_gi->p0(),_gi->p1(),0,0);
    _gi->input_pts_integer( this->_pack_out->_pts[0],coord,nb_tot);
    this->_pack_out->set_nb(nb_tot);
    return this->_pack_out;

}

/***********************************************************************/
/*                                                                     */
/*        ImInReal                                                     */
/*                                                                     */
/***********************************************************************/


ImInReal::ImInReal (
                              const Arg_Fonc_Num_Comp & arg,
                              DataGenIm *gi,
                              bool              with_def_value 
                    ) :

                  Fonc_Num_Comp_TPL<REAL> (arg,1,arg.flux()),
                 _gi                      (gi),
                 _with_def_value          (with_def_value)
{
}

const Pack_Of_Pts *  ImInReal::values(const Pack_Of_Pts * gen_pts)
{
    const Std_Pack_Of_Pts<REAL> * pts = 
           SAFE_DYNC(const Std_Pack_Of_Pts<REAL> *,gen_pts);

    INT nb_tot = pts->nb();
    REAL  ** coord = pts->_pts;

    if ((! _with_def_value) && El_User_Dyn.active())
        pts->verif_inside(_gi->p0(),_gi->p1(),0.0,1.0);
    _gi->input_pts_reel(_pack_out->_pts[0],coord,nb_tot);
    _pack_out->set_nb(nb_tot);
    return _pack_out;

}


/***********************************************************************/
/***********************************************************************/

template class ImInInteger<INT>;
template class ImInInteger<REAL>;
template class ImOutInteger<INT>;
template class ImOutInteger<REAL>;
template class ImOutRLE_Comp<INT>;
template class ImOutRLE_Comp<REAL>;
template class ImInRLE_Comp<INT>;
template class ImInRLE_Comp<REAL>;


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
