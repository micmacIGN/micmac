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
/*       ImRedAssOutInteger                                            */
/*                                                                     */
/***********************************************************************/


template <class TyBase> class ImRedAssOutInteger : public Output_Computed
{
    public :
        ImRedAssOutInteger
        (
              const Arg_Output_Comp &,
              const OperAssocMixte & op,
              TyBase    * pre_out,  // eventually 0
              DataGenIm *gi,
              bool auto_clip
        );

        virtual ~ImRedAssOutInteger()
        {
               if (_free_pre_out)
                  DELETE_VECTOR(_pre_out,0);
        }
   private :
        virtual void update
                     (
                         const Pack_Of_Pts * pts,
                         const Pack_Of_Pts * vals
                     );

        const OperAssocMixte &        _op;
        DataGenIm                      *   _gi;
        bool                          _auto_clip;

        TyBase  *                     _pre_out;
        bool                          _free_pre_out; // true if _pre_out
                                      // allocated by this class
};


template <class TyBase> ImRedAssOutInteger<TyBase>::ImRedAssOutInteger
(
    const Arg_Output_Comp & arg,
    const OperAssocMixte &  op,
    TyBase  *               pre_out,
    DataGenIm *             gi,
    bool                    auto_clip
)   :
     Output_Computed   (1),
     _op               (op),
     _gi               (gi),
     _auto_clip        (auto_clip)
{
    if (pre_out)
    {
       _free_pre_out = false;
       _pre_out = pre_out;
    }
    else
    {
       _free_pre_out = true;
       _pre_out = NEW_VECTEUR(0,arg.flux()->sz_buf(),TyBase);
    }
}



template <class TyBase> 
         void ImRedAssOutInteger<TyBase>::update
         (
            const Pack_Of_Pts * gen_pts,
            const Pack_Of_Pts * gen_vals
         )
{
     const Std_Pack_Of_Pts<INT> *      pts
            =  SAFE_DYNC(const  Std_Pack_Of_Pts<INT> *,gen_pts);

     const Std_Pack_Of_Pts<TyBase> *      vals
            =  SAFE_DYNC(const  Std_Pack_Of_Pts<TyBase> *,gen_vals);

     if ((!_auto_clip) && El_User_Dyn.active())
        pts->verif_inside(_gi->p0(),_gi->p1(),0,0);

     _gi->out_assoc
     (
         _pre_out,
         _op,
         pts->_pts,
         pts->nb(),
         vals->_pts[0]
     );

}


/***********************************************************************/
/*                                                                     */
/*       ImRedAssOutRle                                                */
/*                                                                     */
/***********************************************************************/

template <class TyBase> class ImRedAssOutRle : public Output_Computed
{
    public :
        ImRedAssOutRle
        (
              const Arg_Output_Comp &,
              const OperAssocMixte & op,
              TyBase    * pre_out,  // eventually 0
              DataGenIm *gi,
              bool auto_clip
        );

        virtual ~ImRedAssOutRle()
        {
               if (_free_pre_out)
                  DELETE_VECTOR(_pre_out,0);
               DELETE_VECTOR(_out,0);
        }
   private :
        virtual void update
                     (
                         const Pack_Of_Pts * pts,
                         const Pack_Of_Pts * vals
                     );

        const OperAssocMixte &        _op;
        DataGenIm                      *   _gi;
        bool                          _auto_clip;

        TyBase  *                     _pre_out;
        TyBase  *                     _out;
        bool                          _free_pre_out; // true if _pre_out
                                      // allocated by this class
};


template <class TyBase> ImRedAssOutRle<TyBase>::ImRedAssOutRle
(
    const Arg_Output_Comp & arg,
    const OperAssocMixte &  op,
    TyBase  *               pre_out,
    DataGenIm *             gi,
    bool                    auto_clip
)   :
     Output_Computed   (1),
     _op               (op),
     _gi               (gi),
     _auto_clip        (auto_clip)
{
    if (pre_out)
    {
       _free_pre_out = false;
       _pre_out = pre_out;
    }
    else
    {
       _free_pre_out = true;
       _pre_out = NEW_VECTEUR(0,arg.flux()->sz_buf(),TyBase);
    }
    _out = NEW_VECTEUR(0,arg.flux()->sz_buf(),TyBase);
}

template <class TyBase> 
         void ImRedAssOutRle<TyBase>::update
         (
            const Pack_Of_Pts * gen_pts,
            const Pack_Of_Pts * gen_vals
         )
{
    if (int nb = gen_pts->nb())
    {
        const Std_Pack_Of_Pts<TyBase> * vals =
                  SAFE_DYNC(const Std_Pack_Of_Pts<TyBase> *,gen_vals);

        const RLE_Pack_Of_Pts * rle_pts =
                  SAFE_DYNC(const RLE_Pack_Of_Pts *,gen_pts);

        if (!_auto_clip)
        {
           ASSERT_USER
           (
                rle_pts->inside(_gi->p0(),_gi->p1()),
               "outside  in writing in RLE mode, (clip non activated)"
           );
        }

        void * adr = _gi->calc_adr_seg(rle_pts->pt0());

        _gi->void_input_rle(_pre_out,nb,adr,rle_pts->vx0());
        _op.t0_eg_t1_op_t2(_out,_pre_out,vals->_pts[0],nb);

        if (El_User_Dyn.active())
           _gi->verif_in_range_type(_out,gen_pts);

        _gi->out_rle(adr,nb,_out,rle_pts->vx0());
    }
}

/***********************************************************************/
/*                                                                     */
/*       ImRedAssOutNotComp                                            */
/*                                                                     */
/***********************************************************************/

Output_Computed * ImRedAssOutNotComp::compute(const Arg_Output_Comp & arg)
{
     Tjs_El_User.ElAssert
     (
          arg.flux()->dim() == _gi->dim(),
          EEM0 << " Associative reduction : dim of flux should equal dim of bitmap\n"
               << "|    dim flux = " <<  arg.flux()->dim()
               << " , dim bitmap = " <<  _gi->dim()
     );


     Output_Computed * res = 0;
 
     if (arg.flux()->type() ==  Pack_Of_Pts::integer)
     {
          if (_gi->integral_type())
             res = new
                   
                         ImRedAssOutInteger<INT>
                         (arg,_op,(INT *)0,_gi,_auto_clip_int)
                   ;
          else
             res = new
                   
                         ImRedAssOutInteger<REAL>
                         (arg,_op,(REAL *)0,_gi,_auto_clip_int)
                   ;
     }
     else if (arg.flux()->type() ==  Pack_Of_Pts::rle)
     {
          if (_gi->integral_type())
             res = new
                   
                         ImRedAssOutRle<INT>
                         (arg,_op,(INT *)0,_gi,_auto_clip_int)
                   ;
          else
             res = new
                   
                         ImRedAssOutRle<REAL>
                         (arg,_op,(REAL *)0,_gi,_auto_clip_int)
                   ;
     }
     else
     {
          elise_fatal_error
          (
             "handle only Integer Pts in red ass",
             __FILE__,
             __LINE__
          );
     }


     res  = out_adapt_type_fonc
            (
                arg,
                res,
                 _gi->integral_type()    ?
                 Pack_Of_Pts::integer    :
                 Pack_Of_Pts::real
            );

   if (_auto_clip_int)
        res = clip_out_put(res,arg,_gi->p0(),_gi->p1());

    return res;


    return 0;
}


ImRedAssOutNotComp::ImRedAssOutNotComp
(
       const OperAssocMixte & op,
       DataGenIm * gi,
       GenIm  pgi,
       bool   auto_clip

)  :
     ImOutNotComp(gi,pgi,auto_clip,auto_clip),
     _op (op)
{
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
