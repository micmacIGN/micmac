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

/********************************************************************/
/********************************************************************/
/********************************************************************/
/*****                                                           ****/
/*****                                                           ****/
/*****           ASSOCIATIVE                                     ****/
/*****                                                           ****/
/*****                                                           ****/
/********************************************************************/
/********************************************************************/
/********************************************************************/



   /**********************************************************/
   /*                                                        */
   /*         RLE_proj_brd_Comp                              */
   /*                                                        */
   /**********************************************************/


/*
     Cette classe n'est pas prevue pour etre accessible a l'utilisateur.

     Sinon, il faudrait gerer tout un tas de choses comme :

           - le fait que l'on se protege contre une abus de Fnum_Symb

           - le fait que (aujourd'hui du moins) on ne peut pas
           l'utiliser avec des fichiers compresses (car retour arrieres ...)
*/

template <class Type> class RLE_proj_brd_Comp : 
                                 public Fonc_Num_Comp_TPL<Type>
{
    public :


      RLE_proj_brd_Comp
      (
           const Arg_Fonc_Num_Comp &,
           Fonc_Num_Computed       *f,
           INT *                     ,
           INT *                     
      );

      virtual  ~RLE_proj_brd_Comp()
      {
           delete _f;
           delete _proj_pack;
      }

    private :

      const Pack_Of_Pts * values(const Pack_Of_Pts * pts);


      Fonc_Num_Computed *   _f;
      Pack_Of_Pts     *     _proj_pack; // projected pack
      INT *                 _p0;        // rectangle
      INT *                 _p1;        // 
      bool                  _rle;
      INT                   _rab;
};

template <class Type> RLE_proj_brd_Comp<Type>::RLE_proj_brd_Comp
                      (
                           const Arg_Fonc_Num_Comp & arg,
                           Fonc_Num_Computed       *f,
                           INT *                   p0,
                           INT *                   p1  
                      )   :
           Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),arg.flux()),
           _f             (f),
           _proj_pack    (Pack_Of_Pts::new_pck
                             (
                                 arg.flux()->dim(),
                                 arg.flux()->sz_buf(),
                                 arg.flux()->type()
                             )
                          ),
           _p0            (p0),
           _p1            (p1),
           _rle           (arg.flux()->type() == Pack_Of_Pts::rle),
           _rab           ((arg.flux()->type()==Pack_Of_Pts::real) ? 1 : 0)
{
}


template <class Type> const Pack_Of_Pts * 
          RLE_proj_brd_Comp<Type>::values(const Pack_Of_Pts * gen_pts)
{
    if (! _rle)
    {
       _proj_pack->proj_brd(gen_pts,_p0,_p1,_rab);
       return _f->values(_proj_pack);
    }

    const RLE_Pack_Of_Pts * pts =  gen_pts->rle_cast();

    INT nb_in = pts->nb();
    this->_pack_out->set_nb(nb_in);
     
    if (! nb_in)
       return   this->_pack_out;

    if (pts->inside(_p0,_p1))
       return _f->values(pts);

    INT  offs = _proj_pack->proj_brd(pts,_p0,_p1,_rab);
    INT nb_prj = _proj_pack->nb();
    INT offs_end = offs+nb_prj;
    INT nb_end = nb_in -offs_end;

    const Std_Pack_Of_Pts<Type> * vals = 
        _f->values(_proj_pack)->std_cast((Type *)0);

    Type ** tv = vals->_pts;
    Type ** tvo = this->_pack_out->_pts;

    for (INT d=0 ; d<this->_dim_out ; d++)
    {
        Type * v  = tv[d];
        Type * vo = tvo[d];

        set_cste(vo,v[0],offs);
        convert(vo+offs,v,nb_prj);
        set_cste(vo+offs_end,v[nb_prj-1],nb_end);
    }

    return this->_pack_out;
}


class FN_ProlPrj_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
         FN_ProlPrj_Not_Comp(Fonc_Num,const INT *,const INT *,INT dim);

      private :

         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);

         virtual bool integral_fonc(bool integral_flux) const
         {
                return _f.integral_fonc(integral_flux);
         }
         virtual INT dimf_out() const 
         {
                 return _f.dimf_out();
         }


         Fonc_Num _f;
         INT      _p0[Elise_Std_Max_Dim];
         INT      _p1[Elise_Std_Max_Dim];

          virtual Fonc_Num deriv(INT k) const
          {
                ELISE_ASSERT(false,"No formal derivation for filters");
                return 0;
          }
          
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
          virtual void  show(ostream & os) const
          {
                os << "[prolgt]";
          }                   
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for projection");
                return 0;
          }
};

Fonc_Num_Computed * FN_ProlPrj_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
      Fonc_Num_Computed * fc = _f.compute(arg);


    if (fc->integral() )
       return new RLE_proj_brd_Comp<INT> (arg,fc,_p0,_p1);
     else
       return new RLE_proj_brd_Comp<REAL>(arg,fc,_p0,_p1);
}

FN_ProlPrj_Not_Comp::FN_ProlPrj_Not_Comp
     (
          Fonc_Num f,
          const INT * p0,
          const INT * p1,
          INT dim
     )  :
        _f (f)
{
    convert(_p0,p0,dim);
    convert(_p1,p1,dim);
}

Fonc_Num  GenIm::in_proj()
{
     DataGenIm * dgi = data_im();

     return new  FN_ProlPrj_Not_Comp
                 (in(),dgi->p0(),dgi->p1(),dgi->dim());
}

Fonc_Num  Tiff_Im::in_proj()
{
     Pt2di aSz = sz();
     int aP0[2] = {0,0};
     int aP1[2] ;
     aP1[0] = aSz.x;
     aP1[1] = aSz.y;

     return new  FN_ProlPrj_Not_Comp
                 (in(),aP0,aP1,2);
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
