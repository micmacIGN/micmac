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


class Opb_Dilate_Label : public Simple_OPBuf1<INT,INT>
{
    public : 

       Opb_Dilate_Label
       (
             const Chamfer & chamf,
             INT             dmax
       );


       ~Opb_Dilate_Label();

       virtual Simple_OPBuf1<INT,INT> * dup_comp();

       Box2di box() {return _box;}

   private :

       virtual void  calc_buf
                     (
                           INT ** output,
                           INT *** input
                     );


          void init_line(INT y);
          
          void prop
               (
                    INT yloc,
                    const Pt2di * v,
                    const INT   * pds,
                    INT           nbv,
                    INT           x0,
                    INT           x1,
                    INT           dx
               );

          void prop_av(INT y);
          void prop_ar(INT y);

          const Chamfer & _chamf;
          INT  _dmax;
          INT  _y_cpt;

          Box2di _box;


          INT  _nb;
          INT  _delta;
          INT  _per_reaff;


          INT     ***   _input;
          U_INT1  ***   _dist;
};

Opb_Dilate_Label::~Opb_Dilate_Label()
{
    if (_dist)
    {
        DELETE_TAB_MATRICE
        (
            _dist,
            dim_out(),
            Pt2di(x0Buf(),y0Buf()),
            Pt2di(x1Buf(),y1Buf())
        );
    }
}

Simple_OPBuf1<INT,INT> * Opb_Dilate_Label::dup_comp()
{
    Opb_Dilate_Label * odl = new Opb_Dilate_Label(_chamf,_dmax);

    odl->_dist = NEW_TAB_MATRICE
                 (
                      dim_out(),
                      Pt2di(x0Buf(),y0Buf()),
                      Pt2di(x1Buf(),y1Buf()),
                      U_INT1
                 );

    return odl;
}


Opb_Dilate_Label::Opb_Dilate_Label
(
      const Chamfer & chamf,
      INT             dmax
) :

    _chamf (chamf),
    _dmax  (dmax),
    _y_cpt (0),
    _box   (Pt2di(0,0),Pt2di(0,0)),
    _dist  (0)
{
   _delta     = (dmax + _chamf.p_0_1()-1)/_chamf.p_0_1();
   _per_reaff = (INT) (1.5 * _delta) + 5;
   _nb        = _per_reaff+_delta;

   INT r     = chamf.radius();
   _box      = Box2di
               ( 
                   Pt2di(-(r+_delta),-(r+_delta)),
                   Pt2di(r+_delta,_per_reaff+_delta+r-1)
               );

}


void Opb_Dilate_Label::prop
     (
          INT y,
          const Pt2di * v,
          const INT   * pds,
          INT           nbv,
          INT           x0,
          INT           x1,
          INT           dx
     )
{
     for (INT d = 0; d< dim_out(); d++)
     {
         INT ** labs = _input[d]+ y;
         INT * l0   =  labs[0];
         U_INT1 ** dist = _dist[d]+ y;
         U_INT1 *  d0   =  dist[0];
         for (INT x= x0 ; x != x1 ; x+= dx)
             if (d0[x])
             {
                 for (INT k= 0; k<nbv ; k++)
                 {
                     INT new_d = dist[v[k].y][x+v[k].x] + pds[k];
                     if (new_d < d0[x])
                     {
                         d0[x] = new_d;
                         l0[x] = labs[v[k].y][x+v[k].x];
                     }
                 }
             }
     }
}

void Opb_Dilate_Label::prop_av(INT y)
{
    prop
    (
         y,
         _chamf.neigh_yn(),
         _chamf.pds_yn(),
         _chamf.nbv_yn(),
         x0()-_delta,
         x1()+_delta,
         1
    );
}

void Opb_Dilate_Label::prop_ar(INT y)
{
     prop
     (
         y,
         _chamf.neigh_yp(),
         _chamf.pds_yp(),
         _chamf.nbv_yp(),
         x1()+_delta-1,
         x0()-_delta-1,
         -1
     );
}



void Opb_Dilate_Label::init_line(INT y)
{
     for (INT d = 0; d< dim_out(); d++)
     {
          INT * label   = _input[d][y];
          U_INT1 * dist = _dist[d][y];
          for (INT x = x0Buf(); x<x1Buf() ; x++)
              dist[x] = label[x] ? 0 : _dmax;
     }
}

void  Opb_Dilate_Label::calc_buf (INT ** output,INT *** input)
{
     _input = input;

     if (first_line())
     {
         for (INT y=y0Buf(); y<y1Buf()-1 ; y++)
             init_line(y);
     }
     init_line(y1Buf()-1);


     if (first_line())
     {
         for (INT y = -_delta ; y < _nb; y++)
             prop_av(y);
     }
     prop_av(_nb);

     if (! _y_cpt)
         for (INT y = _nb-1 ; y >= 0; y--)
             prop_ar(y);


     {
         for (int d=0 ; d<dim_out() ; d++)
             convert(output[d]+x0(),input[d][0]+x0(),tx());
     }

     _y_cpt = (_y_cpt +1) %_per_reaff;

     for (INT d = 0; d<dim_out() ; d++)
         rotate_plus_data(_dist[d],y0Buf(),y1Buf());
}

Fonc_Num dilate_label
         (
            Fonc_Num          label,
            const Chamfer &   chamf,
            INT               dmax
          )
{
     Opb_Dilate_Label * odl = new Opb_Dilate_Label(chamf,dmax);
      
     return create_op_buf_simple_tpl
            (
                odl,
                0,
                label,
                label.dimf_out(),
                odl->box()
            );
}


/*********************************************************************/
/*                                                                   */
/*         Opb_MajVois_Compl                                         */
/*                                                                   */
/*********************************************************************/

class Opb_MajVois_Compl : public Simple_OPBuf1<INT,INT>
{
  public :

       Opb_MajVois_Compl(INT vmax);
       virtual ~Opb_MajVois_Compl();

  private  :

       friend class cell;

       virtual void  calc_buf (INT ** output, INT *** input);
       void add_line(INT *** fonc,INT y,bool add);
       void cumul (INT x, bool add);


       virtual Simple_OPBuf1<INT,INT> * dup_comp();


       INT * _lut;


       INT * _nb;
       INT * _h;
       INT * _lut_inv;

       INT   _nb_luted;
       INT   _vmax;

       INT ** _etiq;
       INT ** _pds;

       enum {unused = -0x7fffffff};

       void  add_v(INT x,INT y)
       {
             INT e = _etiq[y][x];

             if(_lut[e] == unused)
             {
               _lut_inv[_nb_luted] = e;
               _lut[e] = _nb_luted++;
             }
             _nb[_lut[e]]++;
             _h[_lut[e]]+= _pds[y][x];
       }

       void  supp_v(INT x,INT y)
       {
             INT e = _etiq[y][x];
             INT l = _lut[e];
             _nb[l]--;
             _h[l]-= _pds[y][x];
             if (_nb[l] == 0)
             {
                _lut[e] = unused;
                _nb_luted--;
                 if (l != _nb_luted)
                 {
                     ElSwap(_nb[l],_nb[_nb_luted]);
                     ElSwap(_lut_inv[l],_lut_inv[_nb_luted]);
                     ElSwap(_h[l],_h[_nb_luted]);
                     _lut[_lut_inv[l]] = l;
                 }
             }
       }
};

Opb_MajVois_Compl::Opb_MajVois_Compl(INT vmax) :
     _lut       (0),
     _nb_luted  (0),
     _vmax      (vmax)
{
}



Simple_OPBuf1<INT,INT> * Opb_MajVois_Compl::dup_comp()
{
     Opb_MajVois_Compl * omvc = new Opb_MajVois_Compl(_vmax);

     omvc->_lut = new_vecteur_init(0,_vmax,(INT)unused) ;

     INT nb_lut = ElMin(_vmax,(dx1()-dx0()+1)*(dy1()-dy0()+1));
     omvc->_nb =  new_vecteur_init(0,nb_lut,0);
     omvc->_h  =  new_vecteur_init(0,nb_lut,0);
     omvc->_lut_inv =  new_vecteur_init(0,nb_lut,0);

     return omvc;
}

Opb_MajVois_Compl::~Opb_MajVois_Compl()
{
    if (_lut)
    {
        DELETE_VECTOR(_lut,0);
        DELETE_VECTOR(_nb,0);
        DELETE_VECTOR(_h,0);
        DELETE_VECTOR(_lut_inv,0);
    }
}



void Opb_MajVois_Compl::calc_buf(INT ** output,INT *** input)
{
	INT dy, dx;

      _etiq = input[0];
      _pds  = input[1];

     for 
     (
         dy = (first_line() ? dy0() : dy1()) ; 
         dy <=dy1()                       ; 
         dy++
     )
     {
         El_User_Dyn.ElAssert
         (
              values_in_range(_etiq[dy]+x0Buf(),x1Buf()-x0Buf(),0,_vmax),
              EEM0 << "Value out of predicted range in etiq_maj" 
         );
     }



     for ( dx = dx0(); dx < dx1() ; dx++)
         for ( dy = dy0(); dy <= dy1() ; dy++)
               add_v(x0()+dx,dy);

     for (INT x = x0(); x <x1() ; x++)
     {
         for ( dy = dy0(); dy <= dy1() ; dy++)
               add_v(dx1()+x,dy);


         output[0][x] = _lut_inv[index_vmax(_h,_nb_luted)];

         for ( dy=dy0(); dy<= dy1() ; dy++)
               supp_v(dx0()+x,dy);

     }



     for ( dx = dx0()+1; dx <= dx1() ; dx++)
         for ( dy = dy0(); dy <= dy1() ; dy++)
               supp_v(x1()-1+dx,dy);

}





/*********************************************************************/
/*                                                                   */
/*         Opb_MajVois_Simpl                                         */
/*                                                                   */
/*********************************************************************/


class Opb_MajVois_Simpl : public Simple_OPBuf1<INT,INT>
{
  public :

       Opb_MajVois_Simpl(INT vmax);
       virtual ~Opb_MajVois_Simpl();

  private  :

       virtual void  calc_buf
                     (
                           INT ** output,
                           INT *** input
                     );
       void add_line(INT *** fonc,INT y,bool add);
       void cumul (INT x, bool add);


       virtual Simple_OPBuf1<INT,INT> * dup_comp();


       INT ** _hl;
       INT *  _hcum;
       INT   _vmax;
};


Opb_MajVois_Simpl::Opb_MajVois_Simpl(INT vmax) :
     _hl   (0),
     _hcum (0),
     _vmax (vmax)
{
}

Simple_OPBuf1<INT,INT> * Opb_MajVois_Simpl::dup_comp()
{
    Opb_MajVois_Simpl * omvs = new Opb_MajVois_Simpl(_vmax);
    omvs->_hl = NEW_MATRICE(Pt2di(x0Buf(),0),Pt2di(x1Buf(),_vmax),INT);
    omvs->_hcum = NEW_VECTEUR(0,_vmax,INT);

    for (INT v=0; v<_vmax ; v++)
        set_cste(omvs->_hl[v]+x0Buf(),0,x1Buf()-x0Buf());
    return omvs;
}


Opb_MajVois_Simpl::~Opb_MajVois_Simpl()
{
    if (_hcum)
    {
        DELETE_VECTOR(_hcum,0);
        DELETE_MATRICE(_hl,Pt2di(x0Buf(),0),Pt2di(x1Buf(),_vmax));
    }
}
    

void Opb_MajVois_Simpl::add_line
     (
           INT *** input,
           INT y,
           bool add
     )
{
     INT * etiq = input[0][y];
     INT * pds  = input[1][y];

     El_User_Dyn.ElAssert
     (
          (! add) || values_in_range(etiq+x0Buf(),x1Buf()-x0Buf(),0,_vmax),
          EEM0 << "Value out of predicted range in etiq_maj" 
     );

     INT sign = add ? 1 : -1;

     for (INT x=x0Buf(); x<x1Buf() ; x++)
          _hl[etiq[x]][x] += sign * pds[x];
}

void Opb_MajVois_Simpl::cumul (INT x, bool add)
{
     INT sign = add ? 1 : -1;
     for (INT v=0; v<_vmax ; v++)
          _hcum[v] += sign * _hl[v][x];
}


void Opb_MajVois_Simpl::calc_buf
     (
             INT ** output,
             INT *** input
     )
{
     if (first_line())
         for (INT dy = dy0(); dy <dy1() ; dy++)
             add_line(input,dy,true);

     add_line(input,dy1(),true);

     for (INT v = 0; v<_vmax ; v++)
         _hcum[v] = 0;

     for (INT dx = dx0(); dx < dx1() ; dx++)
         cumul(x0()+dx,true);

     for (INT x = x0(); x <x1() ; x++)
     {
         cumul(x+dx1(),true);
         output[0][x] = index_vmax(_hcum,_vmax);
         cumul(x+dx0(),false);
     }

     add_line(input,dy0(),false);
}


Fonc_Num label_maj
         (
            Fonc_Num     label,
            INT          vmax,
            Box2di       side,
            Fonc_Num     pds,
            bool         Complexe
          )
{
      Tjs_El_User.ElAssert
      (
          (label.dimf_out() == 1) && (pds.dimf_out() == 1),
          EEM0 << "Need 1-D Functions for label_maj"
      );

      
      Simple_OPBuf1<INT,INT> * op =0;

       if (Complexe)
          op = new Opb_MajVois_Compl(vmax);
       else
          op = new Opb_MajVois_Simpl(vmax);

      return create_op_buf_simple_tpl
             (
                    op,
                    0,
                    Virgule(label,pds),
                    1,
                    side
             );
}

Fonc_Num label_maj
         (
            Fonc_Num     label,
            INT          vmax,
            Box2di       side,
            Fonc_Num     pds
          )
{
    return label_maj
           (
               label,
               vmax,
               side,
               pds,
               (vmax > 3* (side._p1.y-side._p0.y+1))
           );
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
