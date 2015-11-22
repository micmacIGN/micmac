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



class  PS_Pts_Not_RLE :  public Mcheck
{
       public :

         PS_Pts_Not_RLE
         (
              Data_Elise_PS_Win *,
              Elise_Palette p
         );


         void put_1L
         (
               Pt2di      p0,
               U_INT1 *   x,
               U_INT1 *   y,
               U_INT1 **  cols,
               INT        nb,
               bool       cste
         );

         void put_1L
         (
               Pt2di             p0,
               U_INT1 *          x,
               U_INT1 *          y,
               U_INT1 *          x1_rect,
               Data_Col_Pal *    dcp,
               INT               nb
          );

       private :

             void put_1L
             (
                   U_INT1 *,
                   INT nb,
                   Data_Elise_PS_Disp::defF *
             );


         
            Data_Elise_PS_Win     * _w;
            Data_Elise_PS_Disp    * _psd;
            ofstream              & _fd;
            Elise_Palette           _p;
            Data_Elise_Palette    * _dep;
};

PS_Pts_Not_RLE::PS_Pts_Not_RLE
(
     Data_Elise_PS_Win * w,
     Elise_Palette       p
)   :
      _w   (w),
      _psd (w->_psd),
      _fd  (_psd->_fd),
      _p   (p),
      _dep (p.dep())
{
}


void PS_Pts_Not_RLE::put_1L
     (
          U_INT1 *  data,
          INT nb,
          Data_Elise_PS_Disp::defF * prim
     )
{
   prim->put_prim(_psd);
   _psd->_LStr85.put_prim(_psd); 

    PS_A85 a85(_fd);
    for (INT i=0 ; i<nb ; i++)
        a85.put(data[i]);

    a85.close_block();
}


void PS_Pts_Not_RLE::put_1L
     (
          Pt2di      p0,
          U_INT1 *   x,
          U_INT1 *   y,
          U_INT1 **  cols,
          INT        nb,
          bool       cste
     )
{
    _fd << p0.x << " " << p0.y << " translate\n";
    _w->set_active();
    _psd->set_active_palette(_p,false);
    for (INT k=0; k <nb ; k+= _psd->max_str)
    {
         INT nb_loc = ElMin((INT)_psd->max_str,nb-k);

         put_1L(x+k,nb_loc,&_psd->_StrX);
         put_1L(y+k,nb_loc,&_psd->_StrY);

         if (cols)
            for (INT d=0 ; d<(cste ? 1 : _dep->ps_dim_out()) ; d++)
                put_1L(cols[d]+k,nb_loc,_psd->_StrC[d]);

         _fd << (nb_loc-1) << " " ;
         _psd->Lpts_put_prim(cste);
    }
    _fd << -p0.x << " " << -p0.y << " translate\n";
}

void PS_Pts_Not_RLE::put_1L
     (
        Pt2di             p0,
        U_INT1 *          x,
        U_INT1 *          y,
        U_INT1 *          x1_rect,
        Data_Col_Pal *    dcp,
        INT               nb
     )
{
     _w->set_active();
     _w->set_col(dcp);
     put_1L(p0,x,y,&x1_rect,nb,true);
}

/**************************************************************/
/*                                                            */
/*         PS_Out_RLE_computed                                */
/*                                                            */
/**************************************************************/

/*
void Ps_Multi_Filter::put
(
     Elise_Palette  p,
     U_INT1 ***    data,
     INT           dim_out,
     Pt2di         sz,
     Pt2di         p0,
     Pt2di         p1
)
*/

class PS_Out_RLE_computed : public GW_computed
{

   public  :

      virtual ~PS_Out_RLE_computed();

      PS_Out_RLE_computed
      (
          const Arg_Output_Comp &,
          const Data_El_Geom_GWin *,
          Data_Elise_Palette *,
          Data_Elise_PS_Win *,
          bool              rle
      );

      // when number of seg is > to this value, use image bits mode
      // for constante
      enum { nb_seg_max_cste = 2500 };

   private :

      class dalle 
      { 
           public :
               dalle(INT dim,Pt2di p0,Pt2di sz) :
                    _dim     (dim),
                    _lptc    (dim),
                    _nb      (0),
                    _p0      (p0),
                    _sz      (sz),
                    _im_load (false),
                    _imb     (1,1)
               {
               }

               void make_im_bits(bool erase_lpt);
               void set_seg(INT x,INT y,INT nb);
                
               INT                    _dim;
               Liste_Pts<U_INT1,INT>  _lptc;
               INT                    _nb;
               Pt2di                  _p0;
               Pt2di                  _sz;
               bool                   _im_load;
               Im2D_Bits<1>           _imb;
      };

      Pt2di  adr_dalle(Pt2di p)
      {
           return Pt2di((p.x-_up0.x)/256,(p.y-_up0.y)/256);
      }

      Box2di box_dalles(Box2di b)
      {
            return Box2di
                   (
                        adr_dalle(b._p0),
                        adr_dalle(b._p1-Pt2di(1,1))+Pt2di(1,1)
                   );
      }

      //  Pt2di  P0_dalle(Pt2di pd) { return _up0 + pd * 256; }

      bool dump_im_bits (dalle *,INT nb_byte_max);
      void empty_dalle (dalle *);
      dalle * add_dalle(Pt2di pt);


      virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);
      void update_rle(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);
      void update_not_rle(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);

      void flush_image(bool);

      Box2di               _rect;
      bool                _is_rect;
      bool                  _is_cste;
      INT                 _nb_buf_y;
      INT                    _y_cur;
      bool                     _rle;
      INT                     _dout;
      Ps_Multi_Filter          _pmf;
      Elise_Palette            _pal;
      Data_Elise_Palette     * _dep;
      Pt2di                  _p0buf;
      Pt2di                  _p1buf;
      U_INT1 ***            _buf;
      U_INT1 **          _buf_tr;

      INT               _last_x0;
      INT               _last_x1;
      INT                _last_y;
      bool                _first;
      Data_Elise_PS_Win     * _w;
      Data_Elise_PS_Disp    *_psd;

      INT                   _vcste[Elise_Std_Max_Dim];

      Data_Col_Pal              _dcp;
      RLE_Pack_Of_Pts *         _rle_pck;  // for convertion from not RLE
      Std_Pack_Of_Pts<INT> *    _rle_values;

      dalle ***                 _dalles;
      Pt2di                     _p0d;
      Pt2di                     _p1d;
      INT                       _nb_min_rle;
      INT                       _d_lpt;
      INT                       _nb_empty;
      U_INT1 *                  _buf_add;

};

PS_Out_RLE_computed::PS_Out_RLE_computed
(
     const Arg_Output_Comp  &      arg,
     const Data_El_Geom_GWin *     degw,
     Data_Elise_Palette  *         dep,
     Data_Elise_PS_Win   *         w,
     bool                          rle
)  :
   GW_computed     (arg,degw,dep),
  _rect            (Pt2di(0,0),Pt2di(0,0)),  // Bidon,of course
  _is_rect         (arg.flux()->is_rect_2d(_rect)),
  _is_cste         (arg.fonc()->icste(_vcste)),
  _nb_buf_y        (_is_rect ? 30 : 1),
  _y_cur           (0),
  _rle             (rle),
  _dout            (dep->ps_dim_out()),
  _pmf             (w->_psd),
  _pal             (dep),
  _dep             (dep),
  _p0buf           (ElMin(0,_up0.x-1),0),
  _p1buf           (ElMax(_up1.x+1,arg.flux()->sz_buf()),_nb_buf_y),
  _buf             (NEW_TAB_MATRICE(_dout,_p0buf,_p1buf,U_INT1)),
  _buf_tr          (NEW_VECTEUR(0,_dout,U_INT1 *)),
  _last_x0         (0xffffffff),
  _last_x1         (0xffffffff),
  _last_y          (0xffffffff),
  _first           (true),
  _w               (w),
  _psd             (w->_psd),


  _rle_pck         (RLE_Pack_Of_Pts::new_pck(2,arg.flux()->sz_buf())),
  _rle_values      (Std_Pack_Of_Pts<INT>::new_pck(2,0)),
  _dalles          (0),
  _p0d             (adr_dalle(_up0)),
  _p1d             (adr_dalle(_up1-Pt2di(1,1))+Pt2di(1,1)),
  _nb_min_rle      (10),
  _d_lpt           (2 + (_is_cste ? 1 : _dout)),
  _nb_empty        (_is_cste ? 2500:_psd->max_str),
  _buf_add         (NEW_VECTEUR(0,2+_dout,U_INT1))
{


    if (_is_cste)
    {
       _dcp = Data_Col_Pal(_pal,_vcste[0],_vcste[1],_vcste[2]);
    }

    _dalles = NEW_MATRICE(_p0d,_p1d,dalle *);
    for (INT x=_p0d.x ; x<_p1d.x ; x++)
        for (INT y=_p0d.y ; y<_p1d.y ; y++)
            _dalles[y][x] = 0;
}

PS_Out_RLE_computed::~PS_Out_RLE_computed()
{
     flush_image(true);

     DELETE_VECTOR(_buf_add,0);
     if (_dalles)
     {
         for (INT x=_p0d.x ; x<_p1d.x ; x++)
             for (INT y=_p0d.y ; y<_p1d.y ; y++)
                 if (_dalles[y][x])
                 {
                    empty_dalle(_dalles[y][x]);
                    delete _dalles[y][x];
                 }

        DELETE_MATRICE(_dalles,_p0d,_p1d);
     }
     delete _rle_values;
     delete _rle_pck;
     DELETE_VECTOR(_buf_tr,0);
     DELETE_TAB_MATRICE(_buf,_dout,_p0buf,_p1buf);
}


PS_Out_RLE_computed::dalle * PS_Out_RLE_computed::add_dalle(Pt2di pt)
{

    Pt2di pd = adr_dalle(pt);
    Pt2di p0 = _up0+pd*256;

    if (! _dalles[pd.y][pd.x])
       _dalles[pd.y][pd.x] = new dalle
                                 (
                                    _d_lpt,
                                    p0,
                                    Inf(Pt2di(256,256),Pt2di(_up1-p0))
                                 );
    return _dalles[pd.y][pd.x];
}


void PS_Out_RLE_computed::flush_image(bool /* end */)
{
    if ((_first) || (_last_x0==_last_x1))
       return;

    INT yl = (_last_y-_up0.y)%256;

    if (_is_cste)
    {
        Box2di bd= box_dalles
                   (
                        Box2di
                        (
                           Pt2di(_last_x0,_last_y),
                           Pt2di(_last_x1,_last_y)
                        )
                   );
        for (INT idx=bd._p0.x; idx<bd._p1.x ; idx++)
        {
            INT x0 = ElMax(_last_x0,_up0.x+256*idx);
            INT x1 = ElMin(_last_x1,_up0.x+256*(idx+1));
            INT x0l = (x0-_up0.x)%256;
            INT nb = x1-x0;

            dalle * d = add_dalle(Pt2di(x0,_last_y));
            d->_nb ++;
            if (d->_nb == nb_seg_max_cste)
               d->make_im_bits(true);

            if (d->_im_load)
               d->set_seg(x0l,yl,nb);
            else
            {
                for (INT dy =0; dy <_y_cur ; dy++)
                {
                    _buf_add[0] = x0l;
                    _buf_add[1] = yl-dy;
                    _buf_add[2] = nb-1;
                    d->_lptc.add_pt(_buf_add);
                }
            }
        }
    }
    else
    {
        if (_rle || ((_last_x1-_last_x0) >= _nb_min_rle))
        {
           _pmf.put
           (
              _w,
              _pal,
              _buf,
              _dout,
              Pt2di(_last_x1-_last_x0,_y_cur),
              Pt2di(_last_x0,_last_y+1-_y_cur),
              Pt2di(_last_x0,0),
              -1
           );
        }
        else
        {
            for (INT x = _last_x0; x<_last_x1 ; x++)
            {
                  dalle * d = add_dalle(Pt2di(x,_last_y));
                 _buf_add[0] = (x-_up0.x)%256;
                 _buf_add[1] = yl;
                 d->_nb ++;
                 for (INT dim=2 ; dim<_d_lpt ; dim++)
                     _buf_add[dim] = _buf[dim-2][0][x];
                 d->_lptc.add_pt(_buf_add);
                 if (d->_nb == _nb_empty)
                    empty_dalle(d);
            }
        }
    }
}

void PS_Out_RLE_computed::empty_dalle(PS_Out_RLE_computed::dalle * d)
{


    if (_is_cste)
    {
       if (d->_im_load)
       {
          dump_im_bits(d,-1);
       }
       else
       {
          d->make_im_bits(false);
          if(! dump_im_bits(d,3*d->_lptc.card()))
          {
             PS_Pts_Not_RLE pnrle(_w,_pal);
             Im2D<U_INT1,INT> i = d->_lptc.image();
             pnrle.put_1L
             (
                d->_p0,
                i.data()[0],
                i.data()[1],
                i.data()[2],
                &_dcp,
                d->_nb
             );
          }
       }
    }
    else
    {
       PS_Pts_Not_RLE pnrle(_w,_pal);
       Im2D<U_INT1,INT> i = d->_lptc.image();
       pnrle.put_1L
       (
          d->_p0,
          i.data()[0],
          i.data()[1],
          i.data()+2,
          d->_nb,
          false
       );
     }

    d->_nb = 0;
    d->_lptc = Liste_Pts<U_INT1,INT>(_d_lpt);
}


void PS_Out_RLE_computed::dalle::set_seg(INT x0,INT y,INT nb)
{
     for (INT k=0; k< nb ; k++)
         _imb.set(x0+k,y,1);
}

void PS_Out_RLE_computed::dalle::make_im_bits(bool erase_lpt)
{
    El_Internal.ElAssert
    (
       (! _im_load),
       EEM0 << "Multiple  PS_Out_RLE_computed::dalle::make_im_bits"
    );
    _im_load = true;

    Im2D<U_INT1,INT> i = _lptc.image();
    if (erase_lpt)
        _lptc = Liste_Pts<U_INT1,INT>(_dim);

    _imb = Im2D_Bits<1>(_sz.x,_sz.y,0);      

    INT  nbseg = i.tx();
    U_INT1 *  x =  i.data()[0];
    U_INT1 *  y =  i.data()[1];
    U_INT1 *  nb =  i.data()[2];

    for (INT iseg = 0; iseg<nbseg ; iseg++)
        set_seg(x[iseg],y[iseg],nb[iseg]+1);
}

bool PS_Out_RLE_computed::dump_im_bits(PS_Out_RLE_computed::dalle * d,INT nb_byte_max)
{
   return _pmf.put
          (
                _w,
                &_dcp,
                d->_imb.data(),
                d->_sz,
                d->_p0,
                Pt2di(0,0),
                nb_byte_max
          );
}

void PS_Out_RLE_computed::update_rle(const Pack_Of_Pts * p,const Pack_Of_Pts * v)
{
    INT nb = p->nb();
    if (! nb)
       return;

    const Std_Pack_Of_Pts<INT> * ivals = v->int_cast();
    const RLE_Pack_Of_Pts * rle_pack = p->rle_cast();
    INT ** vals =  ivals->_pts;

    _dep->verif_values_out(vals,nb);

    INT x0 = rle_pack->vx0();
    INT x1 = x0 + nb;
    INT y  = rle_pack->pt0()[1];

    if (_first)
    {
       _first = false;
       _last_x0 = _last_x1 = x0;
       _last_y = y-1;
    }

    if ((_last_y != y) || (_last_x1 != x0))
    {
        if (_y_cur == _nb_buf_y)
        {
            flush_image(false);
            _y_cur = 0;
        }
        _last_x0 = x0;
        _last_y = y;
        _y_cur++;
    }
    _last_x1 = x1;

    for (INT d = 0; d < _dout ; d++)
        _buf_tr[d] = _buf[d][_y_cur-1] + x0;

    _w->_psd->use_conv_colors(_dep,_buf_tr,vals,_dout,nb);
}

void PS_Out_RLE_computed::update_not_rle(const Pack_Of_Pts * p,const Pack_Of_Pts * v)
{
     const Std_Pack_Of_Pts<INT> * pts = p->int_cast();
     const Std_Pack_Of_Pts<INT> * values = v->int_cast();

     INT ** xy = pts->_pts;
     INT *  x  = xy[0];
     INT *  y  = xy[1];
     INT nb    = pts->nb();

     for (INT k1=0 ; k1< nb ; )
     {
         INT k2 = k1 +1;
         while (
                     (k2<nb)
                  && (x[k2] == x[k2-1] +1)
                  && (y[k2] == y[k2-1])
               )
               k2++;

         _rle_pck->set_nb(k2-k1); 
         _rle_pck->set_pt0(Pt2di(x[k1],y[k1]));
         _rle_values->interv(values,k1,k2);
         update_rle(_rle_pck,_rle_values);
         k1 = k2;
     }
}




void PS_Out_RLE_computed::update(const Pack_Of_Pts * p,const Pack_Of_Pts * v)
{
    if (_rle)
       update_rle(p,v);
    else
       update_not_rle(p,v);
}



 
Output_Computed * Data_Elise_PS_Win::rle_out_comp
(
      const Data_El_Geom_GWin * degw,
      const Arg_Output_Comp & arg,
      Data_Elise_Palette *    dep,
      bool                    OnYDiff
)
{
      return new PS_Out_RLE_computed (arg,degw,dep,this,true);
}

Output_Computed * Data_Elise_PS_Win::pint_cste_out_comp
(
      const Data_El_Geom_GWin * degw,
      const Arg_Output_Comp &   arg,
      Data_Elise_Palette *      dep,
      INT        *
)
{
     return new PS_Out_RLE_computed (arg,degw,dep,this,false);
}


Output_Computed * Data_Elise_PS_Win::pint_no_cste_out_comp
(
      const Data_El_Geom_GWin * degw,
      const Arg_Output_Comp &   arg,
      Data_Elise_Palette *      dep
)
{
     return new PS_Out_RLE_computed (arg,degw,dep,this,false);
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
