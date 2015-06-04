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
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Tile_F2d                                           ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/

Tile_F2d::Tile_F2d(Packed_Flux_Of_Byte * pfob) :
    _pfob (pfob)
{
}






       //======================
       // Tile_F2d::read_seg
       //======================

       // This def value is adapted to uncompressed file


void Tile_F2d::read_seg(Fich_Im2d *,void * buf,INT x0,INT x1)
{
     _pfob->Read((U_INT1 *)buf,x1-x0);
}

void Tile_F2d::write_seg(Fich_Im2d *,void * buf,INT x0,INT x1)
{
     _pfob->Write((U_INT1 *)buf,x1-x0);
}



        //======================
        // Tile_F2d::seek_in_line
        //======================

        // This def value may be really un-optimal;
        // but :
        //        o it allows a fast adaptation to many formats
        //        o there is sometine no much better to do
        // So, if it is functionnaly OK, it is recommended to use it
        // before adding a more optimized version.


void Tile_F2d::seek_in_line(Fich_Im2d * ,INT x0,INT x1)
{
     _pfob->Rseek(x1-x0);
}

        //=========================
        // Tile_F2d::seek_pack_line
        //=========================

        // same remark than seek_in_line applies

void Tile_F2d::seek_pack_line(Fich_Im2d * fich,INT y0,INT y1,bool read_mode)
{
     for (INT y =y0; y<y1 ; y++)
     {
         if (read_mode)
            r_new_line(fich,y);
         else
            w_new_line(fich,y);
         seek_in_line(fich,0,_sz_tile_phys);

         if (read_mode)
            r_end_line(fich,y);
         else
            w_end_line(fich,y);
     }
}


        //=========================
        // Tile_F2d::r_new_line
        //=========================

        //  this message is usefull, for example with  CCITT bi-dimensional
        //  coding where some specials actions are to be made for each new-line

void Tile_F2d::r_new_line(Fich_Im2d *,INT)
{
}
void Tile_F2d::w_new_line(Fich_Im2d *,INT)
{
}

void Tile_F2d::r_end_line(Fich_Im2d *,INT)
{
}
void Tile_F2d::w_end_line(Fich_Im2d *,INT)
{
}


        //=========================
        // Tile_F2d::new_tile
        //=========================

        //  this message is usefull, for the file
        //  that  are really tiled; with TIFF file, for example
        //  the file will be reinitialized using tiles offset tags


void Tile_F2d::r_new_tile(Fich_Im2d *)
{
}

void Tile_F2d::w_new_tile(Fich_Im2d *)
{
}


void Tile_F2d::r_end_tile(Fich_Im2d *)
{
}

void Tile_F2d::w_end_tile(Fich_Im2d *)
{
}


const INT Tile_F2d::NO_LAST_TIL_Y = - 100000000;

        //=========================
        // ~Tile_F2d()
        //=========================

Tile_F2d::~Tile_F2d()
{
    if (_pfob)
       delete _pfob;
}



        //=========================
        // Tile_F2d::inst_flush()
        //=========================



        //=========================
        // Tile_F2d::inst_flush()
        //=========================

void Tile_F2d::r_use_this_tile(class Fich_Im2d *)
{
}

void Tile_F2d::w_use_this_tile(class Fich_Im2d *)
{
}


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Fich_Im2d                                          ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


Fich_Im2d::Fich_Im2d
(
      Flux_Pts_Computed * flx,
      char *              usr_buf,
      Pt2di               sz_file,
      Pt2di               sz_tiles,
      INT                 sz_el,
      INT                 dim_out,
      bool                integral,
      bool                compressed,
      const char *        name
)
{
     ASSERT_TJS_USER
     (
          flx->type() == Pack_Of_Pts::rle,
          "Can only handle RLE mode for File-Images"
     );

     ASSERT_TJS_USER
     (
          flx->dim() == 2,
          "Writing a 2-dimensional file with a non 2 dimensional flux"
     );

     _sz_el    = sz_el;
     _dim_out  = dim_out;
     _sz_file  = sz_file;
     _sz_til   = sz_tiles;
     _sztx     = _sz_til.x;
     _nb_tiles = (_sz_file.x + _sz_til.x -1) / _sz_til.x;

      _tab_sz[0] = sz_file.x;
      _tab_sz[1] = sz_file.y;
      _tab_or[0] = 0;
      _tab_or[1] = 0;

     _tiles = NEW_VECTEUR(0,_nb_tiles,Tile_F2d *);
     for (INT i=0; i<_nb_tiles ; i++)
          _tiles[i] = 0;

     _tprov_name   = dup_name_std(name);
     _name        = _tprov_name->coord();
     _usr_buf     =  (usr_buf != 0);
     if (_usr_buf)
        _buf = usr_buf;
     else
        _buf    = NEW_VECTEUR(0,_sz_el*flx->sz_buf(),char);

      _integral_type = integral;

      _compressed = compressed;

     _byte_inversed = false;
}

void Fich_Im2d::SetByteInversed()
{
     _byte_inversed = true;
}


void Fich_Im2d::init_tile(Tile_F2d * tile,INT k,INT padding,bool clip_last)
{
    _tiles[k]       = tile;
    tile->_n_tile   = k;
    tile->_sz_tile_log  = ElMin(_sz_til.x,_sz_file.x -k*_sz_til.x);
    tile->_sz_tile_phys = (clip_last ? tile->_sz_tile_log : _sz_til.x);
    tile->_sz_tile_phys = (tile->_sz_tile_phys +padding-1)/padding;
    tile->_sz_tile_phys *= padding;

    tile->_last_til_Y = Tile_F2d::NO_LAST_TIL_Y;
}

void Fich_Im2d::assert_not_wc(bool ) const
{
/*
     Tjs_El_User.assert
     (
          read_mode || (! _compressed),
          EEM0 << "Write in compressed mode : cannot split tiles \n"
               << "      error occured with file " << _name
     );
*/
}


void Fich_Im2d::read_write_buf(const RLE_Pack_Of_Pts * pack,bool read_mode)
{
/*
std::cout << "Fich_Im2d::read_write_buf " << pack->y()
                               << " " << pack->vx0()
                               << " " << pack->x1()
                   << "\n";
*/

    INT mYCurPack = pack->pt0()[1];
    INT mX0CurPack = pack->pt0()[0];
    INT nb = pack->nb();
    if (! nb)
       return;

    INT mX1CurPack = mX0CurPack + nb;

    //  itxK mCurItY : indice de dalles

    INT itx0 = mX0CurPack     / _sztx;
    INT itx1 = (mX1CurPack-1) / _sztx;
    INT mCurItY  =  mYCurPack     / _sz_til.y;

    INT yt   = mYCurPack -mCurItY*_sz_til.y;

    bool last_line_of_tile;
    {
       INT y_end_tile = ElMin(_sz_til.y,_sz_file.y-mCurItY*_sz_til.y)-1;
       last_line_of_tile = (y_end_tile==yt);
    }

    for (int it = itx0, sum_xt = 0; it<= itx1 ; it++)
    {
         Tile_F2d * tile = _tiles[it];
         ASSERT_INTERNAL
         (
                (tile != 0),
               "Non initialized tile"
         );
         if (tile->_last_til_Y == Tile_F2d::NO_LAST_TIL_Y)
         {
            if (read_mode)
               tile->r_use_this_tile(this);
            else
               tile->w_use_this_tile(this);
            tile->_last_til_Y = mCurItY-1;
         }

         if (tile->_last_til_Y != mCurItY)
         {
            tile->_last_til_Y = mCurItY;
            if (read_mode)
               tile->r_new_tile(this);
            else
               tile->w_new_tile(this);
            tile->_last_x = tile->_sz_tile_phys;
            tile->_last_y = -1;
         }

         INT x0t = ElMax(0,mX0CurPack-it*_sztx);
         INT x1t = ElMin(mX1CurPack-it*_sztx,tile->_sz_tile_log);
         INT nbt = x1t-x0t;


     if ( _compressed)
     {
            ASSERT_INTERNAL
            (
                  (yt > tile->_last_y)
               || ((yt == tile->_last_y) && (x0t >= tile->_last_x )),
               "Standard File 2d can only handle forward flux \n"
            );
     }

         if (yt > tile->_last_y)
         {
            if (tile->_last_x != tile->_sz_tile_phys)
            {
               assert_not_wc(read_mode);
               tile->seek_in_line(this,tile->_last_x,tile->_sz_tile_phys);
            }
            if (yt > tile->_last_y+1)
            {
                assert_not_wc(read_mode);
                tile->seek_pack_line(this,tile->_last_y+1,yt,read_mode);
            }
            tile->_last_y = yt;
            tile->_last_x = 0;
         }

         // if we are here, nb!= 0, so nbt != 0;
         // then we are sure that the new line wont be called
         // severall time because of empty line.
         if (tile->_last_x == 0)
         {
             if (read_mode)
                tile->r_new_line(this,yt);
             else
                tile->w_new_line(this,yt);
         }

         if (tile->_last_x != x0t)
         {
             assert_not_wc(read_mode);
             tile->seek_in_line(this,tile->_last_x,x0t);
         }
         if (read_mode)
         {
            tile->read_seg(this,_buf+sum_xt*_sz_el,x0t,x1t);
            if (_byte_inversed)
            {
                byte_inv_tab( _buf+sum_xt*_sz_el,(_sz_el)/ _dim_out,(x1t-x0t)*_dim_out);
            }
         }
         else
         {
            /*
              SS DOUTE LE + SIMPLE, pour gerer en MODE comprime
              ecriture, le fait que les Tiles peuvent avoir une taille
              logique != de leur taille physique :

                     o ajouter dans la methode Rseek par defaut, le
                     fait  que si le fichier est en write, ob fait
                     N ecriture avec des 0 (convention par defaut
                     utilisee en cas de padding).

                   A faire pour l'ecriture dans les fichiers
                   compresses :
                       1) completer le x1t pour qu'il
                       arrive en _sz_tile_phys si il vaut _sz_tile_log;
                       2) recopier buf dans un buffer local
                       2) padder avec des zero
            */

            // Rustine pour gerer le cas ou on ecrit, par ex, en little-indian sur un fichier
        // cree en big-indian  (cas courrant en calcul distribue).  Pour faire simple tou
        // en garantissant de ne pas modifier la donnee, on fait deux inversion
            if (_byte_inversed)
            {
                byte_inv_tab( _buf+sum_xt*_sz_el,(_sz_el)/ _dim_out,(x1t-x0t)*_dim_out);
            }
            tile->write_seg(this,_buf+sum_xt*_sz_el,x0t,x1t);
            if (_byte_inversed)
            {
                byte_inv_tab( _buf+sum_xt*_sz_el,(_sz_el)/ _dim_out,(x1t-x0t)*_dim_out);
            }
         }

         tile->_last_y = yt;
         tile->_last_x = x1t;
         sum_xt += nbt;

         if(x1t == tile->_sz_tile_log)
         {
            if (tile->_last_x != tile->_sz_tile_phys)
            {
               assert_not_wc(read_mode);
               tile->seek_in_line(this,tile->_last_x,tile->_sz_tile_phys);
               tile->_last_x = tile->_sz_tile_phys;
            }
            if (read_mode)
              tile->r_end_line(this,yt);
            else
              tile->w_end_line(this,yt);
            if (last_line_of_tile)
            {
               if (read_mode)
                  tile->r_end_tile(this);
               else
                  tile->w_end_tile(this);
            }
         }
    }
}


Fich_Im2d::~Fich_Im2d()
{
     if (! _usr_buf)
        DELETE_VECTOR(_buf,0);
     delete _tprov_name;
     for (INT i=0; i<_nb_tiles ; i++)
         if (_tiles[i])
             delete _tiles[i];

     DELETE_VECTOR(_tiles,0);
}


void Fich_Im2d::post_traite(Std_Pack_Of_Pts_Gen *)
{
}

const Pack_Of_Pts * Fich_Im2d::pre_traite
                    (
                         const Pack_Of_Pts * values,
                         Pack_Of_Pts *       ,
                         Pack_Of_Pts *
                    )
{
   return values;
}

/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Std_Bitm_Fich_Im_2d                                ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/



Std_Bitm_Fich_Im_2d::Std_Bitm_Fich_Im_2d
(
       Flux_Pts_Computed * flx,
       Pt2di               sz_file,
       Pt2di               sz_tiles,
       INT                 dim_out,
       const char *        name,
       GenIm               gi,
       bool                compressed,
       INT                 sz_el_spec,
       r_special_transf    r_spec_transf,
       w_special_transf    w_spec_transf

)   :
      Fich_Im2d
      (
             flx,
             (char *)(gi.data_im()->data_lin_gen()),
             sz_file,
             sz_tiles,
             dim_out *  (r_spec_transf ? sz_el_spec : gi.data_im()->sz_el()) ,
             dim_out,
             gi.data_im()->integral_type(),
             compressed,
             name
      ),
      _gi (gi),
      _bim (gi.data_im()),
      _r_spec_transf (r_spec_transf),
      _w_spec_transf (w_spec_transf)
{
}

Std_Bitm_Fich_Im_2d::~Std_Bitm_Fich_Im_2d()
{
}


void Std_Bitm_Fich_Im_2d::input_transfere(Std_Pack_Of_Pts_Gen * pack)
{
     if (_r_spec_transf)
       _r_spec_transf(pack,_bim->data_lin_gen());
     else
         _bim->striped_input_rle
         (
              pack->adr_coord(),
              pack->nb(),
              _dim_out,
              _bim->data_lin_gen(),
              0
         );
}

void Std_Bitm_Fich_Im_2d::output_transfere(const Std_Pack_Of_Pts_Gen * pack)
{
     if (_w_spec_transf)
       _w_spec_transf(pack,_bim->data_lin_gen());
     else
         _bim->striped_output_rle
         (
              _bim->data_lin_gen(),
              pack->nb(),
              _dim_out,
              pack->adr_coord(),
              0
         );
}



/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Fonc_Fich_Im2d<Type>                               ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/
template <class Type> class Fonc_Fich_Im2d :  public Fonc_Num_Comp_TPL<Type>
{
       public :
             Fonc_Fich_Im2d
             (
                   const Arg_Fonc_Num_Comp &,
                   Fich_Im2d               *,
                   bool                    with_def_value
             );

         virtual ~Fonc_Fich_Im2d();
         virtual const Pack_Of_Pts * values(const Pack_Of_Pts *);

         Fich_Im2d * _f2d;
         bool        _with_def_value;
};



template <class Type> Fonc_Fich_Im2d<Type>::Fonc_Fich_Im2d
                      (
                          const Arg_Fonc_Num_Comp & arg,
                          Fich_Im2d               * f2d,
                          bool                      with_def_value
                      )   :
           Fonc_Num_Comp_TPL<Type>(arg,f2d->_dim_out,arg.flux()),
           _f2d (f2d),
           _with_def_value (with_def_value)
{

}

template <class Type>  Fonc_Fich_Im2d<Type>::~Fonc_Fich_Im2d()
{
      delete _f2d;
}

template <class Type> const Pack_Of_Pts *
                      Fonc_Fich_Im2d<Type>::values
                      (const Pack_Of_Pts * pack_pts)
{
    const RLE_Pack_Of_Pts * rle_pts = pack_pts->rle_cast();


   if (! rle_pts->inside(_f2d->tab_or(),_f2d->tab_sz()))
   {
       cout << "NAME FILE = [" << _f2d->_name <<"]\n";
       ASSERT_USER
       (
            false,
           "outside reading file in RLE mode"
       );
   }


    this->_pack_out->set_nb(pack_pts->nb());
    _f2d->read_write_buf(rle_pts,true);

    // eventuellement modifiable , par defaut :
    // effectue  un unstrip (par ex, convertion d'un U_INT1* = RVBRVB
    // a un INT ** = RRRR.. VVV.. BBB..
    //
    // si _spec_transf est defini, fait un appel a _spec_tranf
    // par ex, TGA : convertion ad hoc pour les 16 ou 32 bits

    _f2d->input_transfere(this->_pack_out);

    // par defaut ne fait rien; il s'agit d'un post-traitement sur la
    // representation normalisee (au sens ELISE) du signal.
    //
    //  TGA : repasse du BGR au RGB pour le true color; reatalone entre
    //  0 et 255 pour le noir et blanc.
    //
    //  BMP : repasse du BGR au RGB pour le 24 bits.
    //
    //  TIFF : (par exemple, non fait) remet des convention standard / au TAG
    //  blakc_is_0

    _f2d->post_traite(this->_pack_out);
    return this->_pack_out;
}


Fonc_Num_Computed * fonc_num_std_f2d
                    (
                         const Arg_Fonc_Num_Comp &    arg,
                         Fich_Im2d *            f2d,
                         bool           with_def_val,
                         REAL                def_val
                    )
{
     Fonc_Num_Computed * res;

     if (f2d->integral_type())
        res = new Fonc_Fich_Im2d<INT> (arg,f2d,with_def_val);
     else
        res = new Fonc_Fich_Im2d<REAL>(arg,f2d,with_def_val);

     if (with_def_val)
     {
        res = clip_fonc_num_def_val
              (
                   arg,
                   res,
                   arg.flux(),
                   f2d->tab_or(),
                   f2d->tab_sz(),
                   def_val
              );
     }

     return res;
}


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                  Out_Fich_Im2d                                  ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Out_Fich_Im2d :  public Output_Computed
{
     public :
         Out_Fich_Im2d
         (
               const Arg_Output_Comp &,
               Fich_Im2d               *
         );

         virtual ~Out_Fich_Im2d();

     private :


         void update
         (
                const Pack_Of_Pts * pts,
                const Pack_Of_Pts * vals
         );

         Fich_Im2d * _f2d;
         Pack_Of_Pts *    _empty_buf;
         Pack_Of_Pts *    _buf;
};


Out_Fich_Im2d::Out_Fich_Im2d
(
     const Arg_Output_Comp & arg,
     Fich_Im2d               * f2d
)    :
     Output_Computed(f2d->_dim_out)
{

     _f2d = f2d;

     Pack_Of_Pts::type_pack tp =
                 f2d->integral_type() ?
                 Pack_Of_Pts::integer :
                 Pack_Of_Pts::real    ;

     _empty_buf = Pack_Of_Pts::new_pck(f2d->_dim_out,0,tp);
     _buf       = Pack_Of_Pts::new_pck(f2d->_dim_out,arg.flux()->sz_buf(),tp);
}



Out_Fich_Im2d::~Out_Fich_Im2d()
{
    delete _empty_buf;
    delete _buf;
    delete _f2d;
}

void Out_Fich_Im2d::update
(
     const Pack_Of_Pts * pts,
     const Pack_Of_Pts * vals
)
{
    const RLE_Pack_Of_Pts * rle_pts = pts->rle_cast();
    ASSERT_USER
    (
        rle_pts->inside(_f2d->tab_or(),_f2d->tab_sz()),
       "outside reading file in RLE mode"
    );
     const Pack_Of_Pts * v_pre = _f2d->pre_traite(vals,_empty_buf,_buf);
    _f2d->output_transfere(v_pre->std_cast());
    _f2d->read_write_buf(rle_pts,false);
}



Output_Computed * out_std_f2d
                  (
                         const Arg_Output_Comp & arg,
                         Fich_Im2d *            f2d
                  )
{
    ELISE_ASSERT
    (
        f2d->dim_out()<= arg.fonc()->idim_out(),
        "Insufficient image channel when wrintig in Image-File"
    );

    Output_Computed * res = new Out_Fich_Im2d (arg,f2d) ;
    res = out_adapt_type_fonc
          (
                arg,
                res,
                f2d->integral_type()       ?
                Pack_Of_Pts::integer       :
                Pack_Of_Pts::real
          );

    res = clip_out_put
          (
               res,
               arg,
               f2d->tab_or(),
               f2d->tab_sz()
          );

    return res;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
