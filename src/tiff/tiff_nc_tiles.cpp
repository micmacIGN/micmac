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
/***              Tiff_Tiles                                         ***/ 
/***                                                                 ***/ 
/***                                                                 ***/ 
/***********************************************************************/
/***********************************************************************/

Tiff_Tiles::Tiff_Tiles() : 
      Tile_F2d(0) ,
      mCurNumYTileFile (-1),
      mTifTile         (0),
      mDTITile         (0)
{}

Tiff_Tiles::~Tiff_Tiles()
{
   delete mTifTile;
}

void Tiff_Tiles::new_tile(Fich_Im2d * f2d,bool read_mode)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;
     tFileOffset offs = Tiff_Im::UN_INIT_TILE;

     if (dti->mUseFileTile)
     {
         int itFY = _last_til_Y / dti->mNbTTByTF.y;
	 if (itFY != mCurNumYTileFile)
	 {
            delete _pfob;
	    delete mTifTile;
	    int itFX = _n_tile / dti->mNbTTByTF.x;
	    std::string aName = dti->NameTileFile(Pt2di(itFX,itFY));
	    mTifTile = new Tiff_Im (aName.c_str());
	    mDTITile = mTifTile->dtifd();
            _pfob = init_pfob(mDTITile,tf2d,read_mode);
	    mCurNumYTileFile = itFY;
	 }
	 offs = mDTITile->offset_tile
		(
                        _n_tile     % dti->mNbTTByTF.x,
                        _last_til_Y % dti->mNbTTByTF.y,
			tf2d->_kth_ch
                );
     }
     else
     {
         offs = dti->offset_tile(_n_tile,_last_til_Y,tf2d->_kth_ch);
     }


     Tjs_El_User.ElAssert
     (
            offs != Tiff_Im::UN_INIT_TILE,
            EEM0 << " Probable use of incompletly initialized Tiff file\n"
                 << "|   FILE = " << dti->_name  << "\n"
                 << "|   Unitialized tile = (" 
                 << _n_tile << "," <<_last_til_Y  << ")"
     );

     _pfob->AseekFp(offs);
}

Std_Packed_Flux_Of_Byte * Tiff_Tiles::init_pfob
                          (
                             DATA_Tiff_Ifd * dti,
                             Tiff_file_2d * tf2d,
                             bool           read_mode
                          )
{
    INT sz_el;

    if (dti->_nbb_ch0 < 8)
       sz_el = 1;
    else if (dti->_mode_compr!= Tiff_Im::No_Compr)
       sz_el = 1;
    else
       sz_el = (dti->_nbb_ch0/8) * tf2d->_nb_chan;
       
     return new Std_Packed_Flux_Of_Byte
                (
                     dti->_name,
                     sz_el,
                     0,
                     (read_mode ? ELISE_fp::READ : ELISE_fp::READ_WRITE)
                );
}

/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/ 
/***                                                                 ***/ 
/***              Tiff_file_2d - Tiff_Tiles_NC                       ***/ 
/***                                                                 ***/ 
/***                                                                 ***/ 
/***********************************************************************/
/***********************************************************************/




void Tiff_Tiles_NC::use_this_tile(Fich_Im2d * f2d,bool read_mode)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;

     if (dti->mUseFileTile)
     {
         mCurNumYTileFile = -1;
     }
     else
     {

        _pfob = Tiff_Tiles::init_pfob(dti,tf2d,read_mode);

        if (dti->_nbb_ch0 < 8)
           _pfob = new BitsPacked_PFOB
                     (
                         _pfob,
                         dti->_nbb_ch0,
                         dti->_msbit_first,
                         read_mode,
                         tf2d->_nb_chan
                     );
     }
}

void Tiff_Tiles_NC::r_use_this_tile(class Fich_Im2d * f2d)
{

     use_this_tile(f2d,true);
}
void Tiff_Tiles_NC::w_use_this_tile(class Fich_Im2d * f2d)
{
     use_this_tile(f2d,false);
}





void Tiff_Tiles_NC::r_new_tile(class Fich_Im2d * f2d)
{
     Tiff_Tiles::new_tile(f2d,true);
}

void Tiff_Tiles_NC::w_new_tile(class Fich_Im2d * f2d)
{
     Tiff_Tiles::new_tile(f2d,false);
}

void Tiff_Tiles_NC::seek_pack_line (Fich_Im2d *,INT y0,INT y1,bool)
{
  tLowLevelFileOffset anOffset = tLowLevelFileOffset(y1-y0)* tLowLevelFileOffset(_sz_tile_phys);
  _pfob->Rseek(anOffset);
}


bool Tiff_file_2d::is_tile_front
     (
         INT x,
         INT tx,
         INT sz_tile,
         INT & num
     )
{
    x = ElMax(0,ElMin(x,tx));

    if (x==tx)
    {
       tx = (tx + sz_tile-1)/sz_tile;
       tx *= sz_tile;
       x = tx;
    }

    if (x%sz_tile)
       return false;
    num = x/sz_tile;
    return true;
}


bool Tiff_file_2d::is_tile_corner
     (
        Pt2di  p,
        Pt2di  sz,
        Pt2di  tile,
        Pt2di  & num
     )
{
    return
             is_tile_front(p.x,sz.x,tile.x,num.x)
          && is_tile_front(p.y,sz.y,tile.y,num.y);
}



Tiff_file_2d::Tiff_file_2d
(
       Flux_Pts_Computed * flx,
       DATA_Tiff_Ifd *     dti,
       bool                read_mode,
       INT                 nb_chan,
       INT                 kth_channel,
       Tiff_file_2d *     princ
)   :
    Std_Bitm_Fich_Im_2d
    (
          flx,
          dti->_sz,
          dti->_sz_tile,
          nb_chan,
          dti->_name,
          alloc_im1d
          (
              dti->_unpacked_type_el,
              flx->sz_buf()* nb_chan
          ),
          (dti->_mode_compr!= Tiff_Im::No_Compr)
    ),
    _dti (dti),
    _nb_chan (nb_chan),
    _kth_ch (kth_channel),
    _princ  (princ),
    _stdpf   (0),
    _fp      (_Bidon_for_ref),   // because warning to non init
    _read    (read_mode)
{

    if (!dti->_byte_ordered)
       Fich_Im2d::SetByteInversed();

    for (INT i = 0; i<_nb_tiles ; i++)
       init_tile 
       (
            alloc_tile(),
            i,
            dti->_padding_constr,
            dti->_clip_last
       );

    if ( (! read_mode) && (dti->_mode_compr!=Tiff_Im::No_Compr))
    {
       Box2di b(Pt2di(0,0),Pt2di(1,1));

       Tjs_El_User.ElAssert
       (
            flx->is_rect_2d(b),
            EEM0 
              << "Output compressed Tiff-file with non rectangular flux\n"
              << "|    file : " << dti->_name
       );

       Pt2di n0,n1;

       Tjs_El_User.ElAssert
       (
               is_tile_corner(b._p0,dti->_sz,dti->_sz_tile,n0)
           &&  is_tile_corner(b._p1,dti->_sz,dti->_sz_tile,n1),
            EEM0 
              << "Output compressed Tiff-file \n"
              << "| Rectangles frontiers do not match  tiling \n"
              << "|    file : " << dti->_name
       );

       _single_tile = 
                        ((ElAbs(n1.x-n0.x)<=1) || (dti->_sz_tile.y == 1))
                     && (
                              (dti->_plan_conf==Tiff_Im::Chunky_conf)    
                           || (dti->_nb_chanel == 1)
                        );

    }
}

Tiff_file_2d::~Tiff_file_2d()
{
     if (_stdpf && (!_princ))
        delete _stdpf;
}


ELISE_fp Tiff_file_2d::_Bidon_for_ref;

void Tiff_file_2d::init_stdpf()
{
    if (! _stdpf)
    {
       if (_princ)
          _stdpf = _princ->_stdpf;
      
       else     
       {
           _stdpf = new Std_Packed_Flux_Of_Byte
                         (_dti->_name,1,0,ELISE_fp::READ_WRITE);
       }
       _fp = _stdpf->fp();
       _fp.seek_end(0);
    }
}


/*********************************************************************/
/*                                                                   */
/*                  Tiff_Im_Not_Comp                                 */
/*                                                                   */
/*********************************************************************/

class Tiff_Im_Not_Comp : public Fonc_Num_Not_Comp
{
      public :

        Tiff_Im_Not_Comp(Tiff_Im,bool wdv,REAL def_val,INT nb_chan,INT kth_ch);

      private :

 
         Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);


         bool integral_fonc(bool) const
         {
              return _integral;
         }

         INT dimf_out() const { return _nb_chan; }
         void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

         Tiff_Im           _tim;
         bool              _with_def_value;
         REAL              _def_value;
         INT               _nb_chan;
         INT               _kth_ch;
         bool              _integral;
};

Tiff_Im_Not_Comp::Tiff_Im_Not_Comp
(
        Tiff_Im tim,
        bool wdv,
        REAL def_val,
        INT nb_chan,
        INT kth_ch
)  :
    _tim (tim),
    _with_def_value (wdv),
    _def_value      (def_val),
    _nb_chan        (nb_chan),
    _kth_ch         (kth_ch),
    _integral       (type_im_integral(_tim.dtifd()->_type_el))
{

}


Fonc_Num_Computed * Tiff_Im_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
     DATA_Tiff_Ifd  * dtifd = _tim.dtifd();
     
     Tiff_file_2d *tf2d = 
               new Tiff_file_2d
                   (
                        arg.flux(),
                        dtifd,
                        true,
                        _nb_chan,
                        _kth_ch
                   );
     
     return fonc_num_std_f2d
            (
                 arg,
                 tf2d,
                 _with_def_value,
                 _def_value
            );
}

Fonc_Num Tiff_Im::in(bool wdef,REAL def)
{
     verif_usable(true);

     DATA_Tiff_Ifd  * dt = dtifd();

     if (dt->_plan_conf == Tiff_Im::Chunky_conf)
         return new Tiff_Im_Not_Comp
                    (*this,wdef,def,dt->_nb_chanel,0);
     
    Fonc_Num res = new Tiff_Im_Not_Comp(*this,wdef,def,1,0);

   for (INT i=1 ; i<dt->_nb_chanel; i++)
       res = Virgule(res, Fonc_Num(new Tiff_Im_Not_Comp(*this,wdef,def,1,i)));

   return res;
}

Fonc_Num Tiff_Im::in()
{
     return in(false,0.0);
}

Fonc_Num Tiff_Im::in(REAL def)
{
     return in(true,def);
}

Fonc_Num  Tiff_Im::in_gen(eModeCoul aModCoul,eModeProl aModeProl,REAL aDef)
{
   Fonc_Num aF = this->in();
   if (aModeProl == eModeProlProj)
      aF = this->in_proj();
   if (aModeProl == eModeProlDef)
      aF = this->in(aDef);

   if (aModCoul==eModeCoulStd)
      return aF;

   Symb_FNum aSF(aF);
   switch (this->phot_interp())
   {
          case RGB :
              if (aModCoul==eModeCoulGray)
                 aF = (aSF.v0()+aSF.v1()+aSF.v2()) / 3;

          break;

	  case BlackIsZero :
              if (aModCoul==eModeCoulRGB)
                 aF = Virgule(aSF,aSF,aSF);
          break;

	  default :
	     ELISE_ASSERT(false,"Coul Mode in Tiff_Im::in_gen");
          break;

   }

   return aF;
}
 




/*********************************************************************/
/*                                                                   */
/*                  Tiff_Out_Not_Comp                                */
/*                                                                   */
/*********************************************************************/

class Tiff_Out_Not_Comp : public Output_Not_Comp
{
      public :

        Tiff_Out_Not_Comp(Tiff_Im,INT nb_chan,INT kth_ch,Tiff_Out_Not_Comp * princ = 0);

      private :

         Output_Computed * compute(const Arg_Output_Comp  &);

         Tiff_Im             _tim;
         INT                 _nb_chan;
         INT                 _kth_ch;
         Tiff_Out_Not_Comp * _princ;
         Tiff_file_2d      * _tf2d ; 
};

Tiff_Out_Not_Comp::Tiff_Out_Not_Comp
(
       Tiff_Im tim,
       INT nb_chan,
       INT kth_ch,
       Tiff_Out_Not_Comp * princ
)  :
   _tim     (tim),
   _nb_chan (nb_chan),
   _kth_ch  (kth_ch),
   _princ   (princ)
{
}


Output_Computed * Tiff_Out_Not_Comp::compute(const Arg_Output_Comp  & arg)
{
     DATA_Tiff_Ifd  * dtifd = _tim.dtifd();

     
      _tf2d = new  Tiff_file_2d
                   (
                        arg.flux(),
                        dtifd,
                        false,
                        _nb_chan,
                        _kth_ch,
                        (_princ ? _princ->_tf2d : 0)
                   );
     
     return out_std_f2d (arg,_tf2d);
}

Output Tiff_Im::out()
{
   verif_usable(false);

   DATA_Tiff_Ifd  * dt = dtifd();


   if (dt->_plan_conf == Tiff_Im::Chunky_conf)
         return new Tiff_Out_Not_Comp(*this,dt->_nb_chanel,0);
     
   Tiff_Out_Not_Comp * out0 = new Tiff_Out_Not_Comp(*this,1,0);

   Output res(out0);
   for (INT i=1 ; i<dt->_nb_chanel; i++)
       res = Virgule(res, new Tiff_Out_Not_Comp(*this,1,i,out0));

   return res;
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
