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



class Data_BMP_File : public RC_Object
{
    friend class  Bmp_Im_Not_Comp;
    friend class  BMP_tile;
    friend class  Bmp_Im;
    friend class  BMP_file_2d;
    friend class  BMP_Rle_8_PFOB;

    public :

       Pt2di sz() const { return Pt2di(_szx,_szy);}


       Data_BMP_File(const char * name);

       typedef enum
       {
           msw3 = 0x4d42
       } VERSIONS;

       typedef enum
       {
           absolute,
           rle,
           transition
       } mode_rle;

       virtual  ~Data_BMP_File(){delete _tprov_name;}

    private :

       Disc_Pal         _pal;
       Tprov_char *     _tprov_name;
       char *           _name;

       INT              _version;
       INT              _szofile;   // why not !!
       tFileOffset      _offs_im;

       INT              _sz_header;
       tFileOffset              _offs_cmap;
       INT              _szx;
       INT              _szy;
       INT              _bits_pp;
       Bmp_Im::mode_compr _compr;

       INT               _sz_bitm;
       INT               _H_resol;
       INT               _V_resol;
       INT               _sz_cmap;
       INT               _nb_imp_col;

       INT               _nb_chan;
};

typedef struct
{
     U_INT1 blue;
     U_INT1 green;
     U_INT1 red;
     U_INT1 reserved;
} BMP_col;

/*********************************************************************/
/*                                                                   */
/*                  Data_BMP_File                                    */
/*                                                                   */
/*********************************************************************/

Data_BMP_File::Data_BMP_File(const char * name) :
   _pal (Disc_Pal::PBIN())
{
    _tprov_name   = dup_name_std(name);
    _name         = _tprov_name->coord();

    ELISE_fp fp (name,ELISE_fp::READ);


  //========================================
  // Bitmap header
  //========================================

    _version  = fp.lsb_read_U_INT2();
    ASSERT_TJS_USER
    (
        _version == msw3,
        "Elise bmp : handle only microsoft version 3"
    );
    _szofile = fp.lsb_read_INT4();

/*
    ASSERT_TJS_USER
    (
       _szofile == sizeofile(name),
       "bmp size of tag incoherent with system size of file"
    );
*/

    fp.lsb_read_U_INT2(); // reserved
    fp.lsb_read_U_INT2(); // reserved

    _offs_im = fp.lsb_read_INT4();


  //========================================
  // Bitmap header
  //========================================

    _offs_cmap = fp.tell();
cout << "offs_header "<< fp.tell() << "\n";
    _sz_header = fp.lsb_read_INT4();  // size header
    _offs_cmap += _sz_header;

cout << "sz_header "<< _sz_header << "\n";
     if (_sz_header == 12)
     {
          _szx = fp.lsb_read_U_INT2();
          _szy = fp.lsb_read_U_INT2();
#if  (DEBUG_INTERNAL)
          INT nb_pl = fp.lsb_read_U_INT2(); // nb_plane
          ASSERT_INTERNAL(nb_pl==1,"incoherenc in bmp file");
#else
          fp.lsb_read_U_INT2();
#endif

          _bits_pp = fp.lsb_read_U_INT2();
          _compr = Bmp_Im::no_compr;

           //_sz_bitm  do not use
           //_H_resol   do not use
           //_V_resol   do not use

           _sz_cmap = (_bits_pp == 24) ? 0 : 1 << _bits_pp;
           _nb_imp_col = _sz_cmap;   // why not ?
     }
     else if (_sz_header == 40)
     {
         _szx = fp.lsb_read_INT4();
         _szy = fp.lsb_read_INT4();

         fp.lsb_read_U_INT2(); // nb plane == 1 in version 3
         _bits_pp = fp.lsb_read_U_INT2();
         _compr = (Bmp_Im::mode_compr) fp.lsb_read_INT4();

         _sz_bitm = fp.lsb_read_INT4();
         _H_resol = fp.lsb_read_INT4();
         _V_resol = fp.lsb_read_INT4();


         _sz_cmap = fp.lsb_read_INT4();
         if (_bits_pp == 24)
            _sz_cmap = 0;
         else if (_sz_cmap == 0)
             _sz_cmap = 1 << _bits_pp;


         _nb_imp_col = fp.lsb_read_INT4();
    }
    else
        elise_internal_error("unknown  bmp header size",__FILE__,__LINE__);


    _nb_chan = (_bits_pp == 24) ? 3 : 1;
//----------------------------
    printf("BMP version %x \n",_version);
    cout
          << "; sz [" << _szx << "," << _szy << "]"
          << " bpp : " << (int)_bits_pp
          << " compr : " << (int)_compr
          << "_sz_cmap "  << (int)_sz_cmap
          << "\n";

    cout << _offs_cmap  << " " << fp.tell() << "\n";

//----------------------------


    ASSERT_INTERNAL(_offs_cmap == fp.tell(),"incoherence in BMP file");
    if (_sz_cmap)
    {
       Elise_colour * tec = NEW_VECTEUR(0,_sz_cmap,Elise_colour);
       BMP_col        bc ;

       for (int i = 0; i<_sz_cmap ; i++)
       {

           if (_sz_header == 12)
           {
              bc.blue   = fp.lsb_read_U_INT1();
              bc.green  = fp.lsb_read_U_INT1();
              bc.red    = fp.lsb_read_U_INT1();
           }
           else
              fp.read(&bc,sizeof(bc),1);
           tec[i] =  Elise_colour::rgb
                     (
                        bc.red/255.0,
                        bc.green/255.0,
                        bc.blue/255.0
                     );
       }
       _pal = Disc_Pal(tec,_sz_cmap);
       DELETE_VECTOR(tec,0);
    }

    ASSERT_INTERNAL(_offs_im ==  fp.tell(),"incoherence in BMP file");

    fp.close();
}

/*********************************************************************/
/*                                                                   */
/*                 Bmp_Im2d                                          */
/*                                                                   */
/*********************************************************************/

class BMP_Rle_8_PFOB : public Packed_Flux_Of_Byte
{
     public :
         BMP_Rle_8_PFOB(Data_BMP_File *);

         virtual ~BMP_Rle_8_PFOB()
         {
             _fp.close();
         }
     private :

         virtual tFileOffset tell()
         {
              return _fp.tell();
         }

         virtual tFileOffset Read(U_INT1 * res,tFileOffset nb);

         Data_BMP_File::mode_rle _mode;
         Data_BMP_File *         _bmp;

         ELISE_fp  _fp;
         tFileOffset    _nb_buffered;
         INT    _rle_val;
         bool   _pad;
         INT    _nb_tot;

         bool compressed() const { return true;}
};


BMP_Rle_8_PFOB::BMP_Rle_8_PFOB(Data_BMP_File * bmp) :
     Packed_Flux_Of_Byte (sizeof(U_INT1)),
     _mode               (Data_BMP_File::transition),
     _bmp                (bmp),
     _fp                 (bmp->_name,ELISE_fp::READ),
     _nb_buffered        (0),
     _rle_val            (0),  // why not
     _pad                (false),  // why not
    _nb_tot              (0)
{
       _fp.seek_begin(bmp->_offs_im);
}

tFileOffset BMP_Rle_8_PFOB::Read(U_INT1 * res,tFileOffset nbo)
{
    int nb = nbo.CKK_IntBasicLLO();
    for (int i=0 ; i < nb ;)
    {
        if (_nb_buffered == 0)
           _mode =  Data_BMP_File::transition;

        switch (_mode)
        {
              case Data_BMP_File::transition :
              {
                    INT c1 = _fp.lsb_read_U_INT1();
                    if (c1)
                    {
                        _nb_buffered = c1;
                        _rle_val = _fp.lsb_read_U_INT1();
                        _mode = Data_BMP_File::rle;
                    }
                    else
                    {
                         INT c2 = _fp.lsb_read_U_INT1();
                         if (c2 >= 3)
                         {
                             _nb_buffered = c2;
                             _pad = ((c2%2) != 0);
                             _mode = Data_BMP_File::absolute;
                         }
                         else
                         {
                             INT nb_loc = _nb_tot + i;
                             INT y = nb_loc/ _bmp->_szx;
                             INT x = nb_loc% _bmp->_szx;
                             if (c2 == 0)
                             {
                                if (x == 0) // we are at end of previous line
                                            //  and not begin of present ?
                                   _nb_buffered = 0;
                                else
                                   _nb_buffered = _bmp->_szx - x;
                             }
                             else if (c2 == 1)
                                _nb_buffered = _bmp->_szx*(_bmp->_szy-y)-x;
                             else
                             {
                                 INT dx = _fp.lsb_read_U_INT1();
                                 INT dy = _fp.lsb_read_U_INT1();
                                 _nb_buffered = _bmp->_szx* dy + dx;
                             }
                             _rle_val = 0;
                             _mode = Data_BMP_File::rle;
                         }
                    }
              }
              break;

              case Data_BMP_File::rle:
              {
                   INT nb_to_read = ElMin(nb-i,_nb_buffered.CKK_IntBasicLLO());
                    memset(res+i,_rle_val,nb_to_read);
                    i+= nb_to_read;
                   _nb_buffered -= nb_to_read;
              }
              break;


              case Data_BMP_File::absolute:
              {
                   INT nb_to_read = ElMin(nb-i,_nb_buffered.CKK_IntBasicLLO());
                   _fp.read(res+i,sizeof(U_INT1),nb_to_read);
                   i += nb_to_read;
                   _nb_buffered -= nb_to_read;
                   if ((_nb_buffered == 0) && _pad)
                       _fp.lsb_read_U_INT1();
              }
              break;

        }
    }
   _nb_tot += nb;
   return nb;
}


class BMP_tile : public Tile_F2d
{
    public :
        BMP_tile(Data_BMP_File * bmpf,Packed_Flux_Of_Byte *) ;

    private :

        Data_BMP_File * _bmp;
        INT     _szl;


    // redefine, just to avoid useless call to new line
    void seek_pack_line(Fich_Im2d *,INT,INT,bool)
    {
    }

    void r_new_line(Fich_Im2d *,INT y)
    {
         y = _bmp->_szy-y-1;
        _pfob->AseekFp(tFileOffset(y)*_szl+_bmp->_offs_im);
    }

    void w_new_line(Fich_Im2d * f,INT y)
    {
         r_new_line(f,y);
    }

};

BMP_tile::BMP_tile(Data_BMP_File * bmp,Packed_Flux_Of_Byte * pfob)  :
    Tile_F2d(pfob),
    _bmp (bmp)
{
     _szl  = (_bmp->_szx *_bmp->_bits_pp); // number of bits
     _szl  = round_up(_szl/32.0); // number of 32 bits "padded"
     _szl  = _szl *4;
}


class  BMP_file_2d : public Std_Bitm_Fich_Im_2d
{
   public :
       BMP_file_2d(Flux_Pts_Computed *,Data_BMP_File *,bool);

   private :
       Data_BMP_File * _bmp;

       void post_traite(Std_Pack_Of_Pts_Gen * pack)
       {
            if (_bmp->_bits_pp == 24)
               pack->rgb_bgr();
       }

       const Pack_Of_Pts * pre_traite
             (
                         const Pack_Of_Pts * values,
                         Pack_Of_Pts *       empty_buf,
                         Pack_Of_Pts *
             )
       {
            if (_bmp->_bits_pp != 24)
               return values;

            Std_Pack_Of_Pts<INT> * e = const_cast<Std_Pack_Of_Pts<INT> *>(empty_buf->std_cast((INT *)0));

            e->rgb_bgr(values->std_cast());
            return e;

       }

};

BMP_file_2d::BMP_file_2d
(
        Flux_Pts_Computed * flx,
        Data_BMP_File * bmpf,
        bool read_mode
) :
  Std_Bitm_Fich_Im_2d
  (
     flx,
     bmpf->sz(),
     bmpf->sz(),
     bmpf->_nb_chan,
     bmpf->_name,
     alloc_im1d
     (
         GenIm::u_int1,
         flx->sz_buf() *bmpf->_nb_chan
     ),
     (bmpf->_compr!=Bmp_Im::no_compr)
  ),
   _bmp (bmpf)
{
   switch (bmpf->_compr)
   {
        case Bmp_Im::no_compr :
        {
             Packed_Flux_Of_Byte * flb
                    =  new Std_Packed_Flux_Of_Byte
                            (
                                bmpf->_name,
                                bmpf->_nb_chan,
                                bmpf->_offs_im.CKK_IntBasicLLO(),
                                read_mode                ?
                                     ELISE_fp::READ      :
                                     ELISE_fp::READ_WRITE
                             );

             if (bmpf->_bits_pp < 8)
               flb = new BitsPacked_PFOB
                       (flb,bmpf->_bits_pp,true,read_mode,bmpf->_nb_chan);
             init_tile(new BMP_tile(bmpf,flb),0,1,true);
        }
        break;

        case Bmp_Im::rle_8bits :
        {
            /*   There is not padding for bmp file compressed: Got the following
                 files and read them correctly :
                   o  Elise:~/ELISE> ; sz [134,134] bpp : 8 comp
                   o  Elise:~/ELISE> ; sz [166,96] bpp : 8 compr : 1
            */
            init_tile
            (
                new Tile_F2d (new BMP_Rle_8_PFOB(bmpf)),
                0,
                1,
                true
            );
        }
        break;

        case Bmp_Im::rle_4bits :
        {
             elise_internal_error("do not handle bmp 4-bit rle",__FILE__,__LINE__);
        }
        break;
    }
}



/*********************************************************************/
/*                                                                   */
/*                  Bmp_Im_Not_Comp                                  */
/*                                                                   */
/*********************************************************************/


class Bmp_Im_Not_Comp : public Fonc_Num_Not_Comp
{

    public :
       Bmp_Im_Not_Comp(Bmp_Im,bool,REAL);

    private :

       Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);
       Bmp_Im            _bm_im;
       bool              _with_def_value;
       REAL              _def_value;


      virtual bool  integral_fonc (bool) const
      {return true;}

       virtual INT dimf_out() const
       {
               return _bm_im.dbmp()->_nb_chan;
       }

       void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
};

Bmp_Im_Not_Comp::Bmp_Im_Not_Comp(Bmp_Im bm_im,bool wdf,REAL df) :
     _bm_im             (bm_im),
     _with_def_value    (wdf),
     _def_value         (df)
{
}

Fonc_Num_Computed * Bmp_Im_Not_Comp::compute
                      (const Arg_Fonc_Num_Comp & arg)
{
    Data_BMP_File * bmpf = _bm_im.dbmp();

    Std_Bitm_Fich_Im_2d  *  f2d =
                 new BMP_file_2d(arg.flux(),bmpf,true);


    return fonc_num_std_f2d
    (       arg,
            f2d,
            _with_def_value,
            _def_value
    );
}

/*********************************************************************/
/*                                                                   */
/*                  Bmp_Out_Not_Comp                                 */
/*                                                                   */
/*********************************************************************/

class Bmp_Out_Not_Comp : public Output_Not_Comp
{

    public :
       Bmp_Out_Not_Comp(Bmp_Im bm_im) : _bm_im (bm_im) {}


    private :
       Output_Computed * compute(const Arg_Output_Comp & arg)
       {
            Data_BMP_File * bmpf = _bm_im.dbmp();

            Std_Bitm_Fich_Im_2d  *  f2d =
                 new BMP_file_2d(arg.flux(),bmpf,false);


           return  out_std_f2d(arg,f2d);
       }

       Bmp_Im            _bm_im;
};

/*********************************************************************/
/*                                                                   */
/*                  Bmp_Im                                           */
/*                                                                   */
/*********************************************************************/

Bmp_Im::Bmp_Im(const char * name)  :
    PRC0(new Data_BMP_File(name))
{
}

Fonc_Num Bmp_Im::in(INT def)
{
    return new Bmp_Im_Not_Comp(*this,true,def);
}

INT Bmp_Im::bpp() const
{
   return dbmp()->_bits_pp;
}

Bmp_Im::mode_compr Bmp_Im::compr()
{
   return dbmp()->_compr;
}

Pt2di Bmp_Im::sz()
{
   return dbmp()->sz();
}

Disc_Pal Bmp_Im::pal() const
{
    ASSERT_TJS_USER
    (
           bpp() != 24,
           "No palette for bmp, 24 bits image"
    );
    return dbmp()->_pal;
}


Output Bmp_Im::out()
{

     Tjs_El_User.ElAssert
     (
          dbmp()->_compr == Bmp_Im::no_compr,
          EEM0 << "Cannot write in compressed BMP file for now\n"
               << "|  name of file : " << dbmp()->_name
     );
    return new Bmp_Out_Not_Comp(*this);
}

/*********************************************************************/
/*                                                                   */
/*                  Data_BMP_File                                    */
/*                                                                   */
/*********************************************************************/

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
