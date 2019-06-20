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



/****************************************************************/
/****************************************************************/
/*****                                                       ****/
/*****               HuffmanTree                             ****/
/*****                                                       ****/
/****************************************************************/
/****************************************************************/
/*
template <class Type> class cHTNoVal;
template <> class cHTNoVal<INT>
{
    public :
         static INT NoVal() {return -0x7ffffff;}
};
*/






// class HuffmanTree : public cTplHuffmanTree<INT>

template <class Type> class cTplHuffmanTree
{
      friend class HuffmanCodec;
      friend class Huff_Ccitt_1D_Codec;
      friend class Huff_Ccitt_2D_T6;
      friend class Huffman_FOB_Codec;
      friend class MPD_CCIT_T6;


      public :

        cTplHuffmanTree();


      private :

         cTplHuffmanTree * _son[2];

         Type          _val;
         INT           _code;
         INT           _code_lsb;
         INT           _nbb;
         bool          mHasVal;

		 enum
		 {
			_no_val = -0x7ffffff
		 };


         cTplHuffmanTree * add_new_code
                       (
                         const char * name,
                         Type val,
                         INT code =0,
                         INT nbb = 0
                       );

        void show(INT nb);
};


/*
class HuffmanTree : public cTplHuffmanTree<INT>
{
    public : HuffmanTree(-0x7ffffff
};
*/

template <class Type> void cTplHuffmanTree<Type>::show(INT nb)
{
     static char buf[1000];

     if (mHasVal)
     {
         buf[nb] = 0;

         cout << _val   << " ["
              <<  buf   << "] "
              <<  _code << " "
              << _nbb   << "\n";
     }

     for (INT i=0 ; i<2 ; i++)
         if (_son[i])
         {
               buf[nb] = '0' +i;
               _son[i]->show(nb+1);
         }
}
template <class Type> cTplHuffmanTree<Type>::cTplHuffmanTree()
{
   _son[0] = _son[1] = 0;
   mHasVal = false;
}



template <class Type> cTplHuffmanTree<Type> * cTplHuffmanTree<Type>::add_new_code
              (
                   const char * name,
                   Type val,
                   INT code,
                   INT nbb
              )
{
     El_Internal.ElAssert
     (
         ! mHasVal,
         EEM0 << "Conflictual values in Huffman tree creation"
     );

     if (! *name)
     {
         El_Internal.ElAssert
         (
             (! _son[0]) && (! _son[1]),
             EEM0 << "Conflictual values in Huffman tree creation"
         );
         _val  =  val;
         _code = code;
         _nbb  =  nbb;
         _code_lsb = inv_bits_order(code,nbb);
         mHasVal = true;
         return this;
     }

     INT ind = *name - '0';

     if (! _son[ind])
        _son[ind] = NEW_FOR_EVER(cTplHuffmanTree<Type>,());

    return _son[ind]->add_new_code(name+1,val,2*code+ind,nbb+1);
}



    // Pour compat avec anciens


/****************************************************************/
/****************************************************************/
/*****                                                       ****/
/*****               HuffmanCodec                            ****/
/*****                                                       ****/
/****************************************************************/
/****************************************************************/


class HuffmanCodec
{
       public :

          friend class Huffman_FOB_Codec;
          friend class Huff_Ccitt_1D_Codec;
          friend class Huff_Ccitt_2D_T6;
          friend class MPD_CCIT_T6;

          HuffmanCodec(INT nb);

          static const HuffmanCodec * TiffCitt3Black();
          static const HuffmanCodec * TiffCitt3White();
          static const HuffmanCodec * TiffCitt4Vert();
          static const HuffmanCodec * TiffCitt4UComp();
          static const HuffmanCodec * TiffMPDT6();


          const cTplHuffmanTree<INT> *   leaf_of_val(INT val) const;

       private :

          inline const cTplHuffmanTree<INT> * code_ccit_1d(INT x) const
          {
                return (x<64)           ?
                       _tab[x]          :
                       _tab[63+(x/64)]  ;
          }

          cTplHuffmanTree<INT> *   _tree;
          cTplHuffmanTree<INT> **  _tab;
          INT             _nb_tot;
          INT             _nb_cur;

          void init_file(const char * name);

          void verif_init();

          static HuffmanCodec * _Tiff_Citt3_Black;
          static HuffmanCodec * _Tiff_Citt3_White;
          static HuffmanCodec * _TiffCitt4_Vert;
          static HuffmanCodec * _TiffCitt4_UComp;
          static HuffmanCodec * _Tiff_MPD_T6;

          static    HuffmanCodec * init_from_files
                    (
                        HuffmanCodec * v0,
                        const char **    files,
                        INT        nb_files,
                        INT        nb_leaves
                    );

         static const char *  HUFMAN_CCITT_MOD;
         static const char *  HUFMAN_CCIT_UCOMP;
         static const char *  HUFMAN_MPD;
         static const char *  HUFMAN_black;
         static const char *  HUFMAN_makeup;
         static const char *  HUFMAN_white;
};

const cTplHuffmanTree<INT>  * HuffmanCodec::leaf_of_val(INT val) const
{
      for (INT i=0; i<_nb_tot ; i++)
          if (_tab[i]->_val == val)
             return _tab[i];

      El_Internal.ElAssert(0,EEM0<< "Bad call to  HuffmanCodec::leaf_of_val");

      return 0;
}



HuffmanCodec * HuffmanCodec::_Tiff_MPD_T6 = 0;
HuffmanCodec * HuffmanCodec::_Tiff_Citt3_Black = 0;
HuffmanCodec * HuffmanCodec::_Tiff_Citt3_White = 0;
HuffmanCodec * HuffmanCodec::_TiffCitt4_Vert   = 0;
HuffmanCodec * HuffmanCodec::_TiffCitt4_UComp   = 0;


HuffmanCodec::HuffmanCodec(INT nb)
{
     _nb_tot = nb;
     _nb_cur = 0;
     _tree   = NEW_FOR_EVER(cTplHuffmanTree<INT> ,());
     _tab    = NEW_TAB_FOR_EVER(cTplHuffmanTree<INT>  *,_nb_tot);
}



void HuffmanCodec::init_file(const char * name_initial)
{
    char * name = NEW_TAB(1+strlen( name_initial),char);
    memcpy(name, name_initial,1+strlen( name_initial));

    char * str = name;
    while(str[0] != 'X')
    {
            INT val = strtol(str,&str,10);
            while(*str == ' ') str++;
            INT nb = 0;
            while((str[nb]== '0') || (str[nb]=='1')) nb++;
            str[nb] = 0;

            cTplHuffmanTree<INT>  * ht = _tree->add_new_code(str,val);
            El_Internal.ElAssert
            (
                _nb_cur<_nb_tot,
                EEM0 << "bad dim in HuffmanCodec"
            );
            _tab[_nb_cur++] = ht;
            str += nb+1;
    }
    DELETE_TAB(name);
}




void HuffmanCodec::verif_init()
{
     El_Internal.ElAssert
     (
           _nb_cur==_nb_tot,
           EEM0 << "bad dim in HuffmanCodec"
     );
}

HuffmanCodec * HuffmanCodec::init_from_files
                    (
                        HuffmanCodec * v0,
                        const char **    files,
                        INT        nb_files,
                        INT        nb_leaves
                    )
{
    if (! v0)
    {
        v0 = NEW_FOR_EVER(HuffmanCodec,(nb_leaves));
        for (INT k=0 ; k<nb_files ; k++)
            v0->init_file(files[k]);
        v0->verif_init();
    }
    return v0;
}

const HuffmanCodec * HuffmanCodec::TiffCitt3Black()
{
   static const char * name[2] = {HUFMAN_black,HUFMAN_makeup};

   return   _Tiff_Citt3_Black
          = init_from_files(_Tiff_Citt3_Black,name,2,105);
}

const HuffmanCodec * HuffmanCodec::TiffCitt3White()
{
   static const char * name[2] = {HUFMAN_white,HUFMAN_makeup};

   return   _Tiff_Citt3_White
          = init_from_files(_Tiff_Citt3_White,name,2,105);
}

const HuffmanCodec * HuffmanCodec::TiffCitt4Vert()
{
   static const char * name[1] = {HUFMAN_CCITT_MOD};
   return    _TiffCitt4_Vert
          =  init_from_files(_TiffCitt4_Vert,name,1,12);

}

const HuffmanCodec * HuffmanCodec::TiffCitt4UComp()
{
   static const char * name[1] = {HUFMAN_CCIT_UCOMP};
   return    _TiffCitt4_UComp
          =  init_from_files(_TiffCitt4_UComp,name,1,11);

}

const HuffmanCodec * HuffmanCodec::TiffMPDT6()
{
   static const char * name[1] = {HUFMAN_MPD};
   return    _Tiff_MPD_T6
          =  init_from_files(_Tiff_MPD_T6,name,1,6);

}


/****************************************************************/
/****************************************************************/
/*****                                                       ****/
/*****             Huffman_FOB_Codec                         ****/
/*****                                                       ****/
/****************************************************************/
/****************************************************************/

void Huffman_FOB_Codec::show()
{
     cout << "[" <<  (_read?_flx->tell() : _flx_varli->tell())
          << "," << (_read?_kth:_flx_varli->kth()) << "]";
}

void Huffman_FOB_Codec::reset()
{
     if (_read)
        _kth = 8;
     else
        _flx_varli->reset();
}

Huffman_FOB_Codec::Huffman_FOB_Codec
(
        Packed_Flux_Of_Byte*   flx,
        bool                   read,
        bool                   msbf,
        bool                   flush_flx
)   :
    _read      (read),
    _flx       (flx),
    _msbf      (msbf),
    _flush_flx (flush_flx)
{
    if (! read)
    {

       _flx_varli = Flux_OutVarLI::new_flx
                    (
                        new UnPacked_FOB(flx,flush_flx),
                        msbf,
                        true
                    );
    }
    reset();
}


Huffman_FOB_Codec::~Huffman_FOB_Codec()
{
  if (_read)
  {
      if (_flush_flx)
         delete _flx;
  }
  else
  {
     delete _flx_varli;
  }
}

INT Huffman_FOB_Codec::getbit()
{
    if (_kth == 8)
    {
       _flx->Read(&_v_buf,1);
       _kth = 0;
    }

    return _msbf                          ?
           kth_bit(_v_buf,7-_kth++)      :
           kth_bit(_v_buf,_kth++)         ;
}

INT Huffman_FOB_Codec::geti(INT nbb)
{

    INT res = 0;

    if (_msbf)
        for (INT i =0; i<nbb ; i++)
            res = res*2+getbit();
    else
        for (INT i =0; i<nbb ; i++)
            res = res+(getbit()<<i);
   return res;
}

const cTplHuffmanTree<INT>  * Huffman_FOB_Codec::get(const cTplHuffmanTree<INT>  * h)
{

    while (! h->mHasVal) // (h->_val == cHTNoVal<INT>::NoVal())
    {
          INT b = getbit();

          h = h->_son[b];
          El_Internal.ElAssert
          (
              h != 0,
              EEM0 << "impossible link in Huffman Tree"
          );
    }
    return h;
}

void Huffman_FOB_Codec::put(const cTplHuffmanTree<INT>  * ht)
{
     if (_msbf)
         _flx_varli->puti(ht->_code,ht->_nbb);
     else
         _flx_varli->puti(ht->_code_lsb,ht->_nbb);
}

void Huffman_FOB_Codec::put(INT val,INT nbb)
{
    _flx_varli->puti(val,nbb);
}


/****************************************************************/
/****************************************************************/
/*****                                                       ****/
/*****             Huff_Ccitt_1D_Codec                       ****/
/*****                                                       ****/
/****************************************************************/
/****************************************************************/

Huff_Ccitt_1D_Codec::Huff_Ccitt_1D_Codec
(
        Packed_Flux_Of_Byte*   flx,
        bool                   read,
        bool                   msbf,
        bool                   flush_flx
)  :
   Huffman_FOB_Codec(flx,read,msbf,flush_flx),
   _hw              (HuffmanCodec::TiffCitt3White()),
   _hb              (HuffmanCodec::TiffCitt3Black())
{
}


INT  Huff_Ccitt_1D_Codec::get_length(INT coul)
{
    const cTplHuffmanTree<INT>  *  h = coul ? _hb->_tree : _hw->_tree;
    int res =0;

    while(1)
    {
       INT i= get(h)->_val;
       El_Internal.ElAssert
       (
          (i>=0) && (i<=2560),
          EEM0 << "Bad value in ccitt run length, got : " << i
       );
       res += i;
       if (i < 64)
          return res;
    }
}


void Huff_Ccitt_1D_Codec::read(U_INT1 * res,INT nb_tot)
{
     INT coul = 0;
     INT nb = 0;

     while(nb != nb_tot)
     {
        INT dnb =  get_length(coul);
        El_Internal.ElAssert
        (
             (nb+dnb) <= nb_tot,
             EEM0 << "bad check sum in cciit3-1D uncomp"
        );
        if (res)
            memset(res+nb,coul,dnb);
        nb += dnb;
       coul = !coul;
     }
     reset();
}


void Huff_Ccitt_1D_Codec::put_length_partial(INT l,const HuffmanCodec *hc)
{
     put(hc->code_ccit_1d(l));
}

void Huff_Ccitt_1D_Codec::put_length(INT l,INT coul)
{
     const HuffmanCodec *  hc = coul?_hb:_hw;

     INT nb_2560 = l/2560; // number of 2560 lenght to put

     for (int x=0 ; x<nb_2560 ; x++)
         put_length_partial(2560,hc);

     INT reste = l-nb_2560*2560;      // rest of lenth to put
     INT nb_arr_64 = (reste/64)*64;  // part of rest rounded to 64

     if (nb_arr_64)
        put_length_partial(nb_arr_64,hc);

     put_length_partial(reste-nb_arr_64,hc);
}


void Huff_Ccitt_1D_Codec::write(const U_INT1 * im,INT nb_tot)
{
     INT coul = 0;
     INT x0 = 0;

     while(x0 != nb_tot)
     {
        INT x1;
        for
        (
              x1=x0;
              x1<nb_tot && (im[x1] == coul);
              x1++
        )
              ;
        put_length(x1-x0,coul);
        coul = !coul;
        x0 = x1;
     }
     reset();
}

/****************************************************************/
/****************************************************************/
/*****                                                       ****/
/*****             Huff_Ccitt_2D_T6                          ****/
/*****                                                       ****/
/****************************************************************/
/****************************************************************/




Huff_Ccitt_2D_T6::~Huff_Ccitt_2D_T6()
{

     DELETE_VECTOR(_cur,-1);
     DELETE_VECTOR(_prec,-1);
}

void Huff_Ccitt_2D_T6::new_block(INT tx)
{
    if (_read)
       reset();
    _tx = tx;
    memset(_cur-1,0,tx+2);
    _eofb = false;
}

void Huff_Ccitt_2D_T6::end_block(bool phys_end_block)
{
     if (_read)
     {
        if ((! _eofb) && phys_end_block)
        {
            INT val  = get(_hvert->_tree)->_val;
            El_Internal.ElAssert
            (
                 val == eofb,
                 EEM0 << "Not found end of block in citt 2D T6, got :" << val
            );
        }
     }
     else
     {
        put(_ht_eofb);
     }
     reset();
}



Huff_Ccitt_2D_T6::Huff_Ccitt_2D_T6
(
               Packed_Flux_Of_Byte*    flx,
               bool                    read,
               bool                    msbf,
               bool                    flush_flx,
               INT                     sz_buf
)   :
        Huff_Ccitt_1D_Codec(flx,read,msbf,flush_flx)
{
       _hvert =  HuffmanCodec::TiffCitt4Vert();
       _huc   =  HuffmanCodec::TiffCitt4UComp();
       _prec  = NEW_VECTEUR(-1,sz_buf+2,U_INT1);
       _cur   = NEW_VECTEUR(-1,sz_buf+2,U_INT1);


        _ht_pass = _hvert->leaf_of_val(pass_mod);
        _ht_horz = _hvert->leaf_of_val(hor_mod);
        _ht_eofb = _hvert->leaf_of_val(eofb);
}




/*

   La doc du CCITT-T4 est imprecise quand a la position de b1:
      -a) est ce l'element changeant de couleur opposee strictement
      a droite de a0 ?
      -b) est ce l'element changeant de couleur opposee a droite
      ou au-dessus de a0;

  En fait l'ambiguite  est levee si on regarde l'exemple 1 de la figure
  12T/4, page 16 du DDR.

 C'est la reponse a) qui est la bonne.
*/




void Huff_Ccitt_2D_T6::calc_b1()
{
     for
     (
              _b1 = _a0+1;
              (_b1 < _tx)
            && ((_prec[_b1]==_coul)||(_prec[_b1]==_prec[_b1-1]));
            _b1++
     );
}

void Huff_Ccitt_2D_T6::calc_b2()
{
     for
     (
         _b2 = _b1;
         (_b2<_tx)&&(_prec[_b2]!=_coul) ;
         _b2++
     );
}

void Huff_Ccitt_2D_T6::calc_a1()
{
     for
     (
         _a1 = _a0+1;
         (_a1<_tx)&&(_cur[_a1]==_coul) ;
         _a1++
     );
}

void Huff_Ccitt_2D_T6::calc_a2()
{
     for
     (
         _a2 = _a1;
         (_a2<_tx)&&(_cur[_a2]!=_coul) ;
         _a2++
     );
}





void Huff_Ccitt_2D_T6::uncomp_line()
{
     El_Internal.ElAssert
     (
         0,
         EEM0 << "Unexpected CCITT uncompressed mode \n"
     );
/*

     if (_a0 == -1)
        _a0 = 0;

     // "eat" the 3 bits xxx of Table 5/T.4 of DDR
     for (int i=0; i <3 ; i++)
         getbit();

     INT v = 0;
     while (v < 100)
     {
           v = get(_huc->_tree)->_val;
           INT l = (v%100);
           memset(_cur+_a0,0,l);
           _a0 += l;
           if (v<5)
              _cur[_a0++] = 1;
           El_Internal.ElAssert
           (
               (_a0 <= _tx),
               EEM0 << "Bad check sum in CCITT uncompressed"
           );
     }
     cout << "Ucomp mode \n";
     getchar();

     _coul = getbit();
*/
}


void Huff_Ccitt_2D_T6::read(U_INT1 * res)
{

    ElSwap(_cur,_prec);
    _a0 = -1;
    _coul = 0;
    _prec[_a0] = 0;


    if (_eofb)
    {
       memset(_cur,0,_tx);
    }
    else
    {
       while(_a0 < _tx)
       {

            INT val  = get(_hvert->_tree)->_val;
            switch(val)
            {

                 case ucomp_mod :
                 {
                      uncomp_line();
                 }
                 break;

                 case eofb:
                 {
                    memset(_cur+_a0,0,_tx-_a0);
                    _a0 = _tx;
                    _eofb = true;
                 }
                 break;

                 case pass_mod:
                 {
                      calc_b1();
                      calc_b2();
                      memset(_cur+_a0,_coul,_b2-_a0);
                      _a0 = _b2;
                 }
                 break;

                 case hor_mod:
                 {
                      INT l0 = get_length(_coul);
                   /*
                      voir 4.2.1.3.4 du DDR.
                      Pour la premiere "run", en mode horizontal, on code a0a1-1.
                   */

                      if (_a0==-1) l0++;
                      INT l1 = get_length(! _coul);
                      El_Internal.ElAssert
                      (
                         (_a0+l0+l1) <= _tx,
                         EEM0 << "Bad check sum in ccit 2D T6 compression Horz" << _a0
                      );
                      memset(_cur+_a0,_coul,l0);
                      _a0 += l0;
                      memset(_cur+_a0,!_coul,l1);
                      _a0 += l1;
                 }
                 break;

                 default:
                 {
                       El_Internal.ElAssert
                      (
                         (val>=-3) && (val<=3),
                         EEM0 << "Unexpected code in ccit 2D T6, got " << val
                      );
                      calc_b1();
                      _a1 = _b1+val;
                      El_Internal.ElAssert
                      (
                         (_a1) <= _tx,
                         EEM0 << "Bad check sum in ccit 2D T6 compression vert" << _a0
                      );
                      memset(_cur+_a0,_coul,_a1-_a0);
                      _a0 = _a1;
                      _coul = !_coul;
                 }
                 break;
            }
       }
    }
    if (res) memcpy(res,_cur,_tx);
}


void Huff_Ccitt_2D_T6::write(const U_INT1 * vals)
{


    ElSwap(_cur,_prec);
    _a0 = -1;
    _coul = 0;
    _prec[_a0] = 0;
    _cur[_a0] = 0;

    memcpy(_cur,vals,_tx);

    while (_a0 != _tx)
    {
         calc_a1();
         calc_b1();
         calc_b2();
         if (_b2 < _a1)
         {
             put(_ht_pass);
             _a0 = _b2;
         }
         else
         {
             INT dif = _a1-_b1;
             if ((dif<=3) && (dif >=-3))
             {
                put(_hvert->_tab[dif+3]);
                _coul = ! _coul;
                _a0   = _a1;
             }
             else
             {
                calc_a2();
                put(_ht_horz);
                put_length(_a1-ElMax(_a0,0),_coul);
                put_length(_a2-_a1,!_coul);
                _a0 = _a2;
             }
         }
    }
}

/****************************************************************/
/****************************************************************/
/*****                                                       ****/
/*****             MPD_CCIT_T6                              ****/
/*****                                                       ****/
/****************************************************************/
/****************************************************************/


MPD_CCIT_T6::MPD_CCIT_T6
(
               Packed_Flux_Of_Byte*    flx,
               bool                    read,
               bool                    msbf,
               bool                    flush_flx,
               INT                     sz_buf,
               INT                     nbb
)   :
    Huff_Ccitt_2D_T6(flx,read,msbf,flush_flx,sz_buf)
{
    _bin  = NEW_VECTEUR(-1,sz_buf+2,U_INT1);
    _hmpd = HuffmanCodec::TiffMPDT6();
    _ht_huf_bl =  _hmpd->leaf_of_val(huf_black);
    _nbb = nbb;
    _vmax = (1<<nbb) -1;
    _line = 0;
}

MPD_CCIT_T6::~MPD_CCIT_T6()
{
     DELETE_VECTOR(_bin,-1);
}

void MPD_CCIT_T6::put_length_gray(INT l)
{
     if (l<=max_l_gr)
     {
        put(_hmpd->_tab[l]);
     }
     else
     {
         put(_ht_huf_bl);
         put_length(l-max_l_gr,0);
     }
}

INT  MPD_CCIT_T6::get_length_gray()
{
     const cTplHuffmanTree<INT>  *  ht = get(_hmpd->_tree);
     INT res = 0;
     switch (ht->_val)
     {
        case huf_black :
             res = max_l_gr + get_length(0);
        break;

        default :
             res = ht->_val;
        break;
     }
     return res;
}

INT  MPD_CCIT_T6::get_plage_gray(INT a0,bool last)
{
     INT l =  get_length_gray();
     INT a1 = last ? (a0-l) : (a0+l);
     if (last)
        ElSwap(a0,a1);

     if (_vals)
         for (INT a=a0 ; a<a1 ; a++)
             _vals[a] = geti(_nbb);
     else
         for (INT a=a0 ; a<a1 ; a++)
             geti(_nbb);

     return a1;
}

void MPD_CCIT_T6::put_plage_gray(INT a1,INT a2)
{
     put_length_gray(a2-a1);
     for (INT a=a1; a<a2; a++)
         put(_vals[a],_nbb);
}

INT MPD_CCIT_T6::end_pl_gray(INT a)
{
    while(_bin[a] && _vals[a]) a++;
    return a;
}

INT MPD_CCIT_T6::end_pl_pure_black(INT a)
{
    while((a<_tx) && (! _vals[a])) a++;
    return a;
}
INT MPD_CCIT_T6::end_pl_white(INT a)
{
    while(! _bin[a]) a++;
    return a;
}

void MPD_CCIT_T6::read(U_INT1 * res)
{
     _line++;
     _vals = res;
     Huff_Ccitt_2D_T6::read(_bin);

     if (_vals)
     {
         for (INT i=0 ; i<_tx ; i++)
              _vals[i] = (_bin[i] ? 0 : _vmax);
      }
     _bin[-1] = 0;
     _bin[_tx] = 0;
     _bin[_tx+1] = 1;

     INT a0 = -1;
     for(;;)
     {
         a0 = end_pl_white(a0);
         if (a0 >= _tx)
         {
             return;
         }


         while(getbit())
         {
              a0 = get_plage_gray(a0,false);
              a0 = a0 +get_length(0);
         }
         get_plage_gray(a0,false);
         while (_bin[a0]) a0++;
         get_plage_gray(a0,true);
     }
}

void MPD_CCIT_T6::write(const U_INT1 * vals)
{
     _line++;
     _vals = const_cast<U_INT1 *> (vals);

     _bin[-1] = 0;
     _bin[_tx] = 0;
     _bin[_tx+1] = 1;

     for (INT i=0; i<_tx ; i++)
        _bin[i] = (vals[i]!=_vmax) ;

     Huff_Ccitt_2D_T6::write(_bin);

     INT a0 = -1;
     for(;;)
     {
         a0 = end_pl_white(a0);

         if (a0 >= _tx)
         {
             return;
         }

         INT a1 = end_pl_gray(a0);
         INT a2 = end_pl_pure_black(a1);
         INT a3 = end_pl_gray(a2);

         while (_bin[a3])
         {
              put(1,1);
              put_plage_gray(a0,a1);
              put_length(a2-a1,0);
              a0 = a2;
              a1 = a3;
              a2 = end_pl_pure_black(a1);
              a3 = end_pl_gray(a2);
         }
         put(0,1);
         put_plage_gray(a0,a1);
         put_plage_gray(a2,a3);
         a0 = a3;
     }
}


const char * HuffmanCodec::HUFMAN_CCITT_MOD =
       "-3      0000010 "
       "-2      000010 "
       "-1      010 "
       "0       1 "
       "1       011 "
       "2       000011 "
       "3       0000011 "
       "100     0001 "
       "200     001 "
       "-2000   000000000001000000000001 "
       "-3000   000000001 "
       "-3000   0000001 X";


const char * HuffmanCodec::HUFMAN_CCIT_UCOMP =
       "0       1 "
       "1       01 "
       "2       001 "
       "3       0001 "
       "4       00001 "
       "5       000001 "
       "100     0000001 "
       "101     00000001 "
       "102     000000001 "
       "103     0000000001 "
       "104     00000000001 X";


const char * HuffmanCodec::HUFMAN_MPD =
     "0      1100 "
     "1      0 "
     "2      10  "
     "3      1110 "
     "4      1111 "
     "888    1101 X";


const char * HuffmanCodec::HUFMAN_black =
       "0    0000110111 "
       "1    010 "
       "2    11 "
       "3    10 "
       "4    011 "
       "5    0011 "
       "6    0010 "
       "7    00011 "
       "8    000101 "
       "9    000100 "
       "10   0000100 "
       "11   0000101 "
       "12   0000111 "
       "13   00000100 "
       "14   00000111 "
       "15   000011000 "
       "16   0000010111 "
       "17   0000011000 "
       "18   0000001000 "
       "19   00001100111 "
       "20   00001101000 "
       "21   00001101100 "
       "22   00000110111 "
       "23   00000101000 "
       "24   00000010111 "
       "25   00000011000 "
       "26   000011001010 "
       "27   000011001011 "
       "28   000011001100 "
       "29   000011001101 "
       "30   000001101000 "
       "31   000001101001 "
       "32   000001101010 "
       "33   000001101011 "
       "34   000011010010 "
       "35   000011010011 "
       "36   000011010100 "
       "37   000011010101 "
       "38   000011010110 "
       "39   000011010111 "
       "40   000001101100 "
       "41   000001101101 "
       "42   000011011010 "
       "43   000011011011 "
       "44   000001010100 "
       "45   000001010101 "
       "46   000001010110 "
       "47   000001010111 "
       "48   000001100100 "
       "49   000001100101 "
       "50   000001010010 "
       "51   000001010011 "
       "52   000000100100 "
       "53   000000110111 "
       "54   000000111000 "
       "55   000000100111 "
       "56   000000101000 "
       "57   000001011000 "
       "58   000001011001 "
       "59   000000101011 "
       "60   000000101100 "
       "61   000001011010 "
       "62   000001100110 "
       "63   000001100111 "
       "64   0000001111 "
       "128  000011001000 "
       "192  000011001001 "
       "256  000001011011 "
       "320  000000110011 "
       "384  000000110100 "
       "448  000000110101 "
       "512  0000001101100 "
       "576  0000001101101 "
       "640  0000001001010 "
       "704  0000001001011 "
       "768  0000001001100 "
       "832  0000001001101 "
       "896  0000001110010 "
       "960  0000001110011 "
       "1024 0000001110100 "
       "1088 0000001110101 "
       "1152 0000001110110 "
       "1216 0000001110111 "
       "1280 0000001010010 "
       "1344 0000001010011 "
       "1408 0000001010100 "
       "1472 0000001010101 "
       "1536 0000001011010 "
       "1600 0000001011011 "
       "1664 0000001100100 "
       "1728 0000001100101 X";

const char * HuffmanCodec::HUFMAN_makeup =
      "1792   00000001000 "
      "1856   00000001100 "
      "1920   00000001101 "
      "1984   000000010010 "
      "2048   000000010011 "
      "2112   000000010100 "
      "2176   000000010101 "
      "2240   000000010110 "
      "2304   000000010111 "
      "2368   000000011100 "
      "2432   000000011101 "
      "2496   000000011110 "
      "2560   000000011111 "
      "-1000  000000000001 X";


const char * HuffmanCodec::HUFMAN_white =
       "0      00110101 "
       "1      000111 "
       "2      0111 "
       "3      1000 "
       "4      1011 "
       "5      1100 "
       "6      1110 "
       "7      1111 "
       "8      10011 "
       "9      10100 "
       "10     00111 "
       "11     01000 "
       "12     001000 "
       "13     000011 "
       "14     110100 "
       "15     110101 "
       "16     101010 "
       "17     101011 "
       "18     0100111 "
       "19     0001100 "
       "20     0001000 "
       "21     0010111 "
       "22     0000011 "
       "23     0000100 "
       "24     0101000 "
       "25     0101011 "
       "26     0010011 "
       "27     0100100 "
       "28     0011000 "
       "29     00000010 "
       "30     00000011 "
       "31     00011010 "
       "32     00011011 "
       "33     00010010 "
       "34     00010011 "
       "35     00010100 "
       "36     00010101 "
       "37     00010110 "
       "38     00010111 "
       "39     00101000 "
       "40     00101001 "
       "41     00101010 "
       "42     00101011 "
       "43     00101100 "
       "44     00101101 "
       "45     00000100 "
       "46     00000101 "
       "47     00001010 "
       "48     00001011 "
       "49     01010010 "
       "50     01010011 "
       "51     01010100 "
       "52     01010101 "
       "53     00100100 "
       "54     00100101 "
       "55     01011000 "
       "56     01011001 "
       "57     01011010 "
       "58     01011011 "
       "59     01001010 "
       "60     01001011 "
       "61     00110010 "
       "62     00110011 "
       "63     00110100 "
       "64     11011 "
       "128    10010 "
       "192    010111 "
       "256    0110111 "
       "320    00110110 "
       "384    00110111 "
       "448    01100100 "
       "512    01100101 "
       "576    01101000 "
       "640    01100111 "
       "704    011001100 "
       "768    011001101 "
       "832    011010010 "
       "896    011010011 "
       "960    011010100 "
       "1024   011010101 "
       "1088   011010110 "
       "1152   011010111 "
       "1216   011011000 "
       "1280   011011001 "
       "1344   011011010 "
       "1408   011011011 "
       "1472   010011000 "
       "1536   010011001 "
       "1600   010011010 "
       "1664   011000 "
       "1728   010011011 X";









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
