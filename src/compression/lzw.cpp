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


/*
     The entry structure :

          * each time a new code is added as a prolongation of one value V
            of a old one O, it is sufficient to memoize O+V. This
            what memoes entry : val = V and next = O;
            so in fact the entry struct is +or- a linked list
           (in fact a linked tree)


      Une remarque sur le codage LZW qui me tracasse a chaque
      fois. Considerons la sequence, non compr, suivante :

           1 2 0 0 0 3 ...

      Avec une table initialisee sur 8 bit et en l'absence de sequence
      speciale, on aura le codage suivant

                      1 ->  out <<        ;                  ;
        tmp = "1"     2 ->  out << $1     ; t[256] = "12"    ;
        tmp = "2"     0 ->  out << $2     ; t[257] = "20"    ;
        tmp = "0"     0 ->  out << $0     ; t[258] = "00"    ;
        tmp = "0"     0 ->                ;                  ;
        tmp = "00"    3 ->  out << $258   ; t[259] = "003"   ;
        tmp = "3"   .........



        au decodage, on aura donc la sequence suivante :

        $1 $2 $0 $128  .....


               $1   -> out << 1    ;
               $2   -> out << 2    ;  t[256] = "12"  // prec + prem de "2"
               $0   -> out << 0    ;  t[257] = "20"  // prec + prem de "0"
               $258 -> out << 00   ;  t[258] = "00"  // ????
               $3   -> out << 3    ;  t[259] = "003" // prec + prem de "3"

       De maniere generale, le principe est simple pour alimenter
    la table : on rajoute la chaine du precedent code lu +
    le premier caractere du code courant (on recree ainsi la chaine
    inconnue lors du processus de codage). Reste le cas marque ??? :

           * on lit a un code, que l'on ne connait pas
             (c'est la premiere entree libre la table) et que
             l'on doit utiliser pour alimenter la table et sortir
             un output;

           * en fait il suffit d'ecrire formellement l'equation
             generale :
                   -  t[258] = last + prem de t[258]

           * et on voit donc que
                       prem de t[258] = last de t[258]

           * soit, dans le cas particulier ou l'on lit un code
             egal a la premiere entree libre de la table :



              /---------------------------------------------------\
             //---------------------------------------------------\\
             ||  REMARQUE 1                                       ||
             ||                                                   ||
             ||   nouv code = code prec + prem de code prec       ||
             \\---------------------------------------------------//
              \---------------------------------------------------/

*/

#define LZW_TYPE U_INT1


class LZW_decoder : public Mcheck
{
      friend void LZW_GIF_TIF_inst_reset(class LZW_decoder * dec);

      public :

             //  To simplify the inteface (to have single call to get code)
             //  we define this class

             class  wcodes
             {
                  public :
                     INT   c[4];
                     INT   nbc;
                     INT   nbb[4];
             };


                // return the current number of bit needed to code
                // the expected code

           inline INT nb_bit_cur() const {return _nb_bit_cur;}


               // given a char c, put in res the code to write

           void write_codes(LZW_TYPE c,wcodes & res,bool end);

               // do "everything" : ie update the table and
               // return the "string" associated to the code
               // eventually : return 0 + set nb to 0 (clear code)

           const LZW_TYPE * new_code (int & nb,INT code);


               // On all the protocol I know, the end code is
               // redundant in the meaning where you can (easily)
               // know when it is to arrive (equal you have received
                // the excpected number of values)
               // With these protocol
               // never use new_code(..) with a end code (it woukd make a
               // fatal error) but eventually verifie that when you think
               // you got it, it is OK by this method.

           bool  is_it_end_code(INT code);

           inline INT  nb_val_max(){return   _nb_val_max;}

    // use a pointer to function for instance reseting because
    // need a virtual call in constructor :

           typedef void (* LZW_INST_RESET)(class LZW_decoder *);


            virtual ~LZW_decoder();
            void re_init();

      protected :


            LZW_decoder(LZW_INST_RESET,INT nb_bits_init,INT nb_bits_max,bool read);



            // manipulation of this may vary on different format
            // default init values are Tiff-Gif value:

            INT _clear_code;
            INT _end_code;
            INT _cur_sz;
            INT _val_clear_write;

            bool  _auto_clear_full; // does a clear occurs automatically
                                   //  when table is full

            INT _rab_code_augm;  // on tif =1 because the table is
                                 // augmented when there is still
                                 // one place free


      private :

            void clear();
            void augment_bits();

            // UTILITARY METHOD

            LZW_INST_RESET  _inst_reset;

            const LZW_TYPE * str_of_code (int & nb, INT code);

            // return code of current buffer
            INT code_of_cur_string();
            bool  is_code_of_cur_string (INT code);


            // LOCAL TYPE DEIFINITION

            typedef struct
            {
                LZW_TYPE   val;
                INT2     next;
                INT2     h_next;
                INT2     code;
            } entry;


            INT h_addr_of_cur_str();
            void h_add_cur_str(INT code);

            inline INT h_code_dif(INT c,INT old_h)
            {
                   return  (c + ((old_h +17) * 15))%_nb_hval;
            }

            inline void add_car_bw(U_INT1 c)
            {
                 _h_cur_addr  =  h_code_dif(c,_h_cur_addr);
                 _buf_write[_nb_in_bufw++] = c;
            }

            inline void set_empty_bw()
            {
                 _h_cur_addr = 0;
                 _nb_in_bufw  = 0;
            }

            inline void set_bw_1car(U_INT1 c)
            {
                 set_empty_bw();
                 add_car_bw(c);
            }


            // SIZE PARAMETRIZATION

            INT      _nb_bits_init;
            INT      _nb_val_init ; // = 1<<_nb_bits_init

            INT      _nb_bits_max;  // 12 with tif and gif
            int      _nb_val_max;   // 1 <<nb_bits_max
            INT      _nb_hval;      //  2* _nb_val_max  , why not ?


            // TABLE and results buffer

            entry        * _entries;  // table of  [_nb_val_max]
            INT2         * _htab;  // table of  [_nb_hval]

            LZW_TYPE     * _buf_res;  // [_nb_val_max+1]
            LZW_TYPE     * _buf_write;  // [_nb_val_max]




            // INTERNAL STATE:

            INT         _nb_bit_cur;
            INT         _next_code_augm;
            INT         _last_code;
            bool        _entr_full;
            LZW_TYPE    _last_prem;
            bool        _read;
            INT         _nb_in_bufw;
            bool        _first_code;
            INT         _h_cur_addr;


/*
      _first_code :  to memoize the fisrt clear code to add at begining
      of tile (required by TIFF)

      _nb_in_bufw : number of char buffered in _buf_write
*/



            // SYMBOLIC CONSTANTES :

			enum
			{
				NO_NEXT = -1,	// more or less NULL pointer
				NO_LAST_CODE = -1,
				NO_CODE      = -1

			};
/*
            static const INT NO_NEXT = -1;          // more or less NULL pointer
            static const INT NO_LAST_CODE = -1;
            static const INT NO_CODE      = -1;
*/
};

// const INT LZW_decoder::NO_NEXT      = -1;
// const INT LZW_decoder::NO_LAST_CODE = -1;
// const INT LZW_decoder::NO_CODE      = -1;


/******************************************************************/
/*                                                                */
/*       LZW_GIF_decoder, LZW_TIF_decoder                         */
/*                                                                */
/******************************************************************/


void LZW_GIF_TIF_inst_reset(class LZW_decoder * dec)
{
   dec->_next_code_augm  =   2*dec->_nb_val_init ;    // GIF
   dec->_cur_sz          =   dec->_nb_val_init+2;
   dec->_nb_bit_cur      =   dec->_nb_bits_init + 1;
}

class LZW_GIF_decoder : public LZW_decoder
{

      public :

            LZW_GIF_decoder(INT nb_bits_init,bool read)  :
                    LZW_decoder(LZW_GIF_TIF_inst_reset,nb_bits_init,12,read)
            {
                    _rab_code_augm  = 0;
                    _auto_clear_full = false; // see Gif 89a preambule
            }
};

class LZW_TIF_decoder : public LZW_decoder
{

      public :

            LZW_TIF_decoder(bool read)    :
                    LZW_decoder(LZW_GIF_TIF_inst_reset,8,12,read)
            {
                    _rab_code_augm   = 1   ; // see doc tiff-6.0 page 61
                    _auto_clear_full = true;
            }
};


/******************************************************************/
/*                                                                */
/*       LZW_decoder                                              */
/*                                                                */
/******************************************************************/


LZW_decoder::LZW_decoder
(
        LZW_INST_RESET INST_RESET,
        INT nb_bits_init,
        INT nb_bits_max,
        bool read
)
{
   _inst_reset      = INST_RESET;
   _nb_bits_init    = nb_bits_init;
   _nb_val_init     = 1 << nb_bits_init;
   _nb_bits_max     = nb_bits_max;
   _nb_val_max      = 1 << nb_bits_max;
   _entries         = NEW_VECTEUR(0,nb_val_max(),entry);
   _buf_res         = NEW_VECTEUR(0,nb_val_max()+1,LZW_TYPE);
   _buf_write       =  read ? 0 : NEW_VECTEUR(0,nb_val_max(),LZW_TYPE);
   _nb_hval         =  nb_val_max() * 2;
   _htab            =  read ? 0 : NEW_VECTEUR(0,_nb_hval,INT2);

   _clear_code      = _nb_val_init;     // TIFF_GIF
   _end_code        = _nb_val_init +1;  // TIFF_GIF
   _val_clear_write = nb_val_max() -4;   // prudent

   _read            = read;

   for (INT i = 0 ; i < _nb_val_init ; i++)
   {
        _entries[i].val  = i;
        _entries[i].next = NO_NEXT;
   }

   for (int code = 0; code<_nb_val_max ; code++)
     _entries[code].code =  code;

   re_init();
}


LZW_decoder::~LZW_decoder()
{
     if (_buf_write)
         DELETE_VECTOR(_buf_write,0);

     if (_htab)
         DELETE_VECTOR(_htab,0);

     DELETE_VECTOR(_buf_res,0);
     DELETE_VECTOR(_entries,0);
}



//    Reset the decoder (for example when receiving a clear code
// or when the table is full with tiff). It will begin
// again to generate "_nb_bits_init +1" values.

void LZW_decoder::clear()
{
   _inst_reset(this);
   _last_code       = NO_LAST_CODE;
   _entr_full       = false;


   if (! _read)
   {
	   {
		 for (INT i=0 ; i<_nb_hval ; i++)
			_htab[i] = NO_NEXT;
	   }

	   {
			for (INT i = 0 ; i < _nb_val_init ; i++)
			{
				set_bw_1car(i);
				h_add_cur_str(i);
			}
	   }
   }

  // when table is cleared, the current buf is added to wcode,
  // so the buffer becomes empty

   set_empty_bw();
}


void LZW_decoder::re_init()
{
   _first_code      = true;
   clear();
}

/*
      Return the "string" associated to a code by
     parsing of linked list.
*/

const LZW_TYPE * LZW_decoder::str_of_code (int & nb, INT code)
{

   ASSERT_INTERNAL((code< _cur_sz), "incoherenve in LZW decoding");

   for (nb =0 ; code != NO_NEXT ; nb++,code = _entries[code].next)
   {
        ASSERT_INTERNAL((nb <_nb_val_max),"incoherenve in LZW decoding");
       _buf_res[_nb_val_max-1-nb] = _entries[code].val;

   }
   return _buf_res + _nb_val_max -nb;
}

bool  LZW_decoder::is_code_of_cur_string (INT code)
{
   INT nb;
   for
   (
         nb =_nb_in_bufw -1 ;
         (code != NO_NEXT) && (nb>=0) && ( _entries[code].val == _buf_write[nb]) ;
         nb--,code = _entries[code].next
   );
   return (nb == -1) &&  (code == NO_NEXT);
}



bool LZW_decoder::is_it_end_code(INT code)
{
     return (code == _end_code);
}

        //=====================================
        //    WRITE
        //=====================================

/*
     Version 0.0; hyper-lente, juste pour tester le reste.

     Pour l'instant on garde ca peut toujours servir en cas de doute.

INT LZW_decoder::code_of_cur_string()
{
    for (int i=0; i<_cur_sz; i++)
        if (
                 (i!=_clear_code)
             &&  (i!=_end_code)
           )
    {
       INT nb;
       const U_INT1 * str =  str_of_code(nb,i);
       if (nb == _nb_in_bufw)
       {
           INT k;
           for (k=0; (k<nb) && (str[k]==_buf_write[k]) ; k++);
           if (k==nb)
              return i;
       }
    }
   return NO_CODE;
}
*/

INT LZW_decoder::h_addr_of_cur_str()
{
    return _h_cur_addr;
}


INT LZW_decoder::code_of_cur_string()
{
    for (
          INT h  = _htab[_h_cur_addr];
              h !=  NO_NEXT;
              h  = _entries[h].h_next
        )
    {
       if (is_code_of_cur_string(_entries[h].code))
           return _entries[h].code;
    }
    return NO_CODE;
}

void LZW_decoder::h_add_cur_str(INT code)
{
     _entries[code].h_next = _htab[_h_cur_addr];
     _htab[_h_cur_addr] = code;
}

void LZW_decoder::write_codes(LZW_TYPE c,wcodes & res,bool end)
{
     res.nbc     = 0;

     // Tiff tiles must begin with clear-code and this is
     // no bad for GIF-files
     // It is a case where multiple code will not interfere
     // with bit augmentation because we are at value 258

     if(_first_code)
     {
        res.nbb[res.nbc] = _nb_bit_cur;
        res.c[res.nbc++] = _clear_code;
        _first_code = false;
     }

     if (!end)
     {
          // augment string
          add_car_bw(c);
          INT code = code_of_cur_string();

         // is string still in tabs, do nothing
         if (code!=NO_CODE)
         {
            _last_code = code;
            return;
         }
     }
     else
     {
          /* Do nothing
              _last_code has the good values
          */
     }

     // output : code of buffer string without c
     res.nbb[res.nbc] = _nb_bit_cur;
     res.c[res.nbc++] = _last_code;

     h_add_cur_str(_cur_sz);
     // reinitialize current string
     set_bw_1car(c);

     // add in table new string
     _entries[_cur_sz].val    = c;
     _entries[_cur_sz].next = _last_code;
     augment_bits();
     _cur_sz++;

     _last_code = c;

     if (end)
     {
        res.nbb[res.nbc] = _nb_bit_cur;
        res.c[res.nbc++] = _end_code;
        re_init();
        return;
     }

     // Again, it is a case where multiple code will not interfere
     // with bit augmentation because we are at value 4092
     if (_cur_sz == _val_clear_write)
     {
        res.nbb[res.nbc] = _nb_bit_cur;
        res.c[res.nbc++] = code_of_cur_string();
        res.nbb[res.nbc] = _nb_bit_cur;
        res.c[res.nbc++] = _clear_code;
        clear();
     }
}


        //=====================================
        //    READ
        //=====================================

const LZW_TYPE * LZW_decoder::new_code (int & nb,INT code)
{
  ASSERT_INTERNAL(code != _end_code ,"LZW : unexcpected end code");

  if (code == _clear_code)
  {
      clear();
      nb = 0;
      return (LZW_TYPE *) 0;
  }

  const LZW_TYPE * res;


  if (_last_code ==  NO_LAST_CODE)
  {
     _last_code = code;
     res = str_of_code(nb,code);
     _last_prem = res[0];
     return  res;
  }
  else
  {
       if (_entr_full)
       {
            ASSERT_INTERNAL(! _auto_clear_full,"incoherenve in LZW decoding");
            // In gif mode : table full, do not udpate table
            return str_of_code(nb,code);
       }
       // CASE OF Remark 1 : must augmente code before computing res
       if (code == _cur_sz)
       {
           _entries[_cur_sz].val    = _last_prem;
           _entries[_cur_sz].next = _last_code;
           _cur_sz++;
           res = str_of_code(nb,code);
       }
       // GENERAL case : must compute res before augmenting table
       else
       {
           res = str_of_code(nb,code);
           _entries[_cur_sz].val    = res[0];
           _entries[_cur_sz].next = _last_code;
           _cur_sz++;
       }
       augment_bits();
       _last_prem = res[0];
       _last_code = code;
  }

  return res;
}

void LZW_decoder::augment_bits()
{

  if (_cur_sz == _next_code_augm-_rab_code_augm ) // + (!_read))
  {
     _entr_full = ( _nb_bit_cur == _nb_bits_max);
     if (_entr_full)
     {
         if (_auto_clear_full)
           clear();
     }
     else
     {
         _nb_bit_cur ++;
         _next_code_augm *= 2;
     }
  }
}





/******************************************************************/
/*                                                                */
/*       Packed_LZW_Decompr_Flow                                  */
/*                                                                */
/******************************************************************/

tFileOffset Packed_LZW_Decompr_Flow::tell()
{
    return
           _read           ?
           _flxi->tell()   :
           _flxo->tell()   ;
}



void Packed_LZW_Decompr_Flow::init()
{
    _nb_buffered = 0;
    _deb_buffered = 0;
}


Packed_LZW_Decompr_Flow::Packed_LZW_Decompr_Flow
(
      Flux_Of_Byte *       flx_byte,
      bool                 read,
      bool                 msbf,
      LZW_Protocols::mode  mod,
      INT             nb_bit_init
)    :
     Packed_Flux_Of_Byte(1)
{
    init();

    _read = read;


    if (mod == LZW_Protocols::gif)
        _decoder = new LZW_GIF_decoder(nb_bit_init,read);
    else if (mod == LZW_Protocols::tif)
    {
        ASSERT_TJS_USER(nb_bit_init==8,"nb_bit_init must be 8 in TIFF-LZW");
        _decoder = new LZW_TIF_decoder(read);
    }
    else
        elise_internal_error("unknown LZW mode",__FILE__,__LINE__);

    _flxi = 0;
    _flxo = 0;
    _buf  = 0;
    if (read)
    {
        _flxi = Flux_Of_VarLI::new_flx(flx_byte,msbf,true);
        _buf = NEW_VECTEUR(0,_decoder->nb_val_max(),U_INT1);
    }
    else
    {
        _flxo = Flux_OutVarLI::new_flx(flx_byte,msbf,true);
    }
}


void Packed_LZW_Decompr_Flow::assert_end_code()
{
     INT code =_flxi->nexti(_decoder->nb_bit_cur());

     if (    _nb_buffered.BasicLLO()
          || (! _decoder->is_it_end_code(code))
        )
        elise_internal_error
        (
           "wrongly asserted end code in LZW decompress",
           __FILE__,
           __LINE__
        );
}

tFileOffset Packed_LZW_Decompr_Flow::Read(U_INT1 * res,tFileOffset nbo)
{
      int nb = nbo.CKK_IntBasicLLO();
      int sum_nb_added = ElMin(nb,_nb_buffered.CKK_IntBasicLLO());
      _nb_buffered -= sum_nb_added;
      if (res)
         memcpy(res,_buf+_deb_buffered.BasicLLO(),sum_nb_added);
      _deb_buffered += sum_nb_added;

      if (sum_nb_added == nb)
         return sum_nb_added;

      const U_INT1 * decoded  = 0; // warning init

      int      nb_decoded     = 0;
      int      nb_transfered  = 0;

      while (sum_nb_added < nb)
      {
             decoded = _decoder->new_code
                       (
                            nb_decoded,
                            _flxi->nexti(_decoder->nb_bit_cur())
                       );

              nb_transfered = ElMin(nb_decoded,nb-sum_nb_added);
              if (res)
                  memcpy(res+sum_nb_added,decoded,nb_transfered);
              sum_nb_added += nb_transfered;
      }
      _deb_buffered = 0;
      _nb_buffered = nb_decoded - nb_transfered;
      memcpy(_buf,decoded+nb_transfered,_nb_buffered.CKK_Byte4AbsLLO());

      return nb;
}

tFileOffset RelToAbs(tRelFileOffset anOff)
{
/*
   ELISE_ASSERT(anOff.>=0,"RelToAbs Offset");
   return tFileOffset(anOff);
*/
    return anOff.CKK_AbsLLO();
}


tRelFileOffset Packed_LZW_Decompr_Flow::Rseek(tRelFileOffset nbr)
{
      tFileOffset nb = RelToAbs(nbr);
      tFileOffset sum_nb_added = ElMin(nb,_nb_buffered);
      _nb_buffered -= sum_nb_added;
      _deb_buffered += sum_nb_added;

      if (sum_nb_added == nb)
         return sum_nb_added;


      const U_INT1 * decoded  = 0; // warning init

      int      nb_decoded     = 0;
      int      nb_transfered  = 0;

      while (sum_nb_added < nb)
      {
             decoded = _decoder->new_code
                       (
                            nb_decoded,
                            _flxi->nexti(_decoder->nb_bit_cur())
                       );

              nb_transfered = ElMin(nb_decoded,(nb-sum_nb_added).CKK_IntBasicLLO());
              sum_nb_added += nb_transfered;
      }
      _deb_buffered = 0;
      _nb_buffered = nb_decoded - nb_transfered;
      memcpy(_buf,decoded+nb_transfered,_nb_buffered.CKK_Byte4AbsLLO());

      return nb;
}

Packed_LZW_Decompr_Flow::~Packed_LZW_Decompr_Flow()
{
     delete _decoder;
     if (_read)
     {
         DELETE_VECTOR(_buf,0);
         delete _flxi;
     }
     else
     {
         delete _flxo;
     }
}

tFileOffset Packed_LZW_Decompr_Flow::Write(const U_INT1 * vals,tFileOffset nbo)
{
    int nb = nbo.CKK_IntBasicLLO();
    LZW_decoder::wcodes wc;

    for (int i=0 ; i<nb ; i++)
    {
        _decoder->write_codes(vals[i],wc,false);
        for (INT j =0; j<wc.nbc; j++)
            _flxo->puti(wc.c[j],wc.nbb[j]);
    };
    return nb;
}

void  Packed_LZW_Decompr_Flow::Write(const INT * vals,tFileOffset nbo)
{
    int nb = nbo.CKK_IntBasicLLO();
    const int sz_buf = 100;
    U_INT1 buf[sz_buf];

   for (int i=0 ; i<nb; i+= sz_buf)
   {
       int nb_loc = ElMin(sz_buf,nb-i);
       convert(buf,vals+i,nb_loc);
       Write(buf,nb_loc);
   }
}


bool Packed_LZW_Decompr_Flow::compressed() const
{
     return true;
}

void Packed_LZW_Decompr_Flow::reset()
{
    init();
    if (_read)
       _flxi->reset();
    else
    {
        LZW_decoder::wcodes wc;
        _decoder->write_codes
        (
            0,  // not used, any value will be OK
            wc,
            true
        );
        for (INT j =0; j<wc.nbc; j++)
            _flxo->puti(wc.c[j],wc.nbb[j]);
       _flxo->reset();
    }
    _decoder->re_init();
}


void test_lzw(char * ch)
{
    INT code[120];
    INT nbc = 0;
    {
        LZW_decoder::wcodes wc;
        LZW_TIF_decoder * dec = new LZW_TIF_decoder(false);

        cout << "CH : " << ch << "\n";
        for (;*ch;ch++)
        {
            dec->write_codes((*ch)-'0',wc,false);
            for (INT i= 0; i<wc.nbc ;i++)
            {
                code[nbc++] =  wc.c[i];
                cout << "OUT : " << wc.c[i]  << "\n";
            }
        }
        dec->write_codes(9,wc,true);
        for (INT i= 0; i<wc.nbc ;i++)
        {
             code[nbc++] =  wc.c[i];
             cout << "OUT : " << wc.c[i]  << "\n";
        }
    }
cout << "=============================== \n";

    {
        LZW_TIF_decoder * dec = new LZW_TIF_decoder(true);

        for (INT i=0; i<nbc -1 ; i++)
        {
             int nb_decoded;

             const U_INT1  * decoded = dec->new_code(nb_decoded,code[i]);
             cout << "IN " << code[i] << " : ";
             for(int j=0 ; j<nb_decoded ; j++)
                cout << (char)('0'+decoded[j]) ;
             cout << "\n";
        }
    }
       getchar();
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
