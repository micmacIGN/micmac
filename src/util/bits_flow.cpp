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





/**************************************************/
/*                                                */
/*     Flux_Of_VarLI                              */
/*                                                */
/**************************************************/

Flux_Of_VarLI::Flux_Of_VarLI(Flux_Of_Byte * aBF,bool flx_flush) :
     _flx_byte  (aBF),
     _flx_flush (flx_flush)
{
}



Flux_Of_VarLI::~Flux_Of_VarLI()
{
     if (_flx_flush)
        delete _flx_byte;
}

tFileOffset Flux_Of_VarLI::tell()
{
    return _flx_byte->tell();
}


/***************************************************/
/*                                                 */
/*      MSBitFirst_Flux_Of_VarLI                   */
/*                                                 */
/***************************************************/


void MSBitFirst_Flux_Of_VarLI::reset()
{
     _last_bit_read = 0;
}

MSBitFirst_Flux_Of_VarLI::MSBitFirst_Flux_Of_VarLI
(
    Flux_Of_Byte * flx_byte,
    bool           flx_flush
) :
           Flux_Of_VarLI (flx_byte,flx_flush)
{
    reset();
}


INT MSBitFirst_Flux_Of_VarLI::nexti(INT nb)
{
    INT res = 0;

    INT nb_bit_to_read;
    for
    (
         INT sum_nb_bit_read=0             ; 
         sum_nb_bit_read<nb                ; 
         sum_nb_bit_read+=nb_bit_to_read
    )
    {
        if (_last_bit_read ==0)
        {
            _last_bit_read = 8;
            _last_char_read = _flx_byte->Getc();
        }

        nb_bit_to_read = ElMin(_last_bit_read,(nb-sum_nb_bit_read));
        res <<= nb_bit_to_read;
        res |= sub_bit
               (
                    _last_char_read,
                    _last_bit_read-nb_bit_to_read,
                    _last_bit_read
               );
        _last_bit_read -= nb_bit_to_read; 
    }
   
    return res;
}

Flux_Of_VarLI * Flux_Of_VarLI::new_flx(  Flux_Of_Byte *   flx_byte,
                                         bool             msbf,
                                         bool             flx_flush
                                      )
{
     
   if (msbf )
      return new MSBitFirst_Flux_Of_VarLI(flx_byte,flx_flush);
   else
      return new LSBitFirst_Flux_Of_VarLI(flx_byte,flx_flush);
};

/***************************************************/
/*                                                 */
/*      LSBitFirst_Flux_Of_VarLI                   */
/*                                                 */
/***************************************************/

Flux_OutVarLI * Flux_OutVarLI::new_flx
(
    Flux_Of_Byte * flx_byte,
    bool msbf,
    bool flx_flush
)
{
      
    if (msbf)
        return new MSBF_Flux_OutVarLI(flx_byte,flx_flush);
    else
        return new LSBF_Flux_OutVarLI(flx_byte,flx_flush);
};

tFileOffset Flux_OutVarLI::tell()
{
   return _flx->tell();
}


void LSBitFirst_Flux_Of_VarLI::reset()
{
    _last_bit_read = 8;
}

LSBitFirst_Flux_Of_VarLI::LSBitFirst_Flux_Of_VarLI
(
       Flux_Of_Byte * flx_byte,
       bool           flx_flush
) :
           Flux_Of_VarLI (flx_byte,flx_flush)
{
    reset();
}


INT LSBitFirst_Flux_Of_VarLI::nexti(INT nb)
{
    INT res = 0;

    INT nb_bit_to_read;
    for
    (
         INT sum_nb_bit_read=0             ; 
         sum_nb_bit_read<nb                ; 
         sum_nb_bit_read+=nb_bit_to_read
    )
    {
        if (_last_bit_read ==8)
        {
            _last_bit_read = 0;
            _last_char_read = _flx_byte->Getc();
        }

        nb_bit_to_read = ElMin(8-_last_bit_read,(nb-sum_nb_bit_read));
        res |= sub_bit
               (
                    _last_char_read,
                    _last_bit_read,
                    _last_bit_read+nb_bit_to_read
               ) << sum_nb_bit_read;
        _last_bit_read += nb_bit_to_read; 
    }
   
    return res;
}


/***************************************************/
/*                                                 */
/*           BitsPacked_PFOB                       */
/*                                                 */
/***************************************************/

tFileOffset BitsPacked_PFOB::tell()
{
   return _pfob->tell();
}




BitsPacked_PFOB::BitsPacked_PFOB
(
     Packed_Flux_Of_Byte *     pfob,
     INT                       nbb,
     bool                      msbf,
     bool                      read_mode,
     INT                       nb_el
) :
            Packed_Flux_Of_Byte  (1),
           _pfob                 (pfob),
           _tbb                  (Tabul_Bits_Gen::tbb(nbb,msbf)),
           _nb_el                (nb_el)  // Nombre de cannaux ?
{
    ASSERT_INTERNAL
    ( 
        (nbb==1) || (nbb==2) || (nbb==4),
        "BitsPacked_PFOB : incorrect number of bit"
    );
    _nb_pb =  8/nbb;
    _v_max = 1 << nbb;

    ASSERT_INTERNAL
    (
        (_pfob->sz_el() == 1),
        "BitsPacked_PFOB : flux to pack , sz_el != 1"
    );
    _read_mode = read_mode;
    _pf_compr = _pfob->compressed();
    _i_buf = 0;

}

bool BitsPacked_PFOB::compressed() const
{
     return _pf_compr;
}


void  BitsPacked_PFOB::AseekFp(tFileOffset nb)
{

     _i_buf =   0 ;
     _pfob->AseekFp(nb);
}


BitsPacked_PFOB::~BitsPacked_PFOB()
{
     delete _pfob;
}

tFileOffset BitsPacked_PFOB::_Read(U_INT1 * res,tFileOffset nbo) 
{
    int nb = nbo.CKK_IntBasicLLO();
    ELISE_ASSERT(nb>=0,"Read-neg-in-BitsPacked_PFOB");
    for (int i = 0 ; i<nb ; i++)
    {
        if( ! _i_buf)
           _pfob->Read(&_v_buf,1);
        res[i] = kieme_val(_v_buf,_i_buf);
        _i_buf = (_i_buf+1) % _nb_pb;
   }
   return nb;
}

tFileOffset BitsPacked_PFOB::Read(U_INT1 * res,tFileOffset nb) 
{
    return _Read(res,nb*_nb_el);
}

tFileOffset BitsPacked_PFOB::_Write(const U_INT1 * data,tFileOffset nbo) 
{
    int nb = nbo.CKK_IntBasicLLO();
    if (El_User_Dyn.active())
    {
         INT index = index_values_out_of_range
                       (data,nb,(U_INT1)0,(U_INT1)_v_max);

	if (index!=INDEX_NOT_FOUND)
	{
        
         El_User_Dyn.ElAssert
         (
             index == INDEX_NOT_FOUND,
             EEM0  << "values out of range in bits-file writing \n"
                   << "|   value = "   << (INT) data[index]     << "\n"
                   << "|   interval = [" << 0 << " ---  " << _v_max  << "["
         );
        }
         
    }
    
    if (_i_buf && (!_pf_compr))
       _pfob->Rseek(-1);

    for (int i = 0 ; i<nb ; i++)
    {
        _v_buf = set_kieme_val(_v_buf,data[i],_i_buf);
        _i_buf = (_i_buf+1) % _nb_pb;
        if( ! _i_buf)
           _pfob->Write(&_v_buf,1);
    }


    if (_i_buf && (!_pf_compr))
    {
             U_INT1 val;
            _pfob->Read(&val,1);
             for (INT i = _i_buf; i<_nb_pb ; i++)
                 _v_buf = set_kieme_val(_v_buf,kieme_val(val,i),i);
             _pfob->Rseek(-1);
             _pfob->Write(&_v_buf,1);
    }

    return nb;

}

tFileOffset BitsPacked_PFOB::Write(const U_INT1 * data,tFileOffset nb) 
{
    return _Write(data,nb*_nb_el);
}

/*
INT BitsPacked_PFOB::_Rseek(INT nb_el) 
{
      U_INT1 buf[8];
      INT nb_el_0 = nb_el;

      INT nb_read = ElMin(nb_el,_nb_pb-_i_buf);
      nb_el -= nb_read;
      INT nb_seek = nb_el/_nb_pb; 

      _Read(buf,nb_read); 

      //std::cout << "PFOB::_Rseek " << nb_seek << " " << nb_read << " " << _nb_pb << "\n";
      if (nb_seek)
      {
         _pfob->Rseek(nb_seek);
         _i_buf = 0;
         nb_el -= nb_seek * _nb_pb;
      }

      _Read(buf,nb_el);
      return nb_el_0;
}
*/

//
// MODIF MPD 23/11/07 pour pb _Rseek negatifs
//
//   Si on imagine que _i_buf varie en dehors de [0 _nb_pp[
//   alors l'octet courant doit etre Div(_i_buf-1,_nb_pp) ,
//   en effet lorsque _i_buf=0 mod _nb_pp, la premier
//   chose que l'on fait dans fread est de lire un octet

/*
tRelFileOffset BitsPacked_PFOB::_Rseek(tRelFileOffset nb_elo) 
{
      int nb_el = nb_elo.CKK_IntBasicLLO();
      // _i_buf += nb_el;
      int aNbSeek = Elise_div(_i_buf+nb_el-1,_nb_pb)-Elise_div(_i_buf-1,_nb_pb);
     _i_buf = mod(_i_buf+nb_el,_nb_pb);

      // Il faut si _i_buf !=0 que l'octet de bufferisation soit remplis, donc
      // on avance de 1 de moins et read ensuite
     if ( _i_buf !=0)
         aNbSeek--;
     _pfob->Rseek(aNbSeek);
     if ( _i_buf !=0)
        _pfob->Read(&_v_buf,1);


     return nb_el;
}
*/

// bool DebugRseek = false;

// MODIF MPD 28/01/21 pour overflow dans CKK_IntBasicLLO()

tRelFileOffset BitsPacked_PFOB::_Rseek(tRelFileOffset nb_rfo) 
{
     // On va parcourir par paquet, de taille sz buf
     tLowLevelFileOffset SzBuf = 1<<30;
     tLowLevelFileOffset nb_elo =  nb_rfo.BasicLLO() ;
     tLowLevelFileOffset aSign = (nb_elo >=0) ? 1 : - 1;  // on va decouper la valeur absolue
     tLowLevelFileOffset nb_abs_elo =  nb_elo * aSign;

     for (tLowLevelFileOffset aOffset = 0 ; aOffset<nb_abs_elo ; aOffset+=SzBuf)
     {
          // ensuite une fois qu'on a decoupe en paquet suffisement petit on le gere comme avant en int
          int nb_el = ElMin(nb_abs_elo-aOffset,SzBuf) * aSign;
          // _i_buf += nb_el;
          int aNbSeek = Elise_div(_i_buf+nb_el-1,_nb_pb)-Elise_div(_i_buf-1,_nb_pb);
         _i_buf = mod(_i_buf+nb_el,_nb_pb);

          // Il faut si _i_buf !=0 que l'octet de bufferisation soit remplis, donc
          // on avance de 1 de moins et read ensuite
         if ( _i_buf !=0)
             aNbSeek--;
         _pfob->Rseek(aNbSeek);
         if ( _i_buf !=0)
            _pfob->Read(&_v_buf,1);
     }

     return nb_rfo;
}
/*
*/


tRelFileOffset BitsPacked_PFOB::Rseek(tRelFileOffset nb) 
{
   return _Rseek(nb*_nb_el);
}

void byte_inv_2(void * t)
{
     ElSwap ( ((char *) t)[0], ((char *) t)[1]);
}

void byte_inv_4(void * t)
{
     ElSwap ( ((char *) t)[0], ((char *) t)[3]);
     ElSwap ( ((char *) t)[1], ((char *) t)[2]);
}

/*
void byte_inv_8(void * t)
{
     ElSwap ( ((char *) t)[0], ((char *) t)[7]);
     ElSwap ( ((char *) t)[1], ((char *) t)[6]);
     ElSwap ( ((char *) t)[2], ((char *) t)[5]);
     ElSwap ( ((char *) t)[3], ((char *) t)[4]);
}
*/




void byte_inv_8(void * t)
{
     ElSwap
     (
          ((char *) t)[0],
          ((char *) t)[7]
     );
     ElSwap
     (
          ((char *) t)[1],
          ((char *) t)[6]
     );
     ElSwap
     (
          ((char *) t)[2],
          ((char *) t)[5]
     );
     ElSwap
     (
          ((char *) t)[3],
          ((char *) t)[4]
     );
}

void byte_inv_16(void * t)
{
     ElSwap
     (
          ((char *) t)[0],
          ((char *) t)[15]
     );
     ElSwap
     (
          ((char *) t)[1],
          ((char *) t)[14]
     );
     ElSwap
     (
          ((char *) t)[2],
          ((char *) t)[13]
     );
     ElSwap
     (
          ((char *) t)[3],
          ((char *) t)[12]
     );
     ElSwap
     (
          ((char *) t)[4],
          ((char *) t)[11]
     );
     ElSwap
     (
          ((char *) t)[5],
          ((char *) t)[10]
     );
     ElSwap
     (
          ((char *) t)[6],
          ((char *) t)[9]
     );
     ElSwap
     (
          ((char *) t)[7],
          ((char *) t)[8]
     );
}

void byte_inv_tab(void * t,INT byte_by_el,INT nb_el)
{
   if (byte_by_el == 1)
      return;

   char * tab = (char *) t;
   for (INT i = 0; i<nb_el ; i++)
   {
       for
       (
             INT byt_0 = 0, byt_1 = byte_by_el-1; 
             byt_0 < byt_1;
             byt_0++,byt_1--
       )
             ElSwap(tab[byt_0],tab[byt_1]);

       tab += byte_by_el;
   }
}




/***************************************************/
/*                                                 */
/*      MSBF_Flux_OutVarLI                         */
/*                                                 */
/***************************************************/

INT Flux_OutVarLI::kth()
{
   return 8-_bit_to_write;
}


Flux_OutVarLI::~Flux_OutVarLI()
{
     if (_flx_flush)
       delete _flx;
}



Flux_OutVarLI::Flux_OutVarLI(Flux_Of_Byte * flx, bool flx_flush) :
   _flx       (flx),
   _flx_flush (flx_flush)
{
}

  //========================================

void MSBF_Flux_OutVarLI::reset()
{
    if (_bit_to_write != 8)
            _flx->Putc(_char_to_write);
    _bit_to_write = 8;
    _char_to_write =0;
}

MSBF_Flux_OutVarLI::MSBF_Flux_OutVarLI(Flux_Of_Byte * flx,bool flx_flush) :
           Flux_OutVarLI (flx,flx_flush)
{
    _bit_to_write = 8;
    reset();
}

INT MSBF_Flux_OutVarLI::puti(INT val,INT nb)
{
    INT res = 0;

    INT nb_bit_to_write;
    for
    (
         INT sum_nb_bit_written=0             ; 
         sum_nb_bit_written<nb                ; 
         sum_nb_bit_written+=nb_bit_to_write
    )
    {
        nb_bit_to_write = ElMin(_bit_to_write,(nb-sum_nb_bit_written));
        _char_to_write = set_sub_bit
                         (
                             _char_to_write,
                             val>>(nb-sum_nb_bit_written-nb_bit_to_write),
                             _bit_to_write-nb_bit_to_write,
                             _bit_to_write
                         );

        _bit_to_write -= nb_bit_to_write;

        if (_bit_to_write ==0)
        {
            _bit_to_write = 8;
            _flx->Putc(_char_to_write);
            _char_to_write =0;
        }
    }
   
    return res;
}



  //========================================

void LSBF_Flux_OutVarLI::reset()
{
    if (_bit_to_write != 0)
            _flx->Putc(_char_to_write);
    _bit_to_write = 0;
    _char_to_write =0;
}

LSBF_Flux_OutVarLI::LSBF_Flux_OutVarLI(Flux_Of_Byte * flx,bool flx_flush) :
           Flux_OutVarLI (flx,flx_flush)
{
    _bit_to_write = 0;
    reset();
}

INT LSBF_Flux_OutVarLI::puti(INT val,INT nb)
{
    INT res = 0;

    INT nb_bit_to_write;
    for
    (
         INT sum_nb_bit_written=0             ; 
         sum_nb_bit_written<nb                ; 
         sum_nb_bit_written+=nb_bit_to_write
    )
    {
        nb_bit_to_write = ElMin(8-_bit_to_write,(nb-sum_nb_bit_written));
        _char_to_write = set_sub_bit
                         (
                             _char_to_write,
                             val>>(sum_nb_bit_written),
                             _bit_to_write,
                             _bit_to_write+nb_bit_to_write
                         );

        _bit_to_write += nb_bit_to_write;

        if (_bit_to_write ==8)
        {
            _bit_to_write = 0;
            _flx->Putc(_char_to_write);
            _char_to_write =0;
        }
    }
   
    return res;
}





INT ElFlagAllocator::flag_alloc()
{
    for (INT k =0; k<32 ; k++)
        if (! _flag.kth(k))
        {
             _flag.set_kth_true(k);
             return k;
        }
    ELISE_ASSERT(false,"ElFlagAllocator::flag_alloc");
    return -1;
}

void ElFlagAllocator::flag_free(INT k)
{
      ELISE_ASSERT(_flag.kth(k),"ElFlagAllocator::flag_free");
     _flag.set_kth_false(k);
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
