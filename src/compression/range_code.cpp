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
#define NoOperatorVirgule

#include "StdAfx.h"




/***************************************************************************/
/*                                                                         */
/*                       MS_RANGE_DECODER                                  */
/*                                                                         */
/***************************************************************************/

int MS_RANGE_DECODER::inbyte()
{
    return(_flxi->nexti(8));
}

U_INT1 MS_RANGE_DECODER::start_decoding()
{
   int c = inbyte();
   // if (c==EOF) return EOF;
   buffer = inbyte();
   low = buffer >> (8-EXTRA_BITS);
   range = 1 << (int) EXTRA_BITS;
   return c;       
}

void MS_RANGE_DECODER::dec_normalize()
{   
    while (range <= Bottom_value)
    {   
        low = (low<<8) | ((buffer<< (int) EXTRA_BITS)&0xff);
        buffer = inbyte();
        low |= buffer >> (8-EXTRA_BITS);
        range <<= 8;
    }                            
}

/* Calculate culmulative frequency for next symbol. Does NO update!*/
/* tot_f is the total frequency                              */
/* or: totf is 1<<shift                                      */
/* returns the culmulative frequency                         */     

MS_RANGE_DECODER::freq MS_RANGE_DECODER::decode_culfreq(freq tot_f )
{   
    freq tmp;
    dec_normalize();
    help = range/tot_f;
    tmp = low/help;
    return (tmp>=tot_f ? tot_f-1 : tmp);
}                   

MS_RANGE_DECODER::freq MS_RANGE_DECODER::decode_culshift(freq shift )
{   
    freq tmp;
    dec_normalize();
    help = range>>shift;
    tmp = low/help;
    return (tmp>>shift ? (1<<shift)-1 : tmp);
}           

void MS_RANGE_DECODER::decode_update(freq sy_f,freq lt_f,freq tot_f)
{   
    code_value tmp;
    tmp = help * lt_f;
    low -= tmp;
    if (lt_f + sy_f < tot_f)
       range = help * sy_f;
    else
       range -= tmp;
}                          

unsigned char MS_RANGE_DECODER::decode_byte()
{   
    unsigned char tmp = decode_culshift(8);
    decode_update(1,tmp,1<<8);
    return tmp;
}               

unsigned short MS_RANGE_DECODER::decode_short()
{   
    unsigned short tmp = decode_culshift(16);
    decode_update(1,tmp,(freq)1<<16);
    return tmp;
}

void MS_RANGE_DECODER::done_decoding()
{   
     dec_normalize();      /* normalize to use up all bytes */
}    

/***************************************************************************/
/*                                                                         */
/*                       Martin_Schindler_RCODE                            */
/*                                                                         */
/***************************************************************************/

// #define Top_value ((code_value)1 << (CODE_BITS-1))   
const U_INT4 Martin_Schindler_RCODE::Top_value 
             = (      (MS_RANGE_ENCODER::code_value)1  
                  <<  (MS_RANGE_ENCODER::CODE_BITS-1)
               );

// #define Bottom_value (Top_value >> 8)  
const U_INT4 Martin_Schindler_RCODE::Bottom_value 
             = (      (MS_RANGE_ENCODER::code_value)1  
                  <<  (MS_RANGE_ENCODER::CODE_BITS-9)
               );


/***************************************************************************/
/*                                                                         */
/*                       MS_RANGE_ENCODER                                  */
/*                                                                         */
/***************************************************************************/

void MS_RANGE_ENCODER::outbyte(INT x)
{
    _flxo->puti(x,8);
}

void MS_RANGE_ENCODER::start_encoding( U_INT1 c )
{   
    low = 0;                /* Full code range */
    range = Top_value;
    buffer = c;
    help = 0;               /* No bytes to follow */
    bytecount = 0;
}                 

/* I do the normalization before I need a defined state instead of */
/* after messing it up. This simplifies starting and ending.       */

void MS_RANGE_ENCODER::enc_normalize()
{   
    while(range <= Bottom_value)     /* do we need renormalisation?  */
    {   
        if (low < (0xff<<(int)SHIFT_BITS))  /* no carry possible --> output */
        {   
            outbyte(buffer);
            for(; help; help--)
                outbyte(0xff);
            buffer = (unsigned char)(low >> SHIFT_BITS);
        } 
        else if (low & Top_value) /* carry now, no future carry */
        {   
            outbyte(buffer+1);
            for(; help; help--)
                outbyte(0);
            buffer = (unsigned char)(low >> SHIFT_BITS);
        } 
        else                           /* passes on a potential carry */
        {
            /*
                bytestofollow ?? Not in class
                ELISE_ASSERT
                (
                    bytestofollow++ != 0xffffffffL,
                    "Too many bytes outstanding - File too large\n"
                );
            */
            help++;
        }
        range <<= 8;
        low = (low<<8) & (Top_value-1);
        bytecount++;
    }
}                                           
void MS_RANGE_ENCODER::encode_freq(freq sy_f,freq lt_f,freq tot_f)
{       
        code_value r, tmp;
        enc_normalize();
        r = range / tot_f;
        tmp = r * lt_f;
        if (lt_f+sy_f < tot_f)
           range = r * sy_f;
        else
           range -= tmp;
        low += tmp;
}               

void MS_RANGE_ENCODER::encode_shift(freq sy_f,freq lt_f,freq shift)
{    

     code_value r, tmp;
     enc_normalize();
     r = range >> shift;
     tmp = r * lt_f;
     if ((lt_f+sy_f) >> shift)
        range -= tmp;
     else
        range = r * sy_f;
     low += tmp;
}            

void MS_RANGE_ENCODER::encode_byte(freq b)
{
     encode_shift((freq)1,(freq)(b),(freq)8);
}
void MS_RANGE_ENCODER::encode_short(freq b)
{
     encode_shift((freq)1,(freq)(b),(freq)16);
}


/* Finish encoding                                           */
/* actually not that many bytes need to be output, but who   */
/* cares. I output them because decode will read them :)     */

void MS_RANGE_ENCODER::done_encoding()
{   
    U_INT tmp;
    enc_normalize();     /* now we have a normalized state */
    bytecount += 5;
    if ((low & (Bottom_value-1)) < (bytecount>>1))
       tmp = low >> SHIFT_BITS;
    else
       tmp = (low >> SHIFT_BITS) + 1;
    if (tmp > 0xff) /* we have a carry */
    {   outbyte(buffer+1);
        for(; help; help--)
            outbyte(0);
    } else  /* no carry */
    {   outbyte(buffer);
        for(; help; help--)
            outbyte(0xff);
    }
    outbyte(tmp & 0xff);
    outbyte((bytecount>>16) & 0xff);
    outbyte((bytecount>>8) & 0xff);
    outbyte(bytecount & 0xff);
}                                  

/***************************************************************************/
/*                                                                         */
/*                       cMS_SimpleArithmEncoder                           */
/*                                                                         */
/***************************************************************************/

cMS_SimpleArithmEncoder::cMS_SimpleArithmEncoder
(
    const std::vector<REAL> & aVProbas,
    INT               aNbBits,
    Flux_OutVarLI *   aFlux,
    char              aV0
) :
  mEnc (aFlux),
  mNbBits (aNbBits),
  mTot    (1<<mNbBits),
  mNbVals ((int) aVProbas.size())
{
   ELISE_ASSERT(mNbVals<=256,"Too Much in cMS_SimpleArithmEncoder");
   INT aCapa = mTot - mNbVals;
   ELISE_ASSERT(aCapa>=0,"Not Enough bits in cMS_SimpleArithmEncoder");

   REAL aSom = 0.0;
   for (INT aK=0 ; aK<mNbVals ; aK++)
   { 
       REAL aP = aVProbas[aK];
       ELISE_ASSERT (aP > 0, "Negative weigth in cMS_SimpleArithmEncoder");
       aSom += aP;
   }


   mCumuls.push_back(0);
   for (INT aK = 0 ; aK< mNbVals ; aK++)
   {
        REAL aP = aVProbas[aK];
        INT aV = round_ni(aCapa * (aP/aSom));
        mFreqs.push_back(1+aV);

        aCapa -= aV;
        aSom -= aP;
        mCumuls.push_back(mCumuls.back()+mFreqs.back());
   }
   mEnc.start_encoding(aV0);

}


void cMS_SimpleArithmEncoder::PushCode(INT aCode)
{
    ELISE_ASSERT((aCode>=0) && (aCode<mNbVals),"Out Code in PushCode");


    mEnc.encode_shift(mFreqs[aCode],mCumuls[aCode],mNbBits);
}

const std::vector<INT> &  cMS_SimpleArithmEncoder::Cumuls() const
{
   return mCumuls;
}

const std::vector<INT> &  cMS_SimpleArithmEncoder::Freqs() const
{
   return mFreqs;
}

INT cMS_SimpleArithmEncoder::Tot() const
{
   return mTot;
}


void   cMS_SimpleArithmEncoder::Done()
{
    mEnc.done_encoding();
}

/***************************************************************************/
/*                                                                         */
/*                        cMS_SimpleArithmDecoder                          */
/*                                                                         */
/***************************************************************************/


cMS_SimpleArithmDecoder::cMS_SimpleArithmDecoder
(
      const std::vector<INT> &  aVCumuls,
      Flux_Of_VarLI *           aFlx

)  :
   mDec    (aFlx),
   mNbVals ((int) aVCumuls.size()),
   mP2     (aVCumuls.back()),
   mNbBits (round_ni(log(double(mP2))/log(2.0)))
{
   ELISE_ASSERT(aVCumuls.size()>1,"cMS_SimpleArithmDecoder:: Sz");
   ELISE_ASSERT(aVCumuls[0]==0,"cMS_SimpleArithmDecoder::   [0]");

   for (INT aK=0 ; aK<(mNbVals-1) ; aK++)
       ELISE_ASSERT
       (
           aVCumuls[aK]<aVCumuls[aK+1],
           "cMS_SimpleArithmDecoder:: K<K+1"
       );

   ELISE_ASSERT(is_pow_of_2(mP2),"cMS_SimpleArithmDecoder:: P2");

   mVDecod.reserve(mP2);
   for (int aC=0 ; aC<(mNbVals-1) ; aC++)
   {
       INT aK0 = aVCumuls[aC];
       INT aK1 = aVCumuls[aC+1];
       for (INT aK=aK0 ; aK<aK1 ; aK++)
           mVDecod.push_back(aC);
       mCumuls.push_back(aK0);
       mFreqs.push_back(aK1-aK0);
   }
   mV0 = mDec.start_decoding();
}

U_INT1 cMS_SimpleArithmDecoder::V0()
{
     return mV0;
}

INT cMS_SimpleArithmDecoder::Dec()
{
    INT aCodeFull = mDec.decode_culshift(mNbBits);
    INT aCode = mVDecod[aCodeFull];
    mDec.decode_update(mFreqs[aCode],mCumuls[aCode],mP2);
    return aCode;
}

void  cMS_SimpleArithmDecoder::Done()
{
    mDec.done_decoding();
}
// void MS_RANGE_DECODER::done_decoding()

    // unsigned char tmp = decode_culshift(8);
    // decode_update(1,tmp,1<<8);
    // return tmp;




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
