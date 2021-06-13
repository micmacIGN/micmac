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





cConvertBaseXXX cConvertBaseXXX::StdBase64()
{
   return cConvertBaseXXX
          (
                    256,3,64,4,
		    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		    "abcdefghijklmnopqrstuvwxyz"
		    "0123456789+\\",
		    80
          );
}

cConvertBaseXXX::cConvertBaseXXX
(
     int aBaseIn, 
     int aNbBaseIn,
     int aBaseOut,  
     int aNbBaseOut, 
     const std::string & aSetChar,
     int aNbMaxParLine
)  :
      mBaseIn       (aBaseIn),
      mNbBaseIn     (aNbBaseIn),
      mNbCCur       (0),
      mCurI    (0),
      mBaseOut      (aBaseOut),
      mNbBaseOut    (aNbBaseOut),
      mSetChar      (aSetChar),
      mNbMaxParLine (aNbMaxParLine ? aNbMaxParLine: -1 ), // cas 0 mal determine
      mNbInCurLine  (0)
{
   ELISE_ASSERT
   (
       mNbBaseOut<theMaxNbOut,
       "Excess-mBaseOut in cExportBaseXXX"
   );
   ELISE_ASSERT
   (
      int(mSetChar.length()) == mBaseOut,
     "Inc-SetChar-mNbBaseOut in cConvertBaseXXX"
   );

   mPowbaseIn[0] = 1;
   for (int aK = 1; aK<mNbBaseIn ; aK++)
       mPowbaseIn[aK] = mPowbaseIn[aK-1] * mBaseIn;
   mPowbaseOut[mNbBaseOut-1] = 1;
   for (int aK = mNbBaseOut-2; aK>=0 ; aK--)
       mPowbaseOut[aK] = mPowbaseOut[aK+1] * mBaseOut;

   for (int aK=0 ; aK<256 ; aK++)
       mLutInv[aK] = -1;
   for (int aK=0 ; aK<mBaseOut; aK++)
       mLutInv[mSetChar[aK]-CHAR_MIN] = aK;
   if (mNbMaxParLine >=0)
   {
      ELISE_ASSERT
      (
          mLutInv['\n'-CHAR_MIN] == -1,
	  "Incoherence-in cConvertBaseXXX"
      );
   }
}
 
void  cConvertBaseXXX::PutNC(const void * aPtr,int aNbC,std::ostream & anOs)
{
    const char * aCPtr = (const char *) aPtr;
    for (int aK=0 ; aK<aNbC ; aK++)
       PutC(aCPtr[aK],anOs);
}


void cConvertBaseXXX::PutC(char aC,std::ostream & aStream)
{ 
   // std::cout << "IN-PUT : " << int (aC) << "\n";
   mCurI = mCurI * mBaseIn + (aC-CHAR_MIN);
   mNbCCur++;
   if (mNbCCur == mNbBaseIn)
   {
      for (int aK=0 ; aK<mNbBaseOut ; aK++)
      {
          int aCout = (int)( mCurI/mPowbaseOut[aK] );
		  aStream << mSetChar[aCout];
		  mCurI = mCurI - aCout * mPowbaseOut[aK];
		  mNbInCurLine++;
		  if (mNbInCurLine==mNbMaxParLine)
		  {
			 aStream <<  "\n";
			 mNbInCurLine=0;
		  }
      }
      mNbCCur = 0;
      // Si mCurI != 0, toute l'info n'a pas ete transmise !!
      ELISE_ASSERT(mCurI==0,"Inc in cConvertBaseXXX::PutC");
   }
}

int  cConvertBaseXXX::GetC(std::istream & aStream)
{
   if (mNbCCur==0)
   {
      mCurI = 0;
      for (int aK=0 ; aK<mNbBaseOut ; )
      {
          char aCin = CHAR_MIN;
	  aStream >> aCin;
	  ELISE_ASSERT(aCin!=CHAR_MIN,"Empty file un cConvertBaseXXX::GetC");
	  int anI = mLutInv[aCin-CHAR_MIN];
	  // std::cout <<  "C=" << int(aCin)  << " " << aCin << " " << anI << "\n";
	  if (anI!=-1)
	  {
               mCurI = mCurI*  mBaseOut + anI;
	       aK++;
	  }
      }
      mNbCCur = mNbBaseIn;
   }
   mNbCCur--;
   int aRes = (int)( mCurI/mPowbaseIn[mNbCCur] );
   mCurI -= aRes * mPowbaseIn[mNbCCur];
   aRes += CHAR_MIN;
   // std::cout << "OUTPUT : " << aRes << "\n";
   return aRes;
}

void cConvertBaseXXX::GetNC(void * aPtr,int aNbC,std::istream & anIs)
{
   char * aCPtr = (char *) aPtr;
   for (int aK=0 ; aK<aNbC ; aK++)
   {
   // std::cout << "K=" << aK << "\n";
       aCPtr[aK] = GetC(anIs);
   }
}


void cConvertBaseXXX::Close(std::ostream & aStream)
{
   while(mNbCCur) PutC(0,aStream);
   if (mNbInCurLine !=0)
   {
      aStream <<  "\n";
      mNbInCurLine=0;
   }
}



void PS_A85::putchar_maxline(char c)
{
     _fd << c;
     _nb ++;
     if (_nb == max_line)
     {
        _nb = 0;
        _fd << "\n";
     }
}


const INT PS_A85::coeffB256[4] = {(1<<24),(1<<16),(1<<8),1};
const INT PS_A85::coeffB85[5] =
      {  85*85*85*85  ,  85*85*85  ,  85*85  ,  85  ,  1  };



void PS_A85::reinit_Base256()
{
     sB256 = 0.0;
     nB256 = 0;
}

void PS_A85::put85(INT nb)
{

     for (INT k=0 ; k<nb ; k++)
     {
         int c =  (int) (sB256 / coeffB85[k]);
         putchar_maxline('!'+c);
         sB256 -= c * (double) coeffB85[k];
     }
}


void PS_A85::close_block()
{
    if (nB256)
      put85(1+nB256);
    reinit_Base256();
    _fd << "~>\n";
}
void PS_A85::put(INT v)
{
    sB256 += v * (double) coeffB256[nB256++];
    if (nB256 == 4)
    {
       if (sB256 == 0)
          putchar_maxline('z');
       else
          put85(5);
       reinit_Base256();
    }
}



/**************************************************************/
/*                                                            */
/*         FILTERS  PS                                        */
/*                                                            */
/**************************************************************/

/*
     A set of classe to encode binary images using PS build-in
    facilities for image compression.
*/

      //================================
      //      Ps_Filter 
      //================================

class Ps_Filter : public Mcheck
{
    public :

        virtual ~Ps_Filter()
        {
             delete _TopFlx;
        }

        Ps_Filter() :
            _mflx       (new Mem_Packed_Flux_Of_Byte(2000,1)) ,
            _TopFlx     (0)
        {
        }
        INT  nb_byte() {return _mflx->nbbyte().CKK_IntBasicLLO();}

        void new_line();
        void put_mem(INT);
        void open_block(INT nbb);
        void put_file(ostream &);

        virtual void close_block() = 0;
        virtual void init_dico_image(Data_Elise_PS_Disp *) = 0;
        virtual void init_dico_1l_image(Data_Elise_PS_Disp *) = 0;
        virtual void init_dico_imagemask(Data_Elise_PS_Disp *) = 0;

   protected :

        void init_top_flux(Packed_Flux_Of_Byte * flx)
        {
            _TopFlx = new MSBF_Flux_OutVarLI (new UnPacked_FOB(flx,true),true);
        }
        Mem_Packed_Flux_Of_Byte   * _mflx;

   private :

        MSBF_Flux_OutVarLI        * _TopFlx;
        INT                         _nbb;
};

void Ps_Filter::put_mem(INT v)
{
     _TopFlx->puti(v,_nbb); 
}

void Ps_Filter::new_line()
{
    _TopFlx->reset();
}


void Ps_Filter::open_block(INT nbb)
{
    _nbb = nbb;
    _mflx->reset();
}

void Ps_Filter::put_file(ostream & fd)
{
    PS_A85  p85(fd);

    for (INT k =0; k<_mflx->nbbyte().CKK_IntBasicLLO() ; k++)
        p85.put((*_mflx)[k]);
    p85.close_block();
}


      //================================
      //      Ps_LZW_Filter 
      //================================


class  Ps_LZW_Filter  : public Ps_Filter
{
       public :


        Ps_LZW_Filter() :
            Ps_Filter   (),
            _lzw        (   new Packed_LZW_Decompr_Flow
                            (    new UnPacked_FOB(_mflx,true)
                               , false
                               , true
                               , LZW_Protocols::tif
                               , 8
                            )
                        )
        {
             init_top_flux(_lzw);
        }

        private :

           virtual void close_block() ;
           virtual void init_dico_image(Data_Elise_PS_Disp *);
           virtual void init_dico_1l_image(Data_Elise_PS_Disp *);
           virtual void init_dico_imagemask(Data_Elise_PS_Disp *);

           Packed_LZW_Decompr_Flow   * _lzw;
};




void Ps_LZW_Filter::close_block()
{
    _lzw->reset();
}

void Ps_LZW_Filter::init_dico_image(Data_Elise_PS_Disp * psd)
{
       psd->_FLZW.put_prim(psd);
}

void Ps_LZW_Filter::init_dico_1l_image(Data_Elise_PS_Disp * psd)
{
    psd->_F1LZW.put_prim(psd);
}

void Ps_LZW_Filter::init_dico_imagemask(Data_Elise_PS_Disp * psd)
{
       psd->_MLZW.put_prim(psd);
}

      //================================
      //      Ps_PackBits_Filter 
      //================================

class  Ps_PackBits_Filter  : public Ps_Filter
{
       public :


        Ps_PackBits_Filter() :
            Ps_Filter  (),
            _pckb      (new Pack_Bits_Flow(_mflx,false,1024))
        {
             init_top_flux(_pckb);
        }

        private :

           virtual void close_block() ;
           virtual void init_dico_image(Data_Elise_PS_Disp * psd);
           virtual void init_dico_1l_image(Data_Elise_PS_Disp *);
           virtual void init_dico_imagemask(Data_Elise_PS_Disp *);

           Pack_Bits_Flow   * _pckb;
};



void Ps_PackBits_Filter::close_block()
{
    _pckb->reset();
}

void Ps_PackBits_Filter::init_dico_image(Data_Elise_PS_Disp * psd)
{
     psd->_FRLE.put_prim(psd);
}

void Ps_PackBits_Filter::init_dico_1l_image(Data_Elise_PS_Disp * psd)
{
     psd->_F1RLE.put_prim(psd);
}

void Ps_PackBits_Filter::init_dico_imagemask(Data_Elise_PS_Disp * psd)
{
     psd->_MRLE.put_prim(psd);
}

      //================================
      //      Ps_Ucompr_Filter 
      //================================

class  Ps_Ucompr_Filter  : public Ps_Filter
{
       public :


        Ps_Ucompr_Filter() :
            Ps_Filter  ()
        {
             init_top_flux(_mflx);
        }

        private :

           virtual void close_block() ;
           virtual void init_dico_image(Data_Elise_PS_Disp * psd);
           virtual void init_dico_1l_image(Data_Elise_PS_Disp *);
           virtual void init_dico_imagemask(Data_Elise_PS_Disp *);

};

void Ps_Ucompr_Filter::close_block()
{
}

void Ps_Ucompr_Filter::init_dico_image(Data_Elise_PS_Disp * psd)
{
     psd->_FUcomp.put_prim(psd);
}

void Ps_Ucompr_Filter::init_dico_1l_image(Data_Elise_PS_Disp * psd)
{
       psd->_F1Ucomp.put_prim(psd);
}

void Ps_Ucompr_Filter::init_dico_imagemask(Data_Elise_PS_Disp * psd)
{
     psd->_MUcomp.put_prim(psd);
}


      //================================
      //      Ps_Multi_Filter 
      //================================


bool Ps_Multi_Filter::put
(
     Data_Elise_PS_Win * w,
     Elise_Palette  p,
     U_INT1 ***    data,
     INT           dim_out,
     Pt2di         sz,
     Pt2di         p0,
     Pt2di         p1,
     bool          mask,
     INT           nb_byte_max
)
{
	INT f;

     if ((sz.x<= 0) || (sz.y <= 0))
        return true;

     w->set_active();
     _psd->set_active_palette(p,true);
     INT nbb =  mask ? 1 : 8;

     for ( f = 0 ; f < _nb_filter ; f++)
     {
          Ps_Filter * psf =  _filter[f];
          psf->open_block(nbb);

          for (INT y = 0 ; y < sz.y ; y++)
          {
              for (INT x = 0 ; x < sz.x ; x++)
                  for (INT d = 0; d < dim_out ; d++)
                  {
                      INT val =
                          mask                              ?
                          kth_bit_msbf( data[d][p1.y+y],p1.x+x)  :
                          data[d][p1.y+y][p1.x+x]           ;
                      psf->put_mem(val);
                  }
              psf->new_line();
          }
          psf->close_block();
     }

     Ps_Filter * psf =  _filter[0];
     for ( f = 1 ; f < _nb_filter ; f++)
        if  (_filter[f]->nb_byte() < psf->nb_byte())
            psf = _filter[f];

     if (
              (nb_byte_max >0)
           && (psf->nb_byte()>nb_byte_max)
        )
        return false;

     _psd->_MatIm.load_prim(_psd);
     _psd->_DicIm.load_prim(_psd);

     _psd->_nbbIm.put_prim(_psd,nbb);
     
     if (mask)
     {
        _psd->_x0Im.put_prim(_psd,p0.x);
        _psd->_y0Im.put_prim(_psd,p0.y);
        _psd->_fd << sz.x << " " << sz.y << " ";
         psf->init_dico_imagemask(_psd);
     }
     else
     {
         _psd->_tyIm.put_prim(_psd,sz.y);
         if (sz.y == 1)
         {
           _psd->_fd << sz.x << " " << p0.y << " " << p0.x << " ";
           psf->init_dico_1l_image(_psd);
         }
         else
         {
             _psd->_x0Im.put_prim(_psd,p0.x);
             _psd->_y0Im.put_prim(_psd,p0.y);
             _psd->_txIm.put_prim(_psd,sz.x);
             psf->init_dico_image(_psd);
         }
     }
    
     psf->put_file(_psd->_fd);

    return true;
}


bool Ps_Multi_Filter::put
(
     Data_Elise_PS_Win * w,
     Elise_Palette  p,
     U_INT1 ***    data,
     INT           dim_out,
     Pt2di         sz,
     Pt2di         p0,
     Pt2di         p1,
     INT           nb_byte_max
)
{
    return put(w,p,data,dim_out,sz,p0,p1,false,nb_byte_max);
}


bool Ps_Multi_Filter::put
(
     Data_Elise_PS_Win * w,
     Data_Col_Pal      * dcp,
     U_INT1 **          data,
     Pt2di               sz,
     Pt2di               p0,
     Pt2di               p1,
     INT                 nb_byte_max
)
{
    w->set_active();
    w->set_col(dcp);
    return put(w,dcp->pal(),&data,1,sz,p0,p1,true,nb_byte_max);
}




Ps_Multi_Filter::~Ps_Multi_Filter()
{
     for (INT f = 0 ; f < _nb_filter ; f++)
         delete _filter[f];
}

Ps_Multi_Filter::Ps_Multi_Filter
(
       Data_Elise_PS_Disp * psd
)  :
   _psd       (psd),
   _nb_filter (0)
{

     _filter[_nb_filter++] = new Ps_Ucompr_Filter;
     if (_psd->_use_pckb)
        _filter[_nb_filter++] = new Ps_PackBits_Filter;
     if (_psd->_use_lzw)
        _filter[_nb_filter++] = new Ps_LZW_Filter;
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
