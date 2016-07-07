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


Im2D_Bits<1> ImMarqueurCC(Pt2di aSz)
{
   Im2D_Bits<1> aMasq(aSz.x,aSz.y,1);
   ELISE_COPY(aMasq.border(1),0,aMasq.out());
   return aMasq;
}

void ResetMarqueur(TIm2DBits<1> & aMarq,const std::vector<Pt2di> & aVPts)
{
   for (int aKP=int(aVPts.size()) -1 ; aKP>=0 ;aKP--)
     aMarq.oset(aVPts[aKP],1);
}


template <const INT nbb,const bool msbf>  INT
         Tabul_Bits<nbb,msbf>::kieme_val(INT byte,INT k) const
{
    return input_tab[byte][k];
}

template <const INT nbb,const bool msbf>  INT
         Tabul_Bits<nbb,msbf>::set_kieme_val(INT old_byte,INT val,INT k) const
{
    return out_tab[old_byte][val][k];
}

template<const INT nbb,const bool msbf> void
           Tabul_Bits<nbb,msbf>::init_tabul()
{

     input_tab = new  tLineInputTab [256] ;
     out_tab = new tLineOutputTab[256];


     INT masq = nb_val-1;


     for (INT pos = 0 ; pos < nb_per_byte ; pos++)
     {
         INT num_in_byte = msbf ? (nb_per_byte -1-pos) : pos;
         INT pos_bits =  num_in_byte * nbb;
         INT masq_compl = 0xFF & (~(masq << pos_bits));

         for (INT byte = 0; byte < 256 ; byte ++)
         {
             input_tab[byte][pos] =  (byte >> pos_bits) & masq;
             for (INT val = 0; val< 1<<nbb; val++)
                 out_tab[byte][val][pos] =
                         (byte & masq_compl)
                       | (val << pos_bits);
         }
   }
}

template <const INT nbb,const bool msbf>
         Tabul_Bits<nbb,msbf>::Tabul_Bits(int )
{
}

 template <const INT nbb,const bool msbf>
 Tabul_Bits<nbb,msbf>::~Tabul_Bits()
 {

     delete[] input_tab ;
     delete[] out_tab ;
 }

template <const INT nbb,const bool msbf>  void
         Tabul_Bits<nbb,msbf>::input
         (
               INT * out,
               const U_INT1 * in,
               INT   x0,
               INT   x1
         )     const
{
    for (INT x = x0; x<x1 ; x++)
       *(out++) = input_tab[in[x/nb_per_byte]][x%nb_per_byte];
}

template <const INT nbb,const bool msbf>  void
         Tabul_Bits<nbb,msbf>::output
         (
               U_INT1 * out,
               const INT * in,
               INT   x0,
               INT   x1
         )     const
{
   U_INT1 * ox;

    for (INT x = x0; x<x1 ; x++,in++)
    {
       ox = out +x/nb_per_byte;
      *ox = out_tab[*ox][*in][x%nb_per_byte];
    }
}

template <const INT nbb,const bool msbf>  void
         Tabul_Bits<nbb,msbf>::output
         (
               U_INT1 * out,
               const REAL * in,
               INT   x0,
               INT   x1
         )     const
{
   U_INT1 * ox;

    for (INT x = x0; x<x1 ; x++,in++)
    {
       ox = out +x/nb_per_byte;
      *ox = out_tab[*ox][(INT)*in][x%nb_per_byte];
    }
}

template <const INT nbb,const bool msbf>  void
         Tabul_Bits<nbb,msbf>::output
         (
               U_INT1 * out,
               const _INT8 * in,
               INT   x0,
               INT   x1
         )     const
{
   U_INT1 * ox;

    for (INT x = x0; x<x1 ; x++,in++)
    {
       ox = out +x/nb_per_byte;
      *ox = out_tab[*ox][(INT)*in][x%nb_per_byte];
    }
}




template <const INT nbb,const bool msbf> INT
          Tabul_Bits<nbb,msbf>::sz_line(INT nb_el)
{
    return (nb_el+nb_per_byte-1)/nb_per_byte;
}



    //==============================
// static template class Tabul_Bits<1,true>;

void Tabul_Bits_Gen::init_tabul_bits()
{

     Tabul_Bits<1,true>::init_tabul();
     Tabul_Bits<1,false>::init_tabul();
     Tabul_Bits<2,true>::init_tabul();
     Tabul_Bits<2,false>::init_tabul();
     Tabul_Bits<4,true>::init_tabul();
     Tabul_Bits<4,false>::init_tabul();
}




void Tabul_Bits_Gen::unpack
     (
          U_INT1 *          out,
          const U_INT1 *    in,
          INT               nb,
          INT               nbb,
          bool              msbf
     )
{
    const Tabul_Bits_Gen & tbg = Tabul_Bits_Gen::tbb(nbb,msbf);

    static const INT sz_buf = 100;
    INT  buf[sz_buf];

    for (INT i=0; i<nb ; i+=sz_buf)
    {
        INT nb_loc = ElMin(nb-i,sz_buf);
        tbg.input(buf,in,i,i+nb_loc);
        convert(out+i,buf,nb_loc);
    }
}

void Tabul_Bits_Gen::pack
     (
          U_INT1 *          out,
          const U_INT1 *    in,
          INT               nb,
          INT               nbb,
          bool              msbf
     )
{
    const Tabul_Bits_Gen & tbg = Tabul_Bits_Gen::tbb(nbb,msbf);

    static const INT sz_buf = 100;
    INT  buf[sz_buf];

    for (INT i=0; i<nb ; i+=sz_buf)
    {
        INT nb_loc = ElMin(nb-i,sz_buf);
        convert(buf,in+i,nb_loc);
        tbg.output(out,buf,i,i+nb_loc);
    }
}


/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                                                               *****/
/*****   DataGenImBits<const INT nbb>                                *****/
/*****                                                               *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

template <const INT nbb> INT DataGenImBits<nbb>::sz_tot() const
{
    return _sz_tot;
}

template <const INT nbb> DataGenImBits<nbb>::DataGenImBits
                         (
                              INT sz_0,
                              INT sz_tot,
                              void * aDataLin
                         )
{
    mDataLinOwner = (aDataLin==0);
    _sz_line = Tabul_Bits<nbb,true>::sz_line(sz_0);
    sz_tot *= _sz_line;
    if (aDataLin==0)
    {
       if (sz_tot)
       {
         _data_lin =  STD_NEW_TAB_USER(sz_tot,U_INT1);
         MEM_RAZ(_data_lin,sz_tot);
       }
       else
         _data_lin = 0;
     }
     else
         _data_lin = (U_INT1 *) aDataLin;
    _sz_tot = sz_tot;
}

template <const INT nbb> void DataGenImBits<nbb>::SetAll(INT aV)
{
   INT aFlag = 0;

   for (INT aK = 0 ; aK< 8 ; aK+=nbb)
      aFlag |= (aV<<aK);
   memset(_data_lin,aV,_sz_tot);
}


template <const INT nbb> INT DataGenImBits<nbb>::vmin() const
{
    return 0;
}

template <const INT nbb> INT DataGenImBits<nbb>::vmax() const
{
    return Tabul_Bits<nbb,true>::nb_val;
}

template <const INT nbb> DataGenImBits<nbb>::~DataGenImBits()
{
     if (_data_lin && mDataLinOwner)
     {
         STD_DELETE_TAB_USER(_data_lin);
         _data_lin = 0;
     }
}

template <const INT nbb> bool  DataGenImBits<nbb>::integral_type() const
{
   return true;
}

     //=========================
     //  FATAL--ERRORS METHODS :
     //=========================

template <const INT nbb> void DataGenImBits<nbb>::striped_input_rle
                         (void *,INT,INT,const void*,INT) const
{
    elise_internal_error
    (
        "no DataGenImBits<nbb>::striped_input_rle defined",
        __FILE__,__LINE__
    );
}

template <const INT nbb> void DataGenImBits<nbb>::striped_output_rle
                         (void *,INT,INT,const void*,INT) const
{
    elise_internal_error
    (
        "no DataGenImBits<nbb>::striped_output_rle defined",
        __FILE__,__LINE__
    );
}

template <const INT nbb> int DataGenImBits<nbb>::sz_el() const
{
    elise_internal_error
    (
        "no DataGenImBits<nbb>::sz_el defined",
        __FILE__,__LINE__
    );
     return -12345;
}

template <const INT nbb> int DataGenImBits<nbb>::sz_base_el() const
{
    elise_internal_error
    (
        "no DataGenImBits<nbb>::sz_base_el defined",
        __FILE__,__LINE__
    );
     return -12345;
}

template <const INT nbb> void * DataGenImBits<nbb>::data_lin_gen()
{
     return _data_lin;
}




/*
virtual void  out_rle(void *,INT,const INT*,INT offs_0) const;
virtual void  out_rle(void *,INT,const REAL*,INT offs_0) const;
*/



/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                                                               *****/
/*****   DataIm2D_Bits<const INT nbb>                                *****/
/*****                                                               *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/


template <const INT nbb>
         INT DataIm2D_Bits<nbb>::get(INT x,INT y) const
{
    return Tabul_Bits<nbb,true>::input_tab
                     [_data[y][x/nb_per_byte]]
                     [x%nb_per_byte]
                  ;
}

template <const INT nbb>
         INT DataIm2D_Bits<nbb>::get_def(INT x,INT y,INT v) const
{
    return  ((x>=0) && (y>=0) && (x<mTx) && (y<mTy)) ?
            get(x,y)                                 :
            v                                        ;
}

// Greg: deplace dans le header pour cause d'erreur de compilation sous MacOS
/*
template <const INT nbb>
         void DataIm2D_Bits<nbb>::set(INT x,INT y,INT val) const
{
    U_INT1 * adr_x = _data[y] +  x / nb_per_byte;
    *adr_x =  Tabul_Bits<nbb,true>::out_tab[*adr_x][val][x%nb_per_byte];
}
*/

template <const INT nbb> void DataIm2D_Bits<nbb>::out_pts_integer
              (Const_INT_PP pts,INT nb,const void * i)
{
   const INT * tx = pts[0];
   const INT * ty = pts[1];
   const INT * in =  C_CAST(const INT *,i);


   for (int j=0 ; j<nb ; j++)
       set(tx[j],ty[j],in[j]);
}

template <const INT nbb> void DataIm2D_Bits<nbb>::input_pts_integer
              (void * o,Const_INT_PP pts,INT nb) const
{
   const INT * tx = pts[0];
   const INT * ty = pts[1];
   INT  * out =  C_CAST(INT *,o);


   for (int i=0 ; i<nb ; i++)
       out[i] = get(tx[i],ty[i]);
}

template <const INT nbb> void DataIm2D_Bits<nbb>::input_pts_reel
              (REAL * out,Const_REAL_PP pts,INT nb) const
{
   const REAL * tx = pts[0];
   const REAL * ty = pts[1];

   REAL x,y;
   REAL p_0x,p_1x,p_0y,p_1y;
   INT xi,yi;

   for (int i=0 ; i<nb ; i++)
   {
       x = tx[i];
       y = ty[i];
       p_1x = x - (xi= (INT) x);
       p_1y = y - (yi= (INT) y);
       p_0x = 1.0-p_1x;
       p_0y = 1.0-p_1y;

       out[i] =
                 p_0x * p_0y * get(xi    ,  yi  )
               + p_1x * p_0y * get(xi+1  ,  yi  )
               + p_0x * p_1y * get(xi    ,  yi+1)
               + p_1x * p_1y * get(xi+1  ,  yi+1);
   }
}

template <const INT nbb> void * DataIm2D_Bits<nbb>::calc_adr_seg(INT * pts)
{
    return _data[pts[1]] ;
}



template <const INT nbb> DataIm2D_Bits<nbb>::~DataIm2D_Bits()
{
     ASSERT_INTERNAL(_data != 0,"multiple deletion of a bitmap");

     STD_DELETE_TAB_USER(_data);
     _data = 0;
}


template <const INT nbb> const INT *  DataIm2D_Bits<nbb>::p0() const
{
    return PTS_00000000000000;
}

template <const INT nbb> const INT *  DataIm2D_Bits<nbb>::p1() const
{
    return _txy;
}

template <const INT nbb>  INT  DataIm2D_Bits<nbb>::dim() const
{
    return 2;
}

// Deplace dans le header pour pb de correlation MacOS
/*
template <const INT nbb>  DataIm2D_Bits<nbb>::DataIm2D_Bits
                          (
                               INT Tx,
                               INT Ty,
                               bool to_init ,
                               INT  v_init,
                               void * aDataLin
                          ) :
     DataGenImBits<nbb>(Tx,Ty,aDataLin),
     DataIm2DGen(Tx,Ty)
{
     _data =  STD_NEW_TAB_USER(ty(),U_INT1 *);

     for (int y = 0 ; y<ty() ; y++)
         _data[y] = this->_data_lin + y *this->_sz_line;

    if (to_init)
    {
         Tjs_El_User.ElAssert
         (
            (v_init<this->vmax()) && (v_init>=this->vmin()),
            EEM0 << "Bad init value in Im2D_Bits \n"
                 << "|  Got " << v_init << ", expected >= " << this->vmin() << " and < " << this->vmax()
         );
         for (INT b=0; b<nb_per_byte ; b++)
             set(b,0,v_init);
         U_INT1  v0 = this->_data_lin[0];
         set_cste(this->_data_lin,v0,this->_sz_line*ty());
    }
}
 */



template <const INT nbb>  void DataIm2D_Bits<nbb>::out_assoc
         (
                  void * out, // eventually 0
                  const OperAssocMixte & op,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
         )
         const
{
      const INT * v = (INT *) const_cast<void *>(values);
      INT * o =  (INT *) out;
      const INT * x    = coord[0];
      const INT * y    = coord[1];


      for (INT i=0; i < nb ; i++)
      {
          o[i] = get(x[i],y[i]);
          set(x[i],y[i],op.opel(o[i],v[i]));
      }

      verif_value_op_ass
      (
            op,
            o,
            v,
            nb,
            0,
            (INT) Tabul_Bits<nbb,true>::nb_val
      );
}

template  <const int nbb>
          Im2D_U_INT1  DataIm2D_Bits<nbb>::gray_im_red(INT & zoom)
{
     if (zoom < nb_per_byte)
     {
         Im2D_Bits<nbb> ib(this);
         Im2D_U_INT1 res((tx()+zoom-1)/zoom,(ty()+zoom-1)/zoom,0);
         ELISE_COPY
         (
             ib.all_pts(),
             (ib.in() *255)/((nb_val -1)*zoom*zoom),
             res.histo().chc(Virgule(FX,FY)/zoom)
         );
         return res;
     }
     zoom = round_ni(zoom / (REAL) nb_per_byte) * nb_per_byte;
     zoom = ElMax(1,zoom);
     INT txz = tx()/zoom;
     INT tyz = ty()/zoom;

     Im2D_U_INT1 res(txz,tyz,0);
     U_INT1 ** _dres = res.data();
     INT nb_octet_pixel = zoom/nb_per_byte;

     Im1D_INT4  lut(256);
     INT4 * _dlut = lut.data();
     {
         Im2D_Bits<nbb> tmp(nbb,1);
         U_INT1 ** dtmp = tmp.data();
         for (INT k=0; k<256; k++)
         {
             dtmp[0][0]  = k;
             _dlut[k] = 0;
             for (INT x=0; x<nb_per_byte; x++)
                 _dlut[k] += (tmp.get(x,0) * 255) / (nb_val -1);
         }
     }

     for (INT yz=0; yz< tyz; yz++)
     {
          for (int xz=0; xz<txz ; xz++)
          {
              INT som=0;
              for (INT y=0; y<zoom; y++)
              {
                   U_INT1 * l =  _data[yz*zoom+y]+xz*nb_octet_pixel;
                   for (INT x=0; x<nb_octet_pixel; x++)
                       som += _dlut[l[x]];
              }
              _dres[yz][xz] = som/(zoom*zoom);
          }
     }
     return res;
}

template <const INT nbb>  void DataIm2D_Bits<nbb>::SetAll(INT aV)
{
    DataGenImBits<nbb>::SetAll(aV);
}

template <const INT nbb>  void DataIm2D_Bits<nbb>::q_dilate
                   (  Std_Pack_Of_Pts<INT> * set_dilated,
                      char **                    is_neigh,
                      const Std_Pack_Of_Pts<INT> * set_to_dilate,
                      INT ** neigh,
                      INT   nb_v,
                      Image_Lut_1D_Compile   func_selection,
                      Image_Lut_1D_Compile   func_update
                   )
{

   INT * x_in  = set_to_dilate->_pts[0];
   INT * y_in  = set_to_dilate->_pts[1];
   INT * x_out  = set_dilated->_pts[0];
   INT * y_out  = set_dilated->_pts[1];

   INT nb_in = set_to_dilate->nb();
   INT nb_out = 0;
   //INT szb_out = set_dilated->pck_sz_buf();  TRES MAUVAIS IDEE DE COMMENTER 
   set_dilated->pck_sz_buf();

   INT * x_neigh = neigh[0];
   INT * y_neigh = neigh[1];

   INT i,d;
   INT xv,yv,xo,yo;

   for (d=0; d<nb_v ; d++)
   {
       xv = x_neigh[d];
       yv = y_neigh[d];
       for (i=0; i<nb_in; i++)
       {
            xo = x_in[i]+xv;
            yo = y_in[i]+yv;
            if (El_User_Dyn.active())
            {
               if ((xo<0) || (xo>=_txy[0]) || (yo<0) || (yo>=_txy[1]))
                  elise_fatal_error
                  (  "out of bitmap in dilate spec Image",
                     __FILE__,__LINE__);
               INT v = get(xo,yo);
               if  ((v <func_selection._b1)||(v >=func_selection._b2))
                  elise_fatal_error
                  (  "image out of lut range in  dilate spec Image",
                     __FILE__,__LINE__);
               if (func_selection._l[v])
               {
                    if ((v <func_update._b1)||(v >=func_update._b2))
                    {
                        elise_fatal_error
                        (  "image out of lut range in  dilate spec Image",
                           __FILE__,__LINE__);
                    }
                    INT uv = func_update._l[v];
                    if  ((uv <func_selection._b1)||(uv >=func_selection._b2))
                       elise_fatal_error
                    ( "image out of lut range in  dilate spec Image (after update)",
                          __FILE__,__LINE__);
                    if (func_selection._l[uv])
                    {
                        elise_fatal_error
                        (  "update does not supress selection in dilate spec Image",
                           __FILE__,__LINE__);
                    }
               }
            }
            int isnei = (func_selection._l[get(xo,yo)]);
            if (isnei)
            {
                 set(xo,yo,func_update._l[get(xo,yo)]);
                ASSERT_INTERNAL
                (      nb_out < szb_out,
                       "outside Pack_Pts limits in dilate spec Image"
                );
                x_out[nb_out]   = xo;
                y_out[nb_out++] = yo;
            }
            if (is_neigh)
               is_neigh[d][i] = isnei;
       }
   }

   set_dilated->set_nb(nb_out);
}



/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*****                                                               *****/
/*****                                                               *****/
/*****   DataIm2D_Bits<const INT nbb>                                *****/
/*****                                                               *****/
/*****                                                               *****/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

template  <const int nbb> Seg2d Im2D_Bits<nbb>::OptimizeSegTournantSomIm
                          (
                              REAL &                 score,
                              Seg2d                  seg,
                              INT                    NbPts,
                              REAL                   step_init,
                              REAL                   step_limite,
                              bool                   optim_absc,
                              bool                   optim_teta,
                              bool *                 FreelyOpt
                          )
{
    ELISE_ASSERT(false,"No OptimizeSegTournantSomIm for required type");
    return Seg2d(Pt2dr(0,0),Pt2dr(0,0));
}




template  <const int nbb> Im2D_Bits<nbb>::Im2D_Bits(DataIm2D_Bits<nbb>*d) :
    Im2DGen(d)
{
}

template  <const int nbb> Im2D_Bits<nbb>::Im2D_Bits
                          (
                              Im2D_BitsIntitDataLin,
                              INT tx,
                              INT ty,
                              void * aDataLin
                          ) :
        Im2DGen(new DataIm2D_Bits<nbb>(tx,ty,false,0,aDataLin))
{
}





template  <const int nbb> Im2D_Bits<nbb>::Im2D_Bits(INT tx,INT ty,INT v_init) :
        Im2DGen(new DataIm2D_Bits<nbb>(tx,ty,true,v_init,0))
{
}

template  <const int nbb> Im2D_Bits<nbb>::Im2D_Bits(Pt2di pt,INT v_init) :
Im2DGen(new DataIm2D_Bits<nbb>(pt.x,pt.y,true,v_init,0))
{
}

template  <const int nbb> DataIm2D_Bits<nbb> * Im2D_Bits<nbb>::didb() const
{
   return (DataIm2D_Bits<nbb> *) _ptr;
}

template  <const int nbb> int Im2D_Bits<nbb>::tx() const
{
      return didb()->tx();
}

template  <const int nbb> int Im2D_Bits<nbb>::ty() const
{
      return didb()->ty();
}


template  <const int nbb> INT Im2D_Bits<nbb>::vmax() const
{
   return  didb()->vmax();
}

template  <const int nbb> U_INT1 ** Im2D_Bits<nbb>::data()
{
   return  didb()->_data;
}
template  <const int nbb> U_INT1 ** Im2D_Bits<nbb>::data() const
{
   return  didb()->_data;
}



template  <const int nbb> INT Im2D_Bits<nbb>::get(INT x,INT y)  const
{
   return  didb()->get(x,y);
}

template  <const int nbb> INT Im2D_Bits<nbb>::get_def(INT x,INT y,INT v)  const
{
   return  didb()->get_def(x,y,v);
}

template  <const int nbb> void Im2D_Bits<nbb>::set(INT x,INT y,INT v)
{
   didb()->set(x,y,v);
}

template  <const int nbb> INT Im2D_Bits<nbb>::GetI(const Pt2di & aP ) const
{
   return get(aP.x,aP.y);
}

template  <const int nbb> double Im2D_Bits<nbb>::Val(const int & x,const int &y) const
{
   return get(x,y);
}

template  <const int nbb> void  Im2D_Bits<nbb>::SetI(const Pt2di  &aP,int aVal )
{
   AssertInside(aP);
   set(aP.x,aP.y,aVal);
   // data()[aP.y][aP.x] = aVal;
}
template  <const int nbb> void  Im2D_Bits<nbb>::SetR(const Pt2di  &aP,double aVal )
{
   AssertInside(aP);
   set(aP.x,aP.y,round_ni(aVal));
   // data()[aP.y][aP.x] = round_ni(aVal);
}



template  <const int nbb> double Im2D_Bits<nbb>::GetR(const Pt2di & aP ) const
{
   return get(aP.x,aP.y);
}

template <const int nbb>
Im2DGen *  Im2D_Bits<nbb>::ImOfSameType(const Pt2di & aSz) const
{
   return new Im2D_Bits<nbb>(aSz.x,aSz.y,0);
}



template  <const int nbb> void Im2D_Bits<nbb>::SetAll(INT aV)
{
   didb()->SetAll(aV);
}


template  <const int nbb> Im2D_U_INT1  Im2D_Bits<nbb>::gray_im_red(INT & zoom)
{
    return didb()->gray_im_red(zoom);
}

#if ElTemplateInstantiation
#endif

#define  Declare_TBB(NBB,MSBF)\
template <> Tabul_Bits<NBB,MSBF> Tabul_Bits<NBB,MSBF>::The_Only_One(123456);\
template <> Tabul_Bits<NBB,MSBF>::tLineInputTab *  Tabul_Bits<NBB,MSBF>::input_tab=0;\
template <> Tabul_Bits<NBB,MSBF>::tLineOutputTab *  Tabul_Bits<NBB,MSBF>::out_tab=0;

Declare_TBB(1,true)
Declare_TBB(1,false)
Declare_TBB(2,true)
Declare_TBB(2,false)
Declare_TBB(4,true)
Declare_TBB(4,false)

template <> GenIm::type_el DataGenImBits<1>::type_el_bitm = GenIm::bits1_msbf;
template <> GenIm::type_el DataGenImBits<2>::type_el_bitm = GenIm::bits2_msbf;
template <> GenIm::type_el DataGenImBits<4>::type_el_bitm = GenIm::bits4_msbf;

const Tabul_Bits_Gen & Tabul_Bits_Gen::tbb(INT nbb,bool msbf)
{
    switch (nbb)
    {
        case   1 :
            if  (msbf)
                return Tabul_Bits<1,true>::The_Only_One;
            else
                return Tabul_Bits<1,false>::The_Only_One;
            
        case   2 :
            if  (msbf)
                return Tabul_Bits<2,true>::The_Only_One;
            else
                return Tabul_Bits<2,false>::The_Only_One;
            
        case   4 :
            if  (msbf)
                return Tabul_Bits<4,true>::The_Only_One;
            else
                return Tabul_Bits<4,false>::The_Only_One;
    };
    
    elise_internal_error("Tabul_Bits_Gen::tbb",__FILE__,__LINE__);
    
    return  Tabul_Bits<1,false>::The_Only_One;
}

template class Im2D_Bits<1>;
template class Im2D_Bits<2>;
template class Im2D_Bits<4>;

template class DataGenImBits<1>;
template class DataGenImBits<2>;
template class DataGenImBits<4>;

template class DataIm2D_Bits<1>;
template class DataIm2D_Bits<2>;
template class DataIm2D_Bits<4>;

#if (0)
// template <> int cTestTPL<int>::theTab[4] ={0,1,2,3};
template <> Tabul_Bits<1,true> Tabul_Bits<1,true>::The_Only_One;
template <> U_INT1  Tabul_Bits<1,true>::input_tab[256][8];
template <> U_INT1  Tabul_Bits<1,true>::out_tab[256][2][8];

    //==============================

template <> Tabul_Bits<1,false> Tabul_Bits<1,false>::The_Only_One;
template <> U_INT1  Tabul_Bits<1,false>::input_tab[256][8];
template <> U_INT1  Tabul_Bits<1,false>::out_tab[256][2][8];

    //==============================

template <> Tabul_Bits<2,true> Tabul_Bits<2,true>::The_Only_One;
template <> U_INT1  Tabul_Bits<2,true>::input_tab[256][4];
template <> U_INT1  Tabul_Bits<2,true>::out_tab[256][4][4];

    //==============================

template <> Tabul_Bits<2,false> Tabul_Bits<2,false>::The_Only_One;
template <> U_INT1  Tabul_Bits<2,false>::input_tab[256][4];
template <> U_INT1  Tabul_Bits<2,false>::out_tab[256][4][4];

    //==============================

template <> Tabul_Bits<4,true> Tabul_Bits<4,true>::The_Only_One;
template <> U_INT1  Tabul_Bits<4,true>::input_tab[256][2];
template <> U_INT1  Tabul_Bits<4,true>::out_tab[256][16][2];

    //==============================

template <> Tabul_Bits<4,false> Tabul_Bits<4,false>::The_Only_One;
template <> U_INT1  Tabul_Bits<4,false>::input_tab[256][2];
template <> U_INT1  Tabul_Bits<4,false>::out_tab[256][16][2];
#endif

    //==============================

Im2D_Bits<1> MasqFromFile(const std::string & aName)
{
  Tiff_Im aTif(aName.c_str());
  Pt2di aSz = aTif.sz();
  Im2D_Bits<1> aRes(aSz.x,aSz.y);
  ELISE_COPY(aTif.all_pts(),aTif.in(),aRes.out());
  return aRes;
}


Im2D_Bits<1> MasqFromFile(const std::string & aName,const Pt2di & aSz)
{
   if (ELISE_fp::exist_file(aName))
      return MasqFromFile(aName);
   return Im2D_Bits<1>(aSz.x,aSz.y,1);
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
