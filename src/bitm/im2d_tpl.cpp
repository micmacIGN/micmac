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



cFoncI2D::~cFoncI2D()
{
}

Im2DGen gray_im_red_centred(Im2DGen aI)
{
    Im2DGen anIConv = D2alloc_im2d(aI.TypeEl(),aI.tx(),aI.ty());
    Im2D_REAL8 aMasq
    (
           3,3,
           " 1 2 1 "
           " 2 4 2 "
           " 1 2 1 "
    );

    ELISE_COPY
    (
         aI.all_pts(),
         som_masq(aI.in_proj(),aMasq) /16,
         anIConv.out()
    );

    Im2DGen aRes = D2alloc_im2d(aI.TypeEl(),aI.tx()/2,aI.ty()/2);
    ELISE_COPY
    (
        aRes.all_pts(),
    anIConv.in_proj()[Virgule(FX,FY)*2],
    aRes.out()
    );
    return aRes;
}

Tiff_Im gray_file_red_centred(Tiff_Im aTif,const std::string & aName)
{
    Im2DGen aI = aTif.ReadIm();
    Im2DGen aI2 = gray_im_red_centred(aI);
    return Tiff_Im::CreateFromIm(aI2,aName);
}


 // Handle "only" a dimension max of 30 for bitmaps.

INT PTS_00000000000000[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

template <class TyBase>
         void verif_value_op_ass
              (
                  const OperAssocMixte & op,
                  const TyBase *         pre_out,
                  const TyBase *         values,
                  INT                    nb0,
                  TyBase                 v_min,
                  TyBase                 v_max
              )
{
    if ((El_User_Dyn.active()) && (v_max != v_min))
    {
         static const INT nb_buf = 100;
         TyBase v[nb_buf];
         for (INT i=0; i<nb0 ; i+=nb_buf )
         {
             INT nb_loc = ElMin(nb_buf,nb0-i);
             op.t0_eg_t1_op_t2(v,pre_out+i,values+i,nb_loc);
             INT index =    index_values_out_of_range(v,nb_loc,v_min,v_max);
             El_User_Dyn.ElAssert
             (
                 index == INDEX_NOT_FOUND,
                 EEM0
                    << "values out of range in bitmap-assoc\n"
                    << "|   " << pre_out[i+index]
                    << " o "  << values[i+index]
                    << " => " << v[index] << "\n"
                    << "|  interval = [" <<  v_min
                    << " --- " << v_max << "["
             );
         }
    }
}

/***********************************************************************/
/*                                                                     */
/*       cTpIm2DInter                                                 */
/*                                                                     */
/***********************************************************************/

template <class Type,class TyBase>
         class cTpIm2DInter : public  cIm2DInter
{
    public :

      double Get(const Pt2dr & aP)
      {
          return mInt->GetVal(mD,aP);
      }

      double GetDef(const Pt2dr & aP,double aVal)
      {
          if ( (aP.x<mX0)||(aP.y<mY0)||(aP.x>=mX1)||(aP.y>=mY1))
             return aVal;
          return mInt->GetVal(mD,aP);
      }

      int  SzKernel() const
      {
          return mInt->SzKernel();
      }

      cTpIm2DInter
      (
         Im2D<Type,TyBase>  anIm,
         const cInterpolateurIm2D<Type> * anInterp
      ) :
        mIm  (anIm),
        mInt (anInterp),
        mX0  (mInt->SzKernel()),
        mY0  (mInt->SzKernel()),
        mX1  (mIm.tx()-mInt->SzKernel()),
        mY1  (mIm.ty()-mInt->SzKernel()),
        mD   (mIm.data())
      {
      }
    private :
      Im2D<Type,TyBase>                    mIm;
      const cInterpolateurIm2D<Type> *     mInt;

      int                                  mX0;
      int                                  mY0;
      int                                  mX1;
      int                                  mY1;

      Type**                               mD;
};

/***********************************************************************/
/*                                                                     */
/*       DataGenImType                                                 */
/*                                                                     */
/***********************************************************************/

template <class Type,class TyBase>
         int DataGenImType<Type,TyBase>::sz_tot() const
{
    return _sz_tot;
}

template <class Type,class TyBase>  void  DataGenImType<Type,TyBase>::raz()
{
    MEM_RAZ(_data_lin,_sz_tot);
}


template <class Type,class TyBase>
         GenIm::type_el DataGenImType<Type,TyBase>::type() const
{
     return type_el_bitm;
}


template <class Type,class TyBase>
        INT DataGenImType<Type,TyBase>::vmax() const
{
     return v_max;
}

template <class Type,class TyBase>
        INT DataGenImType<Type,TyBase>::vmin() const
{
     return v_min;
}



template <class Type,class TyBase>
        void DataGenImType<Type,TyBase>::out_rle(void * v,INT nb,const REAL16* i,INT offs_0) const
{
   ELISE_ASSERT(false,"::out_rle");
}


template <class Type,class TyBase>
        void DataGenImType<Type,TyBase>::out_rle(void * v,INT nb,const INT* i,INT offs_0) const
{
     convert(C_CAST(Type *,v) + offs_0,i,nb);
}

template <class Type,class TyBase>
        void DataGenImType<Type,TyBase>::out_rle(void * v,INT nb,const _INT8* i,INT offs_0) const
{
     convert(C_CAST(Type *,v) + offs_0,i,nb);
}


template <class Type,class TyBase>
        void DataGenImType<Type,TyBase>::out_rle(void * v,INT nb,const REAL * i,INT offs_0) const
{
     convert(C_CAST(Type *,v)+offs_0,i,nb);
}


template <class Type,class TyBase>  void
         DataGenImType<Type,TyBase>::void_input_rle
              (void * v_out,INT nb,const void* v_in,INT offs_0) const
{
     convert(C_CAST(TyBase *,v_out),C_CAST(const Type *,v_in) + offs_0,nb);
}

template <class Type,class TyBase>  void
         DataGenImType<Type,TyBase>::int8_input_rle
              (_INT8 * v_out,INT nb,const void* v_in,INT offs_0) const
{
     convert(v_out,C_CAST(const Type *,v_in) + offs_0,nb);
}




template <class Type,class TyBase>  void
         DataGenImType<Type,TyBase>::striped_input_rle
            (void * v_out,INT nb,INT dim,const void* v_in,INT offs_0) const
{
     // when dim ==1, this in fact not striped and convertion will be faster
     if (dim == 1)
        void_input_rle(*((TyBase **)v_out),nb,v_in,offs_0);
     else
     {
          TyBase ** out = C_CAST(TyBase **,v_out);
          Type   * in = C_CAST(Type *,const_cast<void *>(v_in)) +offs_0*dim;

          int i_in =0;
          for(int x=0; x<nb ; x++)
             for (int d=0 ; d < dim ; d++)
                 out[d][x] =  in[i_in++];
     }
}

template <class Type,class TyBase>  void
         DataGenImType<Type,TyBase>::striped_output_rle
            (void * v_out,INT nb,INT dim,const void* v_in,INT offs_0) const
{
     // when dim ==1, this in fact not striped and convertion will be faster
     if (dim == 1)
        out_rle(v_out,nb,*((TyBase **)const_cast<void *>(v_in)),offs_0);
     else
     {
          TyBase ** in = C_CAST(TyBase **,const_cast<void *>(v_in));
          Type   * out = C_CAST(Type *,v_out) + dim*offs_0;

          int i_in =0;
          for(int x=0; x<nb ; x++)
             for (int d=0 ; d < dim ; d++)
                 out[i_in++] =  (Type) in[d][x];
     }
}

template <class Type,class TyBase>
         DataGenImType<Type,TyBase>::DataGenImType
         (
              INT sz_tot,
              bool to_init,
              TyBase v_init,
              const char * str_init
         )
{
   Initializer(sz_tot,to_init,v_init,str_init);
}

template <class Type,class TyBase>
         void DataGenImType<Type,TyBase>::Initializer
         (
              INT sz_tot,
              bool to_init,
              TyBase v_init,
              const char * str_init
         )
{
    mSzMemory = sz_tot;
    _sz_tot = sz_tot;
    if (sz_tot)
    {
      _data_lin =  STD_NEW_TAB_USER(sz_tot,Type);
// std::cout << "ALLOC " << sz_tot << "\n";
      if (to_init)
      {
          if (v_max != v_min)
          {
              Tjs_El_User.ElAssert
              (
                   (v_init>= v_min) && (v_init < v_max),
                   EEM0 << "Out of range in bitmap initialisation\n"
                        << "got " << v_init << " was  expecting in ["
                        << v_min << " " << v_max << "[\n"
              );
          }
          if (v_init == 0)
             MEM_RAZ(_data_lin,sz_tot);
          else
             set_cste(_data_lin,(Type) v_init,sz_tot);
      }
      else if (str_init)
      {
           const char * str = str_init;
           for(int k =0; k<sz_tot ;k++)
           {
               double d;
               while ((*str == ' ') && *str) str++;
               Tjs_El_User.ElAssert
               (
                    sscanf(str,"%lf",&d) == 1,
                    EEM0 << "Bad format in string image initilization"
               );
               while ((*str != ' ') && *str) str++;
               _data_lin[k] = (Type) d;
           }
      }
    }
    else
      _data_lin = 0;
    _to_free = true;
}

template <class Type,class TyBase>
         DataGenImType<Type,TyBase>::~DataGenImType()
{
     Desinitializer();
}

template <class Type,class TyBase>
         void DataGenImType<Type,TyBase>::Desinitializer()
{
       if ((_data_lin) && _to_free)
       {
            STD_DELETE_TAB_USER(_data_lin);
            _data_lin = 0;
       }
}

template <class Type,class TyBase>
         void DataGenImType<Type,TyBase>::Resize(INT aSz)
{
   ELISE_ASSERT((aSz>0),"Bad Size in DataGenImType<Type,TyBase>::Resize");
   if (aSz > mSzMemory)
   {
       Desinitializer();
       Initializer(aSz,false,0);
   }
   else
      _sz_tot = aSz;
}




template <class Type,class TyBase>  INT DataGenImType<Type,TyBase>::sz_el() const
{
        return sizeof(Type);
}

template <class Type,class TyBase>  INT DataGenImType<Type,TyBase>::sz_base_el() const
{
        return sizeof(TyBase);
}



template <class Type,class TyBase>  bool DataGenImType<Type,TyBase>::integral_type() const
{
        return _integral_type;
}


template <class Type,class TyBase> void  *
         DataGenImType<Type,TyBase>::data_lin_gen()
{
    return _data_lin;
}

template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<U_INT1,INT>::type_el_bitm = GenIm::u_int1;
template <> CONST_STAT_TPL INT DataGenImType<U_INT1,INT>::v_max = 1<<8;
template <> CONST_STAT_TPL INT DataGenImType<U_INT1,INT>::v_min = 0;
template <> CONST_STAT_TPL bool DataGenImType<U_INT1,INT>::_integral_type = true;
template <> CONST_STAT_TPL DataIm1D<U_INT1,INT> DataIm1D<U_INT1,INT>::The_Bitm =  DataIm1D<U_INT1,INT>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<INT1,INT>::type_el_bitm = GenIm::int1;
template <> CONST_STAT_TPL INT DataGenImType<INT1,INT>::v_max = 1<<7;
template <> CONST_STAT_TPL INT DataGenImType<INT1,INT>::v_min = -(1<<7);
template <> CONST_STAT_TPL bool DataGenImType<INT1,INT>::_integral_type = true;
template <> CONST_STAT_TPL DataIm1D<INT1,INT> DataIm1D<INT1,INT>::The_Bitm =  DataIm1D<INT1,INT>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<U_INT2,INT>::type_el_bitm = GenIm::u_int2;
template <> CONST_STAT_TPL INT DataGenImType<U_INT2,INT>::v_max = 1<<16;
template <> CONST_STAT_TPL INT DataGenImType<U_INT2,INT>::v_min = 0;
template <> CONST_STAT_TPL bool DataGenImType<U_INT2,INT>::_integral_type = true;
template <> CONST_STAT_TPL DataIm1D<U_INT2,INT> DataIm1D<U_INT2,INT>::The_Bitm =  DataIm1D<U_INT2,INT>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<INT2,INT>::type_el_bitm = GenIm::int2;
template <> CONST_STAT_TPL INT DataGenImType<INT2,INT>::v_max = 1<<15;
template <> CONST_STAT_TPL INT DataGenImType<INT2,INT>::v_min = -(1<<15);
template <> CONST_STAT_TPL bool DataGenImType<INT2,INT>::_integral_type = true;
template <> CONST_STAT_TPL DataIm1D<INT2,INT> DataIm1D<INT2,INT>::The_Bitm =  DataIm1D<INT2,INT>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<INT,INT>::type_el_bitm = GenIm::int4;
template <> CONST_STAT_TPL INT DataGenImType<INT,INT>::v_max = 1<<30;
template <> CONST_STAT_TPL INT DataGenImType<INT,INT>::v_min = 1<<30;
template <> CONST_STAT_TPL bool DataGenImType<INT,INT>::_integral_type = true;
template <> CONST_STAT_TPL DataIm1D<INT,INT> DataIm1D<INT,INT>::The_Bitm =  DataIm1D<INT,INT>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<U_INT4,_INT8>::type_el_bitm = GenIm::u_int4;
template <> CONST_STAT_TPL INT DataGenImType<U_INT4,_INT8>::v_max = (1<<31);
template <> CONST_STAT_TPL INT DataGenImType<U_INT4,_INT8>::v_min = 0;
template <> CONST_STAT_TPL bool DataGenImType<U_INT4,_INT8>::_integral_type = true;
// template <> CONST_STAT_TPL DataIm1D<INT,INT> DataIm1D<U_INT4,_INT8>::The_Bitm =  DataIm1D<U_INT4,_INT8>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<REAL4,REAL8>::type_el_bitm = GenIm::real4;
template <> CONST_STAT_TPL INT  DataGenImType<REAL4,REAL8>::v_max = 0;
template <> CONST_STAT_TPL INT  DataGenImType<REAL4,REAL8>::v_min = 0;
template <> CONST_STAT_TPL bool DataGenImType<REAL4,REAL8>::_integral_type = false;
template <> CONST_STAT_TPL DataIm1D<REAL4,REAL8> DataIm1D<REAL4,REAL8>::The_Bitm =  DataIm1D<REAL4,REAL8>(0,0,0,0);

template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<REAL8,REAL8>::type_el_bitm = GenIm::real8;
template <> CONST_STAT_TPL INT DataGenImType<REAL8,REAL8>::v_max = 0;
template <> CONST_STAT_TPL INT DataGenImType<REAL8,REAL8>::v_min = 0;
template <> CONST_STAT_TPL bool DataGenImType<REAL8,REAL8>::_integral_type = false;
template <> CONST_STAT_TPL DataIm1D<REAL8,REAL8> DataIm1D<REAL8,REAL8>::The_Bitm =  DataIm1D<REAL8,REAL8>(0,0,0,0);


template <> CONST_STAT_TPL  GenIm::type_el DataGenImType<REAL16,REAL16>::type_el_bitm = GenIm::real16;
template <> CONST_STAT_TPL INT DataGenImType<REAL16,REAL16>::v_max = 0;
template <> CONST_STAT_TPL INT DataGenImType<REAL16,REAL16>::v_min = 0;
template <> CONST_STAT_TPL bool DataGenImType<REAL16,REAL16>::_integral_type = false;
template <> CONST_STAT_TPL DataIm1D<REAL16,REAL16> DataIm1D<REAL16,REAL16>::The_Bitm =  DataIm1D<REAL16,REAL16>(0,0,0,0);






/***********************************************************************/
/*                                                                     */
/*       DataIm2D                                                      */
/*                                                                     */
/***********************************************************************/


            /****************************/
            /*   DataIm2dGen            */
            /****************************/

DataIm2DGen::DataIm2DGen(INT Tx,INT Ty)
{
    Initializer(Tx,Ty);
    mTyMem = Ty;
}



void DataIm2DGen::Initializer(INT Tx,INT Ty)
{
     ELISE_ASSERT((Tx>=0)&&(Ty)>=0,"Bad Size in DataIm2DGen::Initialize");
     mTx =Tx;
     mTy =Ty;
    _txy[0] = Tx;
    _txy[1] = Ty;
}



            /****************************/
            /*   DataIm2dGen            */
            /****************************/


template <class Type,class TyBase> void DataIm2D<Type,TyBase>::Resize(Pt2di aSz)
{
   DataGenImType<Type,TyBase>::Resize(aSz.x*aSz.y);

   if (aSz.y > mTyMem)
   {
     if (_to_free2)
         STD_DELETE_TAB_USER(_data);
     _data =  STD_NEW_TAB_USER(aSz.y,Type *);
     _to_free2 = true;
     mTyMem = aSz.y;
   }


   DataIm2DGen::Initializer(aSz.x,aSz.y);

   for (int y = 0 ; y<ty() ; y++)
       _data[y] = this->_data_lin + y *mTx;
}


template <class Type,class TyBase>
         DataIm2D<Type,TyBase>::DataIm2D
         (
              INT Tx,
              INT Ty,
              bool to_init,
              TyBase v_init,
              const char * str_init,
              Type *       DataLin,
              Type **      Data,
              INT          tx_phys,
              bool         NoDataLin
         )  :
             DataGenImType<Type,TyBase>
             (
                 ((DataLin== 0)&& (!NoDataLin)) ? Tx*Ty : 0,
                 to_init,
                 v_init,
                 str_init
             ),
             DataIm2DGen(Tx,Ty)
{
    if (DataLin)
    {
       this->_data_lin = DataLin;
       this->_to_free = false;
    }
    else if (NoDataLin)
       this->_to_free = false;

    _to_free2 = (Data == 0);
    if (Data)
       _data = Data;
    else
       _data =  STD_NEW_TAB_USER(ty(),Type *);

    if (tx_phys == -1)
        tx_phys = tx();

    if (this->_data_lin)
        for (int y = 0 ; y<ty() ; y++)
            _data[y] = this->_data_lin + y *tx_phys;

}

template <class Type,class TyBase> const INT * DataIm2D<Type,TyBase>::p0() const
{
    return PTS_00000000000000;
}

template <class Type,class TyBase> const INT * DataIm2D<Type,TyBase>::p1() const
{
    return _txy;
}


template <class Type,class TyBase> INT   DataIm2D<Type,TyBase>::dim() const
{
     return 2;
}


template <class Type,class TyBase>
         void * DataIm2D<Type,TyBase>::calc_adr_seg(INT * pts)
{
    return _data[pts[1]] ;
}


template <class Type,class TyBase> DataIm2D<Type,TyBase>::~DataIm2D()
{
     ASSERT_INTERNAL(_data != 0,"multiple deletion of a bitmap");

     if (_to_free2)
         STD_DELETE_TAB_USER(_data);
     _data = 0;
}



template <class Type,class TyBase> void DataIm2D<Type,TyBase>::out_pts_integer
              (Const_INT_PP pts,INT nb,const void * i)
{
   const INT * tx = pts[0];
   const INT * ty = pts[1];
   const TyBase * in =  C_CAST(const TyBase *,i);

   for (int j=0 ; j<nb ; j++)
       _data[ty[j]][tx[j]] = (Type) in[j];
}

template <class Type,class TyBase> void DataIm2D<Type,TyBase>::input_pts_integer
              (void * o,Const_INT_PP pts,INT nb) const
{
   const INT * tx = pts[0];
   const INT * ty = pts[1];
   TyBase * out =  C_CAST(TyBase *,o);

   for (int i=0 ; i<nb ; i++)
       out[i] = _data[ty[i]][tx[i]];
}


template <class Type,class TyBase> void DataIm2D<Type,TyBase>::input_pts_reel
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
                 p_0x * p_0y * _data[ yi ][ xi ]
               + p_1x * p_0y * _data[ yi ][xi+1]
               + p_0x * p_1y * _data[yi+1][ xi ]
               + p_1x * p_1y * _data[yi+1][xi+1];
   }
}

template <class Type,class TyBase>
         void DataIm2D<Type,TyBase>::out_assoc
         (
                  void * out,
                  const OperAssocMixte & op,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
         )
         const
{
      TyBase * v = (TyBase *) const_cast <void *>(values);
      TyBase * o =  (TyBase *) out;
      const INT * x    = coord[0];
      const INT * y    = coord[1];

      Type  * adr_vxy;
      INT nb0 = nb;

      if (op.id() == OperAssocMixte::Sum)
          while(nb--)
          {
               adr_vxy = _data[*(y++)] + *(x++);
               *(o ++) =  *adr_vxy;
               *adr_vxy += (Type)*(v++);
          }
      else
          while(nb--)
          {
               adr_vxy = _data[*(y++)] + *(x++);
               *(o ++) =  *adr_vxy;
               *adr_vxy = (Type)op.opel((TyBase)(*adr_vxy),*(v++));
          }
      verif_value_op_ass(op,o-nb0,v-nb0,nb0,(TyBase)this->v_min,(TyBase)this->v_max);
}




template <class Type,class TyBase>
    Type **  DataIm2D<Type,TyBase>::data() const {return _data;}

template <class Type,class TyBase>
    Type *  DataIm2D<Type,TyBase>::data_lin() const {return this->_data_lin;}


template <class Type,class TyBase>
         void   DataIm2D<Type,TyBase>::q_dilate
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
   //INT szb_out = set_dilated->pck_sz_buf();
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
               INT v = (INT) _data[yo][xo];
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
            int isnei = (func_selection._l[(INT)_data[yo][xo]]);
            if (isnei)
            {
                _data[yo][xo] = (Type)func_update._l[(INT)_data[yo][xo]];
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



template <class Type,class TyBase>
         REAL DataIm2D<Type,TyBase>::som_rect(Pt2dr p0,Pt2dr p1,REAL def) const
{
    REAL sign = 1;

    if (p0.x > p1.x)
    {
         sign= -1;
         ElSwap(p0.x,p1.x);
    }
    if (p0.y > p1.y)
    {
         sign = -sign;
         ElSwap(p0.y,p1.y);
    }




    p0 = p0 +Pt2dr(0.5,0.5);
    p1 = p1 +Pt2dr(0.5,0.5);

    REAL res =0.0;

    INT x0 = round_down(p0.x);
    INT x1 = round_up  (p1.x);
    INT y0 = round_down(p0.y);
    INT y1 = round_up  (p1.y);

    for (INT x=x0; x<x1; x++)
        for (INT y=y0; y<y1; y++)
        {
             REAL v =
                    ((x>=0) && (x<tx()) && (y>=0) && (y<ty()))  ?
                    _data[y][x]                                 :
                    def                                         ;

             REAL X0 = ElMax(p0.x,(REAL)x);
             REAL X1 = ElMin(p1.x,(REAL)x+1);
             REAL Y0 = ElMax(p0.y,(REAL)y);
             REAL Y1 = ElMin(p1.y,(REAL)y+1);

             res += v * (X1-X0) * (Y1-Y0);
        }

   return res * sign;
}

template <class Type,class TyBase>
         REAL DataIm2D<Type,TyBase>::moy_rect(Pt2dr p0,Pt2dr p1,REAL def) const
{
   return som_rect(p0,p1,def) / ((p1.x-p0.x)*(p1.y-p0.y));
}



template <class Type,class TyBase>
         void DataIm2D<Type,TyBase>::set_brd(Pt2di sz,Type Val)
{
    sz.x = ElMin(sz.x,(mTx+1)/2);
    sz.y = ElMin(sz.y,(mTy+1)/2);

   for (INT wx=0; wx <sz.x ; wx++)
   {
        INT wxC = mTx-wx-1;
        for (INT y =0 ; y<mTy; y++)
            _data[y][wx] =  _data[y][wxC] = Val;
   }

   for (INT wy=0; wy <sz.y ; wy++)
   {
        INT wyC = mTy-wy-1;
        for (INT x =0 ; x<mTx; x++)
            _data[wy][x] =  _data[wyC][x] = Val;
   }
}


template <class Type,class TyBase>
         void DataIm2D<Type,TyBase>::raz(Pt2di p0,Pt2di p1)
{
   pt_set_min_max(p0,p1);
   p0.SetSup(Pt2di(0,0));
   p1.SetInf(Pt2di(mTx,mTy));

   if ((p1.x <= p0.x) || (p1.y<=p0.y))
      return;

   for (INT y=p0.y ; y<p1.y ; y++)
       MEM_RAZ(_data[y]+p0.x,p1.x-p0.x);
}

template <class Type,class TyBase>
         double   DataIm2D<Type,TyBase>::Get
              (const Pt2dr & aP,const cInterpolateurIm2D<Type> & anInterp,double aDef)
{
   int aSz = anInterp.SzKernel();

   if (
           (aP.x < aSz)
    || (aP.y < aSz)
        || (aP.x > mTx -aSz-1)
    || (aP.y > mTy -aSz-1)
      )
   {
      return aDef;
   }
   return anInterp.GetVal(_data,aP);
}

           /***********************/
           /*    Im2DGen          */
           /***********************/

Im2DGen::~Im2DGen(){}
Im2DGen::Im2DGen(DataGenIm * gi) :
      GenIm(gi)
{
}

GenIm::type_el Im2DGen::TypeEl() const
{
    return DGI()->type();
}

DataGenIm *  Im2DGen::DGI() const
{
    return (DataGenIm*) _ptr;
}

INT Im2DGen::tx() const
{
    return DGI()->p1()[0];
}
INT Im2DGen::ty() const
{
    return DGI()->p1()[1];
}

double   Im2DGen::MoyG2() const
{
   ELISE_ASSERT(false,"no Im2DGen::MoyG2");
   return 0;
}

INT Im2DGen::vmax() const
{
    return DGI()->vmax();
}
INT Im2DGen::vmin() const
{
    return DGI()->vmin();
}

cIm2DInter * Im2DGen::BilinIm()
{
   ELISE_ASSERT(false,"no Im2DGen::BilinIm");
   return 0;
}
cIm2DInter * Im2DGen::BiCubIm(double aCoef,double aScale)
{
   ELISE_ASSERT(false,"no Im2DGen::BiCubIm");
   return 0;
}

cIm2DInter * Im2DGen::SinusCard(double SzSin,double SzApod)
{
   ELISE_ASSERT(false,"no Im2DGen::SinusCard");
   return 0;
}




void Im2DGen::TronqueAndSet(const Pt2di &,double aVal)
{
   ELISE_ASSERT(false,"no Im2DGen::TronqueAndSet");
}

double Im2DGen::Val(const int & x,const int & y) const
{
   ELISE_ASSERT(false,"no Im2DGen::Val");
   return 0;
}

Box2di Im2DGen::BoxDef() const
{
  return Box2di(Pt2di(0,0),Pt2di(tx(),ty()));
}

INT     Im2DGen::GetI(const Pt2di & ) const
{
   ELISE_ASSERT(false,"no Im2DGen::GetI");
   return 0;
}
double     Im2DGen::GetR(const Pt2di & ) const
{
   ELISE_ASSERT(false,"no Im2DGen::GetR");
   return 0;
}

void Im2DGen::Resize(Pt2di aSz)
{
   ELISE_ASSERT(false,"no Im2DGen::Resize");
}

void Im2DGen::SetI(const Pt2di & ,int aValI)
{
   ELISE_ASSERT(false,"no Im2DGen::SetI");
}
void Im2DGen::SetR(const Pt2di & ,double aValR)
{
   ELISE_ASSERT(false,"no Im2DGen::SetR");
}




void Im2DGen::SetI_SVP(const Pt2di & aP ,int aValI)
{
   if (Inside(aP))
       SetI(aP,aValI);
}

void Im2DGen::SetR_SVP(const Pt2di & aP,double aValR)
{
   if (Inside(aP))
       SetR(aP,aValR);
}



void Im2DGen::PutData(FILE * aFP,const Pt2di & anI,bool aModeBin) const
{
   ELISE_ASSERT(false,"no Im2DGen::PutData");
}

INT    Im2DGen::GetI(const Pt2di &aP,int aDef ) const
{
   return Inside(aP) ? GetI(aP) : aDef;
}

double Im2DGen::GetR(const Pt2di &aP,double aDef ) const
{
   return Inside(aP) ? GetR(aP) : aDef;
}

Im2DGen  * Im2DGen::ImOfSameType(const Pt2di & aSz) const
{
   ELISE_ASSERT(false,"no Im2DGen::ImOfSameType");
   return new Im2D_U_INT1(0,0);
}

Im2DGen  * Im2DGen:: ImRotate(int aRot ) const
{
   ELISE_ASSERT(false,"no Im2DGen::ImOfSameType");
   return new Im2D_U_INT1(0,0);
}



bool  Im2DGen::Inside(const Pt2di & aP) const
{
   return    (aP.x>=0)
          && (aP.y>=0)
          && (aP.x<tx())
          && (aP.y<ty());
}

void Im2DGen::AssertInside(const Pt2di & aP) const
{
   ELISE_ASSERT(Inside(aP),"Im2DGen::AssertInside");
}




Seg2d   Im2DGen::OptimizeSegTournantSomIm
        (
            REAL &                 score,
            Seg2d                  seg,
            INT                    NbPts,
            REAL                   step_init,
            REAL                   step_limite,
            bool                   optim_absc,
            bool                   optim_teta ,
            bool                   * FreelyOpt
        )
{
    ELISE_ASSERT(false,"No OptimizeSegTournantSomIm for required type");
    return Seg2d(Pt2dr(0,0),Pt2dr(0,0));
}





//  neigh_test_and_set  -> see  "src/neighboor/b2d_spec_neigh.C"



           /***********************/
           /*    Im2D             */
           /***********************/

template <class Type,class TyBase>
          Fonc_Num  Im2D<Type,TyBase>::FoncEtalDyn()
{
   TyBase aMin,aMax;

    ELISE_COPY(all_pts(),in(),VMin(aMin)|VMax(aMax));

    if (aMax == aMin) aMax++; // Pour eviter / 0

    return (in()-double(aMin)) / (REAL) (aMax-aMin);
}


template <class Type,class TyBase>
         Im2D<Type,TyBase>
         Im2D<Type,TyBase>::AugmentSizeTo(Pt2di aSz, Type aValCompl )
{
    if ((aSz.x<=tx()) && (aSz.y<= ty()))
       return * this;

    Im2D<Type,TyBase>  aRes (ElMax(tx(),aSz.x),ElMax(ty(),aSz.y),aValCompl);
    ELISE_COPY(all_pts(),in(),aRes.out());

    return aRes;
}


template <class Type,class TyBase> Im2D<Type,TyBase>::~Im2D()
{
}

template <class Type,class TyBase>
         Im2D<Type,TyBase>::Im2D() :
        Im2DGen(new DataIm2D<Type,TyBase>(1,1,false,0))
{
}


template <class Type,class TyBase>
         Im2D<Type,TyBase>::Im2D(INT tx,INT ty) :
        Im2DGen(new DataIm2D<Type,TyBase>(tx,ty,false,0))
{
}


template <class Type,class TyBase>
        Im2D<Type,TyBase>::Im2D(INT tx,INT ty,TyBase v_init) :
        Im2DGen(new DataIm2D<Type,TyBase>(tx,ty,true,v_init))
{
}

template <class Type,class TyBase>
        Im2D<Type,TyBase>::Im2D(INT tx,INT ty,const char * v_init) :
        Im2DGen(new DataIm2D<Type,TyBase>(tx,ty,false,0,v_init))
{
}


template <class Type,class TyBase>
        Im2D<Type,TyBase>::Im2D(Im2D_NoDataLin,INT tx,INT ty) :
        Im2DGen(new DataIm2D<Type,TyBase>
                    (
                         tx,ty,false,
                         0,(char *)0,
                         (Type *)0,(Type **)0,
                         -1,true
                    )
               )
{
}






template <class Type,class TyBase>
        Im2D<Type,TyBase>::Im2D
        (
              Type *    DataLin,
              Type **   Data,
              INT tx,
              INT ty,
              INT tx_phys
        ) :
        Im2DGen
        (
             new DataIm2D<Type,TyBase>
             (
                 tx,
                 ty,
                 false,
                 0,
                 0,
                 DataLin,
                 Data,
                 tx_phys
             )
         )
{
}


template <class Type,class TyBase> Im2D<Type,TyBase>  Im2D<Type,TyBase>::FromFileStd(const std::string & aName)
{
   Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
   Im2D<Type,TyBase> aRes(aTif.sz().x,aTif.sz().y);
   ELISE_COPY(aRes.all_pts(),aTif.in(),aRes.out());
   return aRes;
}

template <class Type,class TyBase> Im2D<Type,TyBase>  Im2D<Type,TyBase>::FromFileBasic(const std::string & aName)
{
   Tiff_Im aTif = Tiff_Im::BasicConvStd(aName);
   Im2D<Type,TyBase> aRes(aTif.sz().x,aTif.sz().y);
   ELISE_COPY(aRes.all_pts(),aTif.in(),aRes.out());
   return aRes;
}




template <class Type,class TyBase>
Im2D<Type,TyBase>  Im2D<Type,TyBase>::FromFileOrFonc
(
      const std::string & aName,
      Pt2di aSz,
      Fonc_Num aFonc

)
{
   if (! ELISE_fp::exist_file(aName))
       Tiff_Im::Create8BFromFonc(aName,aSz,aFonc);
   return FromFileStd(aName);
}

template <class Type,class TyBase>
        cIm2DInter * Im2D<Type,TyBase>::BilinIm()
{
     cInterpolBilineaire<Type> * aIB = new cInterpolBilineaire<Type>;
      return new cTpIm2DInter<Type,TyBase>(*this,aIB);
}

template <class Type,class TyBase>
        double  Im2D<Type,TyBase>::MoyG2() const
{
   int aTX = tx();
   int aTY = ty();
   Type ** aData = data();
   double aSom = 0.0;
   for (int anY=1 ; anY<aTY ; anY++)
   {
       Type * aLP = aData[anY-1]+1;
       Type * aLCur = aData[anY]+1;
       for (int anX=1 ; anX<aTX ; anX++)
       {
           aSom += ElSquare(double(*aLCur-aLCur[-1])) +  ElSquare(double(*aLP-*aLCur));
           aLCur++;
           aLP++;
       }
   }
   return  aSom / ((aTX-1) * double(aTY-1));
}

template <class Type,class TyBase>
        cIm2DInter * Im2D<Type,TyBase>::BiCubIm(double aCoef,double aScale)
{
     if (aScale==1)
     {
         cInterpolBicubique<Type> * aIB = new cInterpolBicubique<Type>(aCoef);
         return new cTpIm2DInter<Type,TyBase>(*this,aIB);
     }

     cCubicInterpKernel * aCIK = new cCubicInterpKernel (aCoef);
     cScaledKernelInterpol * aScal = new cScaledKernelInterpol(aCIK,aScale);  // deletera aCIK
     cTabIM2D_FromIm2D<Type> * aInt2D = new cTabIM2D_FromIm2D<Type>(aScal,1000,false);

     return  new cTpIm2DInter<Type,TyBase>(*this,aInt2D);
}

template <class Type,class TyBase>
        cIm2DInter * Im2D<Type,TyBase>::SinusCard(double aSzK,double aSzA)
{
    int aNbD = 1000;
    cSinCardApodInterpol1D aKer(cSinCardApodInterpol1D::eTukeyApod,aSzK,aSzA,1e-4,false);
    cTabIM2D_FromIm2D<Type> * aInt2D   =    new cTabIM2D_FromIm2D<Type>(&aKer,aNbD,false);

    return  new cTpIm2DInter<Type,TyBase>(*this,aInt2D);
   //  ELISE_ASSERT(false,"No Im2D<Type,TyBase>::SinusCard");
   //  return 0;
}

/*
      cTpIm2DInter
      (
         Im2D<Type,TyBase>  anIm,
         const cInterpolateurIm2D<Type> & anInterp
      ) :
        mIm  (anIm),
*/
template <class Type,class TyBase> double Im2D<Type,TyBase>::Val(const int & x,const int & y) const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->data()[y][x];
}

template <class Type,class TyBase> Type ** Im2D<Type,TyBase>::data()
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->data();
}

template <class Type,class TyBase> Type ** Im2D<Type,TyBase>::data() const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->data();
}

template <class Type,class TyBase> void  Im2D<Type,TyBase>::SetI(const Pt2di  &aP,int aVal )
{
   AssertInside(aP);
   data()[aP.y][aP.x] = El_CTypeTraits<Type>::Tronque(aVal);
}

template <class Type,class TyBase> void  Im2D<Type,TyBase>::SetR(const Pt2di  &aP,double aVal )
{
   AssertInside(aP);
   data()[aP.y][aP.x] = (Type)ElStdTypeScal<TyBase>::RtoT(aVal);
}


template <class Type,class TyBase> int Im2D<Type,TyBase>::GetI(const Pt2di  &aP ) const
{
   AssertInside(aP);
   return round_ni(data()[aP.y][aP.x]);
}
template <class Type,class TyBase> double Im2D<Type,TyBase>::GetR(const Pt2di & aP) const
{
   AssertInside(aP);
// std::cout << "GetR " << double(data()[aP.y][aP.x]) << " " << aP << "\n";
   return data()[aP.y][aP.x];
}

template <class Type,class TyBase>  void Im2D<Type,TyBase>::TronqueAndSet(const Pt2di & aP,double aVal)
{
   AssertInside(aP);
   data()[aP.y][aP.x] = El_CTypeTraits<Type>::TronqueR(aVal) ;
}

static void  ImPrintf(FILE * aFP,double aV )
{
    fprintf(aFP,"%lf",aV);
}
static void  ImPrintf(FILE * aFP,float aV )
{
    fprintf(aFP,"%f",aV);
}
static void  ImPrintf(FILE * aFP,int aV )
{
    fprintf(aFP,"%d",aV);
}

static void  ImPrintf(FILE * aFP,long double aV )
{
   ELISE_ASSERT(false,"ImPrintf long double ");
}

template <class Type,class TyBase> void  Im2D<Type,TyBase>::PutData
           (FILE * aFP,const Pt2di & aP,bool aModeBin) const
{
   AssertInside(aP);
   Type & aV = data()[aP.y][aP.x];
   if (aModeBin)
   {
         size_t aNb= fwrite(&aV,sizeof(Type),1,aFP);
         ELISE_ASSERT(aNb==1,"cElNuage3DMaille::PlyPutData");

   }
   else
   {
          ImPrintf(aFP,aV);
   }
}

template <class Type,class TyBase>
Im2DGen*   Im2D<Type,TyBase>::ImOfSameType(const Pt2di & aSz) const
{
   return new Im2D<Type,TyBase>(aSz.x,aSz.y,TyBase(0));
}

template <class Type,class TyBase>
Im2DGen*   Im2D<Type,TyBase>::ImRotate(int aIndexRot) const
{
    Pt2di aRot = Pt2di(1,0);
    for (int aK=0 ; aK<aIndexRot ; aK++)
    {
         aRot = aRot * Pt2di(0,1);
    }

    Pt2di aP0((int)1e9,(int)1e9),aP1((int)-1e9,(int)-1e9);
    Box2di aBox(Pt2di(0,0),Pt2di(tx()-1,ty()-1));

    Pt2di aCoins[4];
    aBox.Corners(aCoins);
    for (int aK=0; aK<4; aK++)
    {
        aP0 = Inf(aP0,aCoins[aK]*aRot);
        aP1 = Sup(aP1,aCoins[aK]*aRot);
    }
    // Out  = In * aRot -aP0
    // In = (Out+aP0) / aRot
    Pt2di aSzOut = aP1 -aP0 + Pt2di(1,1);

    Im2D<Type,TyBase>* aRes = new  Im2D<Type,TyBase>(aSzOut.x,aSzOut.y);

    Type ** aDIn = data();
    Type ** aDOut = aRes->data();

    Pt2di aPout;
    for (aPout.y=0 ; aPout.y <aSzOut.y; aPout.y++)
    {
        for (aPout.x=0 ; aPout.x <aSzOut.x; aPout.x++)
        {
             Pt2di aPIn = (aPout+aP0)/ aRot;
             aDOut[aPout.y][aPout.x] = aDIn[aPIn.y][aPIn.x];
        }
    }



   return aRes;
}


template <class Type,class TyBase> Type * Im2D<Type,TyBase>::data_lin()
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->data_lin();
}

template <class Type,class TyBase> const Type * Im2D<Type,TyBase>::data_lin() const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->data_lin();
}





template <class Type,class TyBase> INT   Im2D<Type,TyBase>::tx() const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->tx();
}

template <class Type,class TyBase> INT   Im2D<Type,TyBase>::vmax() const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->v_max;
}

template <class Type,class TyBase> INT   Im2D<Type,TyBase>::vmin() const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->v_min;
}




template <class Type,class TyBase> INT   Im2D<Type,TyBase>::ty() const
{
      return ((DataIm2D<Type,TyBase> *) (_ptr))->ty();
}


template <class Type,class TyBase>
          GenIm::type_el   Im2D<Type,TyBase>::TypeEl()  const
{
    return  type_of_ptr((Type *) 0);
}

template <class Type,class TyBase>  void   Im2D<Type,TyBase>::raz()
{
       di2d()->DataGenImType<Type,TyBase>::raz();
}

template <class Type,class TyBase>  void   Im2D<Type,TyBase>::raz(Pt2di p0,Pt2di p1)
{
       di2d()->raz(p0,p1);
}


template <class Type,class TyBase>
          void   Im2D<Type,TyBase>::dup( Im2D<Type,TyBase> I2)
{
     ELISE_ASSERT
     (
        (tx() == I2.tx()) && (ty() == I2.ty()),
        "Im2D<Type,TyBase>::dup"
     );
     memcpy(data_lin(),I2.data_lin(),tx()*ty()*sizeof(Type));
}

template <class Type,class TyBase> Im2D<Type,TyBase>   Im2D<Type,TyBase>::dup()
{
   Im2D<Type,TyBase> aRes(tx(),ty());
   aRes.dup(*this);
   return aRes;
}

template <class Type,class TyBase> double   Im2D<Type,TyBase>::som_rect()
{
     double aS;
     ELISE_COPY(all_pts(),in(),sigma(aS));
     return aS;
}

template <class Type,class TyBase> Im2D<Type,TyBase>   Im2D<Type,TyBase>::ToSom1()
{
    Im2D<Type,TyBase> aRes = dup();
    ELISE_COPY(aRes.all_pts(),aRes.in()/som_rect(),aRes.out());
    return aRes;
}

/*
template <class Type,class TyBase>
Im2D<Type,TyBase>  Im2D<Type,TyBase>::ToSom1()
{
     double aSom = som_rect();
}
*/




template <class Type,class TyBase>
         REAL   Im2D<Type,TyBase>::som_rect(Pt2dr p0,Pt2dr p1,REAL def)
{
    return ((DataIm2D<Type,TyBase> *) (_ptr))->som_rect(p0,p1,def);
}

template <class Type,class TyBase>
         REAL   Im2D<Type,TyBase>::moy_rect(Pt2dr p0,Pt2dr p1,REAL def)
{
    return ((DataIm2D<Type,TyBase> *) (_ptr))->moy_rect(p0,p1,def);
}

template <class Type,class TyBase>
        Im2D<Type,TyBase> Im2D<Type,TyBase>::Reech(REAL aZoom)
{
    INT aTx = round_ni(tx()* aZoom);
    INT aTy = round_ni(ty()* aZoom);

    Im2D<Type,TyBase> aRes(aTx,aTy);

    for (INT anX=0 ; anX<aTx; anX++)
        for (INT anY=0 ; anY<aTy; anY++)
        {
           Pt2dr aP0(anX/aZoom,anY/aZoom);
           Pt2dr aP1((anX+1)/aZoom,(anY+1)/aZoom);
           aRes.data()[anY][anX] = Type(moy_rect(aP0,aP1));
        }
    return aRes;
}

template <class Type,class TyBase>
         void   Im2D<Type,TyBase>::set_brd(Pt2di Sz,Type brd)
{
   di2d()->set_brd(Sz,brd);
}


template <class Type,class TyBase>
         double   Im2D<Type,TyBase>::Get(const Pt2dr & aP,const cInterpolateurIm2D<Type> & anInterp,double aDef)
{
   return di2d()->Get(aP,anInterp,aDef);
}




template <class Type,class TyBase>
         void   Im2D<Type,TyBase>::auto_translate(Pt2di tr)
{
   TIm2D<Type,TyBase> Tim(*this);
   AutoTranslateData(tr,Tim);
}




template <class Type,class TyBase>
         void   Im2D<Type,TyBase>::Resize(Pt2di Sz)
{
   di2d()->Resize(Sz);
}



template <> Seg2d   Im2D<U_INT1,INT>::OptimizeSegTournantSomIm
                    (
                          REAL &                 score,
                          Seg2d                  seg,
                          INT                    NbPts,
                          REAL                   step_init,
                          REAL                   step_limite,
                          bool                   optim_absc,
                          bool                   optim_teta ,
                          bool                   * FreelyOpt
                    )
{
    return ::OptimizeSegTournantSomIm
           (
               score,
               *this,
                seg,
                NbPts,
                step_init,step_limite,
                optim_absc, optim_teta,FreelyOpt
           );
}

template <> Seg2d   Im2D<INT1,INT>::OptimizeSegTournantSomIm
                    (
                          REAL &                 score,
                          Seg2d                  seg,
                          INT                    NbPts,
                          REAL                   step_init,
                          REAL                   step_limite,
                          bool                   optim_absc,
                          bool                   optim_teta ,
                          bool                   * FreelyOpt
                    )
{
    return ::OptimizeSegTournantSomIm
           (
               score,
               *this,
                seg,
                NbPts,
                step_init,step_limite,
                optim_absc, optim_teta,FreelyOpt
           );
}

template <class Type,class TyBase> Seg2d   Im2D<Type,TyBase>::OptimizeSegTournantSomIm
                    (
                          REAL &                 score,
                          Seg2d                  seg,
                          INT                    NbPts,
                          REAL                   step_init,
                          REAL                   step_limite,
                          bool                   optim_absc,
                          bool                   optim_teta ,
                          bool                   * FreelyOpt
                    )
{
    ELISE_ASSERT(false,"No OptimizeSegTournantSomIm for required type");
    return Seg2d(Pt2dr(0,0),Pt2dr(0,0));
}


template <class Type,class TyBase>
        ElMatrix<double> Im2D<Type,TyBase>::ToMatrix() const
{
  ElMatrix<double> aRes(tx(),ty());
  Type ** aD = data();
  for (int anX=0 ; anX<tx(); anX++)
     for (int anY=0 ; anY<ty(); anY++)
        aRes(anX,anY) = aD[anY][anX];
  return aRes;
}


template <class Type,class TyBase>
        Im2D<Type,TyBase> Im2D<Type,TyBase>::gray_im_red(INT zoom)
{
     if (zoom ==1) return *this;
     INT aTx = tx()/ zoom;
     INT aTy = ty()/ zoom;

     Im2D<Type,TyBase> aRes(aTx,aTy);

     Type ** aDataIn  = data();
     Type ** aDataOut = aRes.data();

     INT Z2 = zoom * zoom;

     for
     (
          INT y=0,Y0 = 0, Y1 = zoom;
          y<aTy ;
          y++,Y0+= zoom,Y1+=zoom
     )
     {
          for
          (
               INT x=0,X0 = 0, X1 = zoom;
               x<aTx ;
               x++,X0+= zoom,X1+=zoom
          )
          {
              TyBase aValRes = 0;
              for (INT anY = Y0; anY < Y1; anY++)
                   for (INT anX = X0; anX < X1; anX++)
                       aValRes += aDataIn[anY][anX];
              aDataOut[y][x] = (Type) (aValRes/ Z2);
          }
     }
     return aRes;
}






template <class Type,class TyBase>
        void Im2D<Type,TyBase>::SetLine(INT x0,INT y,INT nb,INT val)
{
   x0 = ElMax(0,ElMin(x0,tx()));
   nb = ElMax(0,ElMin(nb,tx()-x0));
   y  = ElMax(0,ElMin(y,ty()));

   Type * aLine = data()[y]+x0;

   for (INT k=0; k<nb ; k++)
       aLine[k] = (Type) val;
}


template <class Type,class TyBase>
        void Im2D<Type,TyBase>::MulVect
             (
                 Im1D<Type,TyBase> aRes,
                 Im1D<Type,TyBase> aVect
             )
{
   INT aTx = tx();
   INT aTy = ty();

   ELISE_ASSERT
   (
       (aRes.tx()==aTy) && (aVect.tx() == aTx),
       "Bad Size in Im2D<Type,TyBase>::MulVect"
   );
   // Fait X = A * B ; matriciellement
   Type ** A = data();
   Type * X = aRes.data();
   Type * B = aVect.data();

   for (INT y=0 ; y<aTy ; y++)
   {
        X[y] = 0;
        for (INT x=0 ; x<aTx ; x++)
            X[y] += A[y][x] * B[x];
   }
}

template <class Type,class TyBase>
void Im2D<Type,TyBase>::getMinMax(Type &oMin, Type &oMax) const
{
	ELISE_DEBUG_ERROR(tx() <= 0 || ty() <= 0, "Im2D<Type,TyBase>::getMinMax", "sz() = " << sz());

	const Type *itPix = data_lin();
	size_t iPix = size_t(tx()) * size_t(ty());
	// oMin = numeric_limits<Type>::max();
	// oMax = numeric_limits<Type>::min();
        // MPD => Bug car numeric_limits<Type>::min() est l'espilon machine, au moins sur ma version
        oMin = *itPix ;
        oMax = *itPix ;
	while (iPix--)
	{
		if (*itPix < oMin) oMin = *itPix;
		if (*itPix > oMax) oMax = *itPix;
		itPix++;
	}
}

template <class Type,class TyBase>
void Im2D<Type,TyBase>::multiply(const Type &aK)
{
	ELISE_DEBUG_ERROR(tx() <= 0 || ty() <= 0, "Im2D<Type,TyBase>::getMinMax", "sz() = " << sz());
	if (aK == Type(1)) return;

	Type *itPix = data_lin();
	size_t iPix = size_t(tx()) * size_t(ty());
	while (iPix--) *itPix++ *= aK;
}

template <class Type,class TyBase>
void Im2D<Type,TyBase>::bitwise_add(Im2D<Type,TyBase> & aIm, Im2D<Type,TyBase> & aImOut)
{
    ELISE_DEBUG_ERROR(tx() <= 0 || ty() <= 0, "Im2D<Type,TyBase>::getMinMax", "sz() = " << sz());
    ELISE_DEBUG_ERROR(aIm.tx() <= 0 || aIm.ty() <= 0, "Im2D<Type,TyBase>::getMinMax", "sz() = " << sz());

    if ( (aImOut.tx() != tx()) || (aImOut.ty() != ty()) )
        aImOut.Resize(sz());

    Type *itPix = data_lin();
    Type *itPix_aIm = aIm.data_lin();
    Type *itPix_aImOut = aImOut.data_lin();
    size_t iPix = size_t(tx()) * size_t(ty());
    while (iPix--)
    {
//        *itPix_aImOut++ = *itPix++ + *itPix_aIm++;
        *itPix_aImOut = *itPix + *itPix_aIm;
        itPix_aImOut++;
        itPix++;
        itPix_aIm++;
    }
}

template <class Type,class TyBase>
void Im2D<Type,TyBase>::substract(const Type &aB)
{
	ELISE_DEBUG_ERROR(tx() <= 0 || ty() <= 0, "Im2D<Type,TyBase>::getMinMax", "sz() = " << sz());
	if (aB == Type(0)) return;

	Type *itPix = data_lin();
	size_t iPix = size_t(tx()) * size_t(ty());
	while (iPix--) *itPix++ -= aB;
}

template <class Type,class TyBase>
void Im2D<Type,TyBase>::ramp(const Type &aMin, const Type &aK)
{
	ELISE_DEBUG_ERROR(tx() <= 0 || ty() <= 0, "Im2D<Type,TyBase>::getMinMax", "sz() = " << sz());

	if (aMin == Type(0))
	{
		multiply(aK);
		return;
	}

	if (aK == Type(1))
	{
		substract(aMin);
		return;
	}

	Type *itPix = data_lin();
	size_t iPix = size_t(tx()) * size_t(ty());
	while (iPix--)
	{
		*itPix = (*itPix - aMin) * aK;
		itPix++;
	}
}

template <class Type,class TyBase>
void Im2D<Type,TyBase>:: ReadAndPushTif
                 (
                            std::vector<Im2D<Type,TyBase> > & aV,
                            Tiff_Im               aFile
                 )
{
   Pt2di aSzF = aFile.sz();
   Output anOut = Output::onul();

   for (int aKC=0; aKC<aFile.nb_chan() ; aKC++)
   {
      Im2D<Type,TyBase> aNewIm(aSzF.x,aSzF.y);
      anOut = (aKC==0) ? aNewIm.out() : Virgule(anOut,aNewIm.out());
      aV.push_back(aNewIm);
   }
   ELISE_COPY(aFile.all_pts(),aFile.in(),anOut);
}

template <class Type,class TyBase>
INT Im2D<Type,TyBase>::linearDataAllocatedSize() const { return di2d()->DataGenImType<Type,TyBase>::allocatedSize(); }

template <class Type,class TyBase>
INT Im2D<Type,TyBase>::dataAllocatedSize() const { return di2d()->DataIm2DGen::allocatedSize(); }



/***********************************************************************/
/*                                                                     */
/*       DataIm1D                                                      */
/*                                                                     */
/***********************************************************************/

template <class Type,class TyBase>
     const INT *  DataIm1D<Type,TyBase>::p0() const
{
     return PTS_00000000000000;
}

template <class Type,class TyBase>
     const INT *  DataIm1D<Type,TyBase>::p1() const
{
   return &(_tx);
}


template <class Type,class TyBase> void
         DataIm1D<Type,TyBase>::Initializer (int Tx,void * data)
{
    _tx = Tx;
    if (data)
       this->_data_lin = (Type *) data;
    _data = this->_data_lin;
    this->_to_free = (data==0);
}

template <class Type,class TyBase>
         DataIm1D<Type,TyBase>::DataIm1D
         (
             INT Tx,
             void * data,
             bool to_init ,
             TyBase v_init,
             const char * str_init
         )  :
         DataGenImType<Type,TyBase>((data==0)?Tx:0,to_init,v_init,str_init)
{
    Initializer(Tx,data);
/*
    _tx = Tx;
    if (data)
       _data_lin = (Type *) data;
    _data = _data_lin;
    _to_free = (data==0);
*/
}

template <class Type,class TyBase> void DataIm1D<Type,TyBase>::Resize(INT aTx)
{
   DataGenImType<Type,TyBase>::Resize(aTx);
   Initializer(aTx,0);
}


template <class Type,class TyBase> INT DataIm1D<Type,TyBase>::dim() const
{
   return 1;
}

template <class Type,class TyBase> void  * DataIm1D<Type,TyBase>::calc_adr_seg(INT *)
{
    return _data ;
}


template <class Type,class TyBase> DataIm1D<Type,TyBase>::~DataIm1D()
{
    _data = 0;
}




template <class Type,class TyBase> void DataIm1D<Type,TyBase>::out_pts_integer
              (Const_INT_PP pts,INT nb,const void * i)
{
   const INT * tx = pts[0];
   const TyBase * in =  C_CAST(const TyBase *, i);

   for (int j=0 ; j<nb ; j++)
       _data[tx[j]] = (Type) in[j];
}

template <class Type,class TyBase> void DataIm1D<Type,TyBase>::input_pts_integer
              (void * o,Const_INT_PP pts,INT nb) const
{
   const INT * tx = pts[0];
   TyBase * out =  C_CAST(TyBase *,o);

   for (int i=0 ; i<nb ; i++)
       out[i] = _data[tx[i]];
}

template <class Type,class TyBase> void DataIm1D<Type,TyBase>::input_pts_reel
              (REAL * out,Const_REAL_PP pts,INT nb) const
{
   const REAL * tx = pts[0];

   REAL x;
   REAL p_1x;
   INT xi;

   for (int i=0 ; i<nb ; i++)
   {
       x = tx[i];
       p_1x = x - (xi= (INT) x);
       out[i] = (1-p_1x)*_data[xi] + p_1x *_data[xi+1];
   }
}

template <class Type,class TyBase>
         void DataIm1D<Type,TyBase>::out_assoc
         (
                  void * out,
                  const OperAssocMixte & op,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
         )
         const
{
      TyBase * v =  (TyBase *) const_cast<void *>(values);
      TyBase * o =  (TyBase *) out;
      const INT * x    = coord[0];

      Type  * adr_vx;
      INT nb0 = nb;
      if (op.id() == OperAssocMixte::Sum)
         while(nb--)
         {
              adr_vx = _data + *(x++);
              *(o ++) =  *adr_vx;
              *adr_vx += (Type) *(v++);
         }
      else
         while(nb--)
         {
              adr_vx = _data + *(x++);
              *(o ++) =  *adr_vx;
              *adr_vx = (Type) op.opel((TyBase)(*adr_vx),*(v++));
         }
      verif_value_op_ass(op,o-nb0,v-nb0,nb0,(TyBase)this->v_min,(TyBase)this->v_max);
}




template <class Type,class TyBase>
    Type *  DataIm1D<Type,TyBase>::data() const {return _data;}



template <class Type,class TyBase>
    INT DataIm1D<Type,TyBase>::tx() const {return _tx;}

template <class Type,class TyBase> INT   Im1D<Type,TyBase>::vmax() const
{
      return ((DataIm1D<Type,TyBase> *) (_ptr))->v_max;
}


template <class Type,class TyBase>
          void DataIm1D<Type,TyBase>::tiff_predictor
          (INT nb_el,INT nb_ch,INT max_val,bool codage)
{
      nb_el *= nb_ch;

     if (codage)
     {
         for (INT ch=0; ch<nb_ch ; ch++)
         {
              for (INT i=nb_el-1-ch; i>=nb_ch ; i-=nb_ch)
                  _data[i] = (Type)mod(((INT)(_data[i]- _data[i-nb_ch])),max_val);
         }
     }
     else
     {
         for (INT ch=0; ch<nb_ch ; ch++)
         {
              for (INT i=ch+nb_ch; i<nb_el ; i+=nb_ch)
                  _data[i] = (Type)(((INT)(_data[i]+ _data[i-nb_ch])) %max_val);
         }
     }
}

           /***********************/
           /*    Im1D             */
           /***********************/

template <class Type,class TyBase>
        Im1D<Type,TyBase>::Im1D(Im1D<Type,TyBase> *,INT tx,void * d) :
        GenIm (new DataIm1D<Type,TyBase> (tx,d,false,0))
{
}

template <class Type,class TyBase>
        Im1D<Type,TyBase>::Im1D(INT tx) :
        GenIm (new DataIm1D<Type,TyBase> (tx,0,false,0))
{
}

template <class Type,class TyBase>
        Im1D<Type,TyBase>::Im1D(INT tx,TyBase v_init) :
        GenIm (new DataIm1D<Type,TyBase> (tx,0,true,v_init))
{
}

template <class Type,class TyBase>
        Im1D<Type,TyBase>::Im1D(INT tx,const char * v_init) :
        GenIm (new DataIm1D<Type,TyBase> (tx,0,false,0,v_init))
{
}


template <class Type,class TyBase> Type  Im1D<Type,TyBase>::At(int aK) const
{

  ELISE_ASSERT
  (
     (aK>=0) && (aK<tx()),
     " Im1D<Type,TyBase>::At"
  );
  return data()[aK];
}

template <class Type,class TyBase> Type * Im1D<Type,TyBase>::data()
{
      return ((DataIm1D<Type,TyBase> *) (_ptr))->data();
}

template <class Type,class TyBase> const Type * Im1D<Type,TyBase>::data() const
{
      return ((DataIm1D<Type,TyBase> *) (_ptr))->data();
}




template <class Type,class TyBase> INT  Im1D<Type,TyBase>::tx() const
{
      return ((DataIm1D<Type,TyBase> *) (_ptr))->tx();
}

template <class Type,class TyBase>
         Im1D<Type,TyBase>
         Im1D<Type,TyBase>::AugmentSizeTo(INT aSz, Type aValCompl )
{
    if (aSz<=tx())
       return * this;

    Im1D<Type,TyBase>  aRes (ElMax(tx(),aSz),aValCompl);
    ELISE_COPY(all_pts(),in(),aRes.out());

    return aRes;
}

template <class Type,class TyBase>
        void Im1D<Type,TyBase>::Resize(INT aTx)
{
    ((DataIm1D<Type,TyBase> *) (_ptr)) ->Resize(aTx);
}

template <class Type,class TyBase>
        void Im1D<Type,TyBase>::raz()
{
    ((DataIm1D<Type,TyBase> *) (_ptr)) ->raz();
}

template <class Type,class Type_Base> Im2D<Type,Type_Base>
                    ImMediane
                    (
                        const std::vector<Im2D<Type,Type_Base> > & aVIm,
                        Type_Base VaUnused,
                        Type      ValDef,
                        double    aTolTh
                    )
{
   ELISE_ASSERT(aVIm.size(),"Empty Vec Im in ImMediane");
   Im2D<Type,Type_Base> aI0 = aVIm[0];
   Pt2di aSz = aI0.sz();
   for (int aK=1 ; aK<int(aVIm.size()) ; aK++)
   {
       ELISE_ASSERT(aSz==aVIm[aK].sz(),"Sz im diff in ImMediane");
   }

   Im2D<Type,Type_Base> aRes(aSz.x,aSz.y);
   std::vector<Type> aV;
   for (int aX=0 ; aX<aSz.x ; aX++)
   {
       for (int aY=0 ; aY<aSz.y ; aY++)
       {
             aV.clear();
             for (int aK=0 ; aK<int(aVIm.size()) ; aK++)
             {
                  Type aVal = aVIm[aK].data()[aY][aX];
                  if (aVal != VaUnused)
                     aV.push_back(aVal);
             }

             size_t aNb = aV.size();
             if (aNb==0)
             {
                aRes.data()[aY][aX] = ValDef;
             }
             else if (aNb==1)
             {
                aRes.data()[aY][aX] = aV[0];
             }
             else if (aNb==2)
             {
                aRes.data()[aY][aX] = Type((aV[0]+ aV[1])/2.0);
             }
             else
             {
                 Type * aD = VData(aV);
                 std::sort(aD,aD+aNb);
                 double  aSomV= 0;
                 double  aSomP= 0;
                 double aTol = ElMax(aTolTh,1/double(aNb-1));
                 for (size_t aK=0 ; aK<aNb; aK++)
                 {
                     double aRank = ElAbs(aK/double(aNb-1) - 0.5);
                     double aPds = ElMax(0.0,aTol-aRank);
                     aSomV += aPds*aD[aK];
                     aSomP += aPds;
                 }
                 aRes.data()[aY][aX] = Type(aSomV/aSomP);
             }
       }
   }

   return aRes;
}



/***********************************************************************/
/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

/*
#define INSTANTIATE_BITM_KD_GEN_BASE(TyBase)\
template void verif_value_op_ass(OperAssocMixte const &, TyBase const *, TyBase const *, int, TyBase, TyBase);


#define INSTANTIATE_BITM_KD_GEN(Type,TyBase)\
template  Im2D<Type,TyBase> ImMediane(const std::vector<Im2D<Type,TyBase> > & aVIm, TyBase VaUnused,Type,double);\
             else
             {
                 Type * aD = VData(aV);
                 std::sort(aD,aD+aNb);
                 double  aSomV= 0;
                 double  aSomP= 0;
                 for (int aK=0 ; aK<aNb; aK++)
                 {
                     double aRank = aK/double(aNb-1) - 0.5;
                 }
             }
       }
   }

   return aRes;
}
*/

template<class Type,class TypeBase> Output   StdOut(std::vector<Im2D<Type,TypeBase> > & aV)
{
    Output aRes = aV[0].out();
    for (int aK=1 ; aK<int(aV.size()) ; aK++)
    {
       aRes = Virgule(aRes,aV[aK].out());
    }
    return aRes;
}
template<class Type,class TypeBase> Fonc_Num StdInput(std::vector<Im2D<Type,TypeBase> > & aV)
{
    Fonc_Num aRes = aV[0].in();
    for (int aK=1 ; aK<int(aV.size()) ; aK++)
    {
       aRes = Virgule(aRes,aV[aK].in());
    }
    return aRes;
}


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

#define INSTANTIATE_BITM_KD_GEN_BASE(TyBase)\
template void verif_value_op_ass(OperAssocMixte const &, TyBase const *, TyBase const *, int, TyBase, TyBase);


#define INSTANTIATE_BITM_KD_GEN(Type,TyBase)\
template  Im2D<Type,TyBase> ImMediane(const std::vector<Im2D<Type,TyBase> > & aVIm, TyBase VaUnused,Type,double);\
template Fonc_Num StdInput(std::vector<Im2D<Type,TyBase> > & aV);\
template Output   StdOut(std::vector<Im2D<Type,TyBase> > & aV);\
template class Im1D<Type,TyBase>;\
template class DataIm1D<Type,TyBase>;\
template class Im2D<Type,TyBase>;\
template class DataIm2D<Type,TyBase>;\
template class cTpIm2DInter<Type,TyBase>;\
template class DataGenImType<Type,TyBase>;

#if ElTemplateInstantiation
    INSTANTIATE_BITM_KD_GEN(U_INT1,INT);
    INSTANTIATE_BITM_KD_GEN(INT1,INT);
    INSTANTIATE_BITM_KD_GEN(U_INT2,INT);
    INSTANTIATE_BITM_KD_GEN(INT2,INT);
    INSTANTIATE_BITM_KD_GEN(INT4,INT);
    INSTANTIATE_BITM_KD_GEN(REAL8,REAL8);
    INSTANTIATE_BITM_KD_GEN(REAL4,REAL8);
    INSTANTIATE_BITM_KD_GEN(REAL16,REAL16);
    INSTANTIATE_BITM_KD_GEN_BASE(INT4)
    INSTANTIATE_BITM_KD_GEN_BASE(REAL8)
#endif

GenIm alloc_im1d(GenIm::type_el type_el,int tx,void * data)
{
      switch (type_el)
      {
            case GenIm::u_int1 :    return Im1D<U_INT1,INT>  ((Im1D<U_INT1,INT>  *) 0  ,tx,data);
            case GenIm::int1 :      return Im1D<INT1,INT>    ((Im1D<INT1,INT>    *) 0  ,tx,data);
            case GenIm::u_int2 :    return Im1D<U_INT2,INT>  ((Im1D<U_INT2,INT>  *) 0  ,tx,data);
            case  GenIm::int2 :     return Im1D<INT2,INT>    ((Im1D<INT2,INT>    *) 0  ,tx,data);
            case  GenIm::int4 :     return Im1D<INT4,INT>    ((Im1D<INT4,INT>    *) 0  ,tx,data);
            case  GenIm::u_int4 :   return Im1D<U_INT,_INT8>    ((Im1D<U_INT,_INT8>    *) 0  ,tx,data);
            case  GenIm::real4 :    return Im1D<REAL4,REAL8> ((Im1D<REAL4,REAL8> *) 0  ,tx,data);
            case  GenIm::real8 :    return Im1D<REAL8,REAL8> ((Im1D<REAL8,REAL8> *) 0  ,tx,data);
            default :;
      }


       elise_internal_error
                ("unknown type in alloc_im1d\n",__FILE__,__LINE__);
       return Im1D<U_INT1,INT> (-1234);
}

Im2DGen * Ptr_D2alloc_im2d(GenIm::type_el type_el,int tx,int ty)
{
      switch (type_el)
      {
            case GenIm::bits1_msbf :    return new Im2D_Bits<1> (tx,ty);
            case GenIm::bits2_msbf :    return new Im2D_Bits<2> (tx,ty);
            case GenIm::bits4_msbf :    return new Im2D_Bits<4> (tx,ty);

            case GenIm::u_int1 :       return new Im2D<U_INT1,INT> (tx,ty);
            case GenIm::int1 :         return new Im2D<INT1,INT> (tx,ty);
            case GenIm::u_int2 :       return new Im2D<U_INT2,INT> (tx,ty);
            case  GenIm::int2 :        return new Im2D<INT2,INT> (tx,ty);
            case  GenIm::int4 :        return new Im2D<INT4,INT> (tx,ty);
            case  GenIm::real4 :       return new Im2D<REAL4,REAL8> (tx,ty);
            case  GenIm::real8 :       return new Im2D<REAL8,REAL8> (tx,ty);
            default :;
      }


       elise_internal_error
                ("unknown type in alloc_im1d\n",__FILE__,__LINE__);
       return new Im2D<U_INT1,INT> (-12,-34);
}


Im2DGen D2alloc_im2d(GenIm::type_el type_el,int tx,int ty)
{
      switch (type_el)
      {
            case GenIm::bits1_msbf :    return Im2D_Bits<1> (tx,ty);
            case GenIm::bits2_msbf :    return Im2D_Bits<2> (tx,ty);
            case GenIm::bits4_msbf :    return Im2D_Bits<4> (tx,ty);

            case GenIm::u_int1 :       return Im2D<U_INT1,INT> (tx,ty);
            case GenIm::int1 :         return Im2D<INT1,INT> (tx,ty);
            case GenIm::u_int2 :       return Im2D<U_INT2,INT> (tx,ty);
            case  GenIm::int2 :        return Im2D<INT2,INT> (tx,ty);
            case  GenIm::int4 :        return Im2D<INT4,INT> (tx,ty);
            case  GenIm::real4 :       return Im2D<REAL4,REAL8> (tx,ty);
            case  GenIm::real8 :       return Im2D<REAL8,REAL8> (tx,ty);
            default :;
      }


       elise_internal_error
                ("unknown type in alloc_im1d\n",__FILE__,__LINE__);
       return Im2D<U_INT1,INT> (-12,-34);
}



GenIm alloc_im2d(GenIm::type_el type_el,int tx,int ty)
{
    return D2alloc_im2d(type_el,tx,ty);
}


GenIm alloc_im2d(GenIm::type_el type_el,int tx,int ty,void * aDL)
{
      Im2D_BitsIntitDataLin IBIDL;
      switch (type_el)
      {
            case GenIm::bits1_msbf :    return Im2D_Bits<1> (IBIDL,tx,ty,aDL);
            case GenIm::bits2_msbf :    return Im2D_Bits<2> (IBIDL,tx,ty,aDL);
            case GenIm::bits4_msbf :    return Im2D_Bits<4> (IBIDL,tx,ty,aDL);

            case GenIm::u_int1 :       return Im2D<U_INT1,INT> ((U_INT1*)aDL,(U_INT1**)0,tx,ty);
            case GenIm::int1 :         return Im2D<INT1,INT> ((INT1*)aDL,(INT1**)0,tx,ty);
            case GenIm::u_int2 :       return Im2D<U_INT2,INT> ((U_INT2*)aDL,(U_INT2**)0,tx,ty);
            case  GenIm::int2 :        return Im2D<INT2,INT> ((INT2*)aDL,(INT2**)0,tx,ty);
            case  GenIm::int4 :        return Im2D<INT4,INT> ((INT4*)aDL,(INT4**)0,tx,ty);
            case  GenIm::real4 :       return Im2D<REAL4,REAL8> ((REAL4 *)aDL,(REAL4 **)0,tx,ty);
            case  GenIm::real8 :       return Im2D<REAL8,REAL8> ((REAL8 *)aDL,(REAL8 **)0,tx,ty);
            default :;
      }


       elise_internal_error
                ("unknown type in alloc_im1d\n",__FILE__,__LINE__);
       return Im1D<U_INT1,INT> (-1234);
}




bool type_im_integral(GenIm::type_el type_el)
{
      switch (type_el)
      {
            case GenIm::bits1_msbf :
            case GenIm::bits2_msbf :
            case GenIm::bits4_msbf :

            case GenIm::bits1_lsbf :
            case GenIm::bits2_lsbf :
            case GenIm::bits4_lsbf :

            case GenIm::u_int1 :
            case GenIm::int1 :
            case GenIm::u_int2 :
            case  GenIm::int2 :
            case  GenIm::u_int4 :      
            case  GenIm::int4 :      
            case  GenIm::u_int8 :      
            case  GenIm::int8 :      
            return true;

            case  GenIm::real4 :
            case  GenIm::real8 :       return false;


            default :;
      }
      elise_internal_error
         ("unknown type in type_im_integral\n",__FILE__,__LINE__);
      return false;
}

Fonc_Num Tronque(GenIm::type_el aType,Fonc_Num aF)
{
   if (! type_im_integral(aType))
      return aF;
   if (aType == GenIm::int4)
      return aF;

   int aVMin,aVMax;
   min_max_type_num(aType,aVMin,aVMax);


  // std::cout << "TRONQUE " << aVMin << " " << aVMax << "\n";
   return Max(aVMin,Min(aVMax-1,aF));
}

INT nbb_type_num(GenIm::type_el type_el)
{
      switch (type_el)
      {
            case GenIm::bits1_msbf :
            case GenIm::bits1_lsbf :    return 1;

            case GenIm::bits2_msbf :
            case GenIm::bits2_lsbf :    return 2;

            case GenIm::bits4_msbf :
            case GenIm::bits4_lsbf :    return 4;

            case GenIm::u_int1 :
            case GenIm::int1 :          return 8;

            case GenIm::u_int2 :
            case  GenIm::int2 :         return 16;

            case  GenIm::int4 :
            case  GenIm::u_int4 :
            case  GenIm::real4 :    
                                      return 32;

            case  GenIm::int8 :
            case  GenIm::u_int8 :
            case  GenIm::real8 :        
                                     return 64;

            case  GenIm::real16 :        
                                     return 128;

            default :;
      }
      elise_internal_error
         ("unknow type in nbb_type_num\n",__FILE__,__LINE__);
      return -1234;
}

bool msbf_type_num(GenIm::type_el type_el)
{
      switch (type_el)
      {
            case GenIm::bits1_msbf :
            case GenIm::bits2_msbf :
            case GenIm::bits4_msbf :    return true;

            case GenIm::bits1_lsbf :
            case GenIm::bits2_lsbf :
            case GenIm::bits4_lsbf :    return false;

            default :;
      }
      elise_internal_error
         ("non bits type in msbf_type_num\n",__FILE__,__LINE__);
      return false;
}

bool signed_type_num(GenIm::type_el type_el)
{
      switch (type_el)
      {
            case GenIm::bits1_msbf :
            case GenIm::bits2_msbf :
            case GenIm::bits4_msbf :

            case GenIm::bits1_lsbf :
            case GenIm::bits2_lsbf :
            case GenIm::bits4_lsbf :

            case GenIm::u_int1 :
            case GenIm::u_int2 :  
            case GenIm::u_int4 :  
            case GenIm::u_int8 :  
            return false;

            case GenIm::int1 :
            case  GenIm::int2 :
            case  GenIm::int4 :
            case  GenIm::int8 :
            case  GenIm::real4 :
            case  GenIm::real8 :
            case  GenIm::real16 :
                                       return true;

            default :;
      }
      elise_internal_error
         ("float or unknown type in signed_type_num\n",__FILE__,__LINE__);
      return false;
}

GenIm::type_el type_u_int_of_nbb(INT nbb,bool msbf)
{
      switch(nbb)
      {
            case 1: return msbf              ?
                           GenIm::bits1_msbf :
                           GenIm::bits1_lsbf ;

            case 2: return msbf              ?
                           GenIm::bits2_msbf :
                           GenIm::bits2_lsbf ;

            case 4: return msbf              ?
                           GenIm::bits4_msbf :
                           GenIm::bits4_lsbf ;


            case 8  : return GenIm::u_int1;
            case 16 : return GenIm::u_int2;
            case 32 : return GenIm::u_int4;
            case 64 : return GenIm::u_int8;
    }

    El_Internal.ElAssert
    (
          0,
          EEM0 << "No unsigned INT type for nb bits = " << nbb
    );
    return GenIm::no_type;
}

GenIm::type_el type_im(const std::string & aName)
{
   if (aName=="u_int1")
      return GenIm::u_int1;
   if (aName=="int1")
      return GenIm::int1;

   if (aName=="u_int2")
      return GenIm::u_int2;
   if (aName=="int2")
      return GenIm::int2;

   if (aName=="u_int4")
      return GenIm::u_int4;
   if (aName=="int4")
      return GenIm::int4;

   if (aName=="u_int8")
      return GenIm::u_int8;
   if (aName=="int8")
      return GenIm::int8;

   if (aName=="real4")
      return GenIm::real4;
   if (aName=="real8")
      return GenIm::real8;
   ELISE_ASSERT(false,"type_im");
   return GenIm::real8;
}

std::string eToString(const GenIm::type_el & aType)
{
    if (aType==GenIm::u_int1)
       return  "u_int1";
    if (aType==GenIm::int1)
       return  "int1";

    if (aType==GenIm::u_int2)
       return  "u_int2";
    if (aType==GenIm::int2)
       return  "int2";

    if (aType==GenIm::u_int4)
       return  "u_int4";
    if (aType==GenIm::int4)
       return  "int4";

    if (aType==GenIm::u_int8)
       return  "u_int8";
    if (aType==GenIm::int8)
       return  "int8";

    if (aType==GenIm::real4)
       return  "real4";
    if (aType==GenIm::real8)
       return  "real8";
    std::cout << "Enum = GenIm::type_el\n";
    ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
    return "";
}

GenIm::type_el type_im(bool integral,INT nbb,bool Signed,bool msbf)
{
     if (integral)
     {
         if (Signed)
         {
             if (nbb == 8)
               return GenIm::int1;
             if (nbb == 16)
               return GenIm::int2;
             if (nbb == 32)
               return GenIm::int4;
             if (nbb == 64)
               return GenIm::int8;
         }
         else
            return  type_u_int_of_nbb(nbb,msbf);
     }
     else
     {
            if (nbb == 32)
               return GenIm::real4;
            if (nbb == 64)
               return GenIm::real8;
     }

    El_Internal.ElAssert(0,EEM0 << "Incoherent Type Num");
    return GenIm::no_type;
}


int VCentrale_type_num(GenIm::type_el type)
{
   if (
            (! type_im_integral(type))
         || (signed_type_num(type))
      )
      return 0;

   return (1 << (nbb_type_num(type) -1)) ;

}

void min_max_type_num(GenIm::type_el type,INT & v_min,INT &v_max)
{
     if (
             (! type_im_integral(type))
          || (type == GenIm::int4)
        )
     {
        v_max  = v_min  = 0;
        return;
     }

     v_max = 1 <<nbb_type_num(type);

     if (signed_type_num(type))
     {
         v_max /=2;
         v_min = - v_max;
     }
     else
         v_min = 0;
}

GenIm::type_el  type_of_ptr(const U_INT1 * ) {return GenIm::u_int1; }
GenIm::type_el  type_of_ptr(const INT1   * ) {return GenIm::int1;   }
GenIm::type_el  type_of_ptr(const U_INT2 * ) {return GenIm::u_int2; }
GenIm::type_el  type_of_ptr(const INT2   * ) {return GenIm::int2;   }
GenIm::type_el  type_of_ptr(const INT4   * ) {return GenIm::int4;   }
GenIm::type_el  type_of_ptr(const REAL4  * ) {return GenIm::real4;  }
GenIm::type_el  type_of_ptr(const REAL8  * ) {return GenIm::real8;  }
GenIm::type_el  type_of_ptr(const REAL16  * ) {return GenIm::real16;  }

Fonc_Num StdInPut(std::vector<Im2DGen *> aV)
{
    Fonc_Num aRes = aV[0]->in();
    for (int aK=1 ; aK<int(aV.size()) ; aK++)
    {
       aRes = Virgule(aRes,aV[aK]->in());
    }
    return aRes;
}





Output StdOutput(std::vector<Im2DGen *> aV)
{
    Output aRes = aV[0]->out();
    for (int aK=1 ; aK<int(aV.size()) ; aK++)
    {
       aRes = Virgule(aRes,aV[aK]->out());
    }
    return aRes;
}






// Output   StdOutput(std::vector<Im2DGen *>);


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
