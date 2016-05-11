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

#ifndef __BITM_TPL__
#define __BITM_TPL__

template <class Type,class TyBase>  class DataGenImType :
           public DataGenIm
{
   public :
      virtual void  *  data_lin_gen();
      DataGenImType(INT sz_tot,bool to_init,TyBase v_init,const char * =0);
      void Initializer(INT sz_tot,bool to_init,TyBase v_init,const char * =0);
      virtual void  out_rle(void *,INT,const INT*,INT offs_0) const;
      virtual void  out_rle(void *,INT,const _INT8*,INT offs_0) const;
      virtual void  out_rle(void *,INT,const REAL*,INT offs_0) const;
      virtual void  out_rle(void *,INT,const REAL16*,INT offs_0) const;

      virtual void  void_input_rle(void *,INT,const void*,INT offs_0) const;
      virtual void  int8_input_rle(_INT8 *,INT,const void*,INT offs_0) const;
      virtual void  striped_input_rle(void *,INT nb,INT dim,const void*,INT offs_0) const;
      virtual void  striped_output_rle(void *,INT nb,INT dim,const void*,INT offs_0) const;
      virtual bool integral_type() const;


      void raz();

//      void lut_uc(Im1D<Type,TyBase>);
// modif DB
      bool       _to_free;
      INT         _sz_tot;
      INT         mSzMemory;  // est >=  a _sz_tot
      Type *      _data_lin; // for afficionados of manipulations
                             // like _data_lin[x+y*_tx] 
      virtual INT sz_el() const;
      virtual INT sz_base_el() const;
      virtual INT sz_tot() const;


      // max and min integral values of the type: the convention
      // v_max == v_min is used when these values are useless:
      static CONST_STAT_TPL INT v_min;
      static CONST_STAT_TPL INT v_max;
      static GenIm::type_el     type_el_bitm;



      virtual GenIm::type_el type() const;
      virtual INT vmax() const;
      virtual INT vmin() const;

      INT allocatedSize() const { return mSzMemory; }

      static CONST_STAT_TPL bool  _integral_type;
      protected :
          virtual ~DataGenImType();
          void Resize(INT tTot);
          void Desinitializer();
};


class DataIm2DGen
{
   public :

      INT tx() const { return _txy[0];};
      INT ty() const { return _txy[1];};
      INT allocatedSize() const { return mTyMem; }

   protected :
      void Initializer(INT aTx,INT aTy);
      INT           _txy[2];
	  INT           mTx;
	  INT           mTy;
          INT           mTyMem;
      DataIm2DGen(INT tx,INT ty);
};


template <class Type,class TyBase> class DataIm2D : 
             public  DataGenImType<Type,TyBase> ,
             public  DataIm2DGen
{
    public :


      double Get(const Pt2dr & aP ,const cInterpolateurIm2D<Type> &,double aDef);

      virtual void out_pts_integer(Const_INT_PP coord,INT nb,const void *) ;
      virtual void input_pts_integer(void *,Const_INT_PP coord,INT nb) const;
      virtual void input_pts_reel(REAL *,Const_REAL_PP coord,INT nb) const;

      virtual void  *   calc_adr_seg(INT *);
      virtual ~DataIm2D();
	  void raz(Pt2di p0,Pt2di p1);

      Type **  data() const; 
      Type *  data_lin() const; 

      REAL  som_rect(Pt2dr p0,Pt2dr p1,REAL def =0.0) const;
      REAL  moy_rect(Pt2dr p0,Pt2dr p1,REAL def =0.0) const;          
	  void set_brd(Pt2di sz,Type val); 

      virtual INT    dim() const;
      virtual const INT  * p0()  const;
      virtual const INT  * p1()  const;

      Type **     _data;
      bool        _to_free2;

      virtual void q_dilate
                   (  Std_Pack_Of_Pts<INT> * set_dilated,
                      char **                is_neigh,
                      const Std_Pack_Of_Pts<INT> * set_to_dilate,
                      INT **,
                      INT   nb_v,
                      Image_Lut_1D_Compile   func_selection,
                      Image_Lut_1D_Compile   func_update
                   );

      DataIm2D
      (
           INT tx, 
           INT ty,
           bool to_init,
           TyBase v_init,
           const char * =0,
           Type *      dlin = 0,
           Type **     d2 = 0,
           INT         tx_phys = -1,
		   bool        NoDataLin = false
      );

      virtual void out_assoc
              (
                  void * out, // eventually 0
                  const OperAssocMixte &,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
              ) const;

          void Resize(Pt2di aSz);
};


template <class Type,class TyBase> class DataIm1D : 
            public DataGenImType<Type,TyBase>
{
    public :


      static CONST_STAT_TPL  DataIm1D<Type,TyBase> The_Bitm; 


      virtual void out_pts_integer(Const_INT_PP coord,INT nb,const void *) ;
      virtual void input_pts_integer(void *,Const_INT_PP coord,INT nb) const;
      virtual void input_pts_reel(REAL *,Const_REAL_PP coord,INT nb) const;


      DataIm1D(INT tx,void * data ,bool to_init ,TyBase v_init,const char * =0);

      virtual INT    dim() const;
      virtual const INT  * p0()  const;
      virtual const INT  * p1()  const;

      virtual void  *   calc_adr_seg(INT *);
      virtual ~DataIm1D();

      Type *  data() const;
      INT tx() const ;

       INT           _tx;
       Type *     _data;

       virtual void out_assoc
              (
                  void * out, // eventually 0
                  const OperAssocMixte &,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
              ) const;

       virtual void tiff_predictor(INT nb_el,INT nb_ch,INT max_val,bool codage);

       void Resize(INT aTx);
       void Initializer (int Tx,void * data);

     protected :

};


template <class Type,class TyBase> class DataIm3D : 
            public DataGenImType<Type,TyBase>
{
    public :




      virtual void out_pts_integer(Const_INT_PP coord,INT nb,const void *) ;
      virtual void input_pts_integer(void *,Const_INT_PP coord,INT nb) const;
      virtual void input_pts_reel(REAL *,Const_REAL_PP coord,INT nb) const;


      DataIm3D(
                INT tx,INT ty,INT tz,
                bool to_init,TyBase v_init,const char * =0,Type * DataLin  = 0
      );

      virtual INT    dim() const;
      virtual const INT  * p0()  const;
      virtual const INT  * p1()  const;

      virtual void  *   calc_adr_seg(INT *);
      virtual ~DataIm3D();

      Type ***  data() const;
      INT tx() const ;
      INT ty() const ;
      INT tz() const ;

       INT           _txyz[3];
       Type ***    _data;

       virtual void out_assoc
              (
                  void * out, // eventually 0
                  const OperAssocMixte &,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
              ) const;


     protected :

};



template <class Type> class Liste_Pts_in_Comp;

class Data_Liste_Pts_Gen;

class Data_Liste_Pts_Gen  : public RC_Object
{
     friend class Liste_Pts_Out_Comp;
     friend class Liste_Pts_Out_Not_Comp;
     friend class Liste_Pts_in_Comp<INT>;
     friend class Liste_Pts_in_Comp<REAL>;
     friend class Liste_Pts_in_Not_Comp;

     public :
          bool      empty() const;
          INT       card() const;
          INT       dim() const;
          virtual ~Data_Liste_Pts_Gen();


     protected :
         Data_Liste_Pts_Gen(const DataGenIm  *,INT dim);
         void free_el();


         class  el_liste_pts : public Mcheck
         {
             public :
                enum {Elise_SZ_buf_lpts = 48};
                class el_liste_pts * next;
                char * buf(){return reinterpret_cast<char *>(&_buf[0]);}
             private :
                // Assume that double has the maximum aligmnent constraint
                double _buf[(Elise_SZ_buf_lpts+sizeof(double)-1)/sizeof(double)];
         };



         const DataGenIm     *  _gi;
         INT              _dim;
         INT              _nb_last;
         INT              _sz_el;
         INT              _sz_base;
         INT              _nb_by_el;
         bool             _free_for_out;
         el_liste_pts *   _first;
         el_liste_pts *   _last;
         bool _is_copie ; // when true, do not free in ~Data_Liste_Pts_Gen

         void cat_pts(char * * coord,INT nb);
         INT next_pts(char * * coord,INT nb_max);

};


template  <class Type,class TyBase> class Data_Liste_Pts : 
                    public Data_Liste_Pts_Gen
{
    friend class Liste_Pts<Type,TyBase>;
    public :
        virtual ~Data_Liste_Pts();
    private :

        Data_Liste_Pts(INT dim);

        Im2D<Type,TyBase>  image();
        void add_pt(Type *);
};

template <class TyBase>
         void verif_value_op_ass
              (
                  const OperAssocMixte & op,
                  const TyBase *         pre_out,
                  const TyBase *         values,
                  INT                    nb0,
                  TyBase                 v_min,
                  TyBase                 v_max
              );

#endif

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
