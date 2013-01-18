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



#ifndef _ELISE_FLUX_PTS
#define _ELISE_FLUX_PTS



/*
      One of the major feature of the ELISE'lib is the generality of the
    set of point that can be manipulated.  The abstract data_type used
    to describe a set of points in general terms is "Flux_Pts_Compute".

       Logically a Flux_Pts_Compute is an iterator that return points.


*/



/***********************************************************/
/*                                                         */
/*   Packet of points                                      */
/*                                                         */
/***********************************************************/


/*
       For computation speed the Flux_Pts_Compute  do not describe their
    set points by points but by packet of point. A Pack_Of_Pts contains :

    * dim : dimension, ie dim=2 for the majority of still image
      application;

    * nb : number of points;

    * type_pck : describe how the packet is coded, it can have the following
      value :

             rle, for run line encoding, this the way surfacic set are
              described;

             integer for a set of point with integer coordinate where the
              rle coding is not appropriate;

             real  for a set of point with real coordinates;
*/


template <class Type> class Std_Pack_Of_Pts;
class   RLE_Pack_Of_Pts;
class   Std_Pack_Of_Pts_Gen;

class Pack_Of_Pts : public Mcheck
{
     public : 

          const RLE_Pack_Of_Pts * rle_cast () const
          {
               ASSERT_INTERNAL(_type == rle,"RLE Convertion");
               return (const RLE_Pack_Of_Pts *) this;
          }

          const Std_Pack_Of_Pts_Gen * std_cast () const
          {
               ASSERT_INTERNAL(_type != rle,"Std  Convertion");
               return (const Std_Pack_Of_Pts_Gen *) this;
          }

          const Std_Pack_Of_Pts<INT> * std_cast (INT *) const
          {
               ASSERT_INTERNAL(_type == integer,"Std INT Convertion");
               return (const Std_Pack_Of_Pts<INT> *) this;
          }

          const Std_Pack_Of_Pts<REAL> * std_cast (REAL *) const
          {
               ASSERT_INTERNAL(_type == real,"Std REAL Convertion");
               return (const Std_Pack_Of_Pts<REAL> *) this;
          }
          const Std_Pack_Of_Pts<REAL> * real_cast () const
          {
                return std_cast((REAL *) 0);
          }
          const Std_Pack_Of_Pts<INT> * int_cast () const
          {
                return std_cast((INT *) 0);
          }


          typedef enum type_pack
          {
                rle,
                integer,
                real
          }
          type_pack;

          static type_pack type_common(type_pack,type_pack);
          // return the type necessary to represent object
          // of both type in a  common one (+or-, max of type)

          inline INT           nb(void) const {return _nb;};
          inline INT          dim(void) const {return _dim;};
          inline INT          pck_sz_buf(void) const {return _sz_buf;};
          inline INT          not_full() const { return _nb < _sz_buf;}
          inline void         set_nb(INT aNb) {_nb = aNb;};
          inline type_pack    type(void) const {return _type;};

          virtual  ~Pack_Of_Pts() ;
          virtual  void show (void) const =0; // debug purpose

          virtual  void show_kth (INT k) const =0; // debug purpose

          virtual  void * adr_coord() const;

          virtual  void trans(const Pack_Of_Pts *, const INT * tr) =0;
          static Pack_Of_Pts  * new_pck(INT dim,INT sz_buf,type_pack);

          // NEW convention !! the result is the arg.
          virtual void select_tab
                  (Pack_Of_Pts * pck,const INT * tab_sel) const =0;

          virtual Pack_Of_Pts * dup(INT sz_buf = -1) const = 0;
          virtual void  copy(const Pack_Of_Pts *)  = 0;


          void convert_from(const Pack_Of_Pts *) ;
          virtual void  kth_pts(INT *,INT k) const = 0;

           virtual INT   proj_brd
             (
                      const Pack_Of_Pts * pck,
                      const INT * p0,
                      const INT * p1,
                      INT         rab
              ) =0;
     protected :

          virtual void convert_from_rle(const RLE_Pack_Of_Pts *) ;
          virtual void convert_from_int(const Std_Pack_Of_Pts<INT>  *) ;
          virtual void convert_from_real(const Std_Pack_Of_Pts<REAL>  *) ;

          Pack_Of_Pts (INT dim,INT _sz_buf,type_pack);
          INT      _nb;
          const INT      _dim;
          const INT      _sz_buf;
          type_pack      _type;
};


class Curser_on_PoP : public Mcheck
{
   public :

      virtual  Pack_Of_Pts * next()    =0;
      virtual  void re_start(const Pack_Of_Pts *) =0;
      virtual  ~Curser_on_PoP();
      static   Curser_on_PoP * new_curs(INT dim,INT sz_buf,Pack_Of_Pts::type_pack);

   protected :
      Curser_on_PoP(INT dim,INT _sz_buf,Pack_Of_Pts::type_pack);
      INT                       _dim;
      INT                       _sz_buf;
      Pack_Of_Pts::type_pack    _type;
      INT                       _nb_rest;
      
};



class RLE_Pack_Of_Pts : public Pack_Of_Pts
{
    friend class RLE_Curser_on_PoP;

    public :
       virtual ~RLE_Pack_Of_Pts(void);
       virtual  void show (void) const ; // debug purpose
       virtual  void show_kth (INT k) const; // debug purpose
       static RLE_Pack_Of_Pts  * new_pck(INT dim,INT sz_buf);
       void set_pt0(Pt2di);
       void set_pt0(const INT4 *);

       inline INT4     & x0(){return _pt0[0];};
       inline INT4      vx0() const {return _pt0[0];};
       inline INT4     x1() const {return _pt0[0] + _nb;};
       inline INT4 * pt0() const {return _pt0;};

       inline INT4 y() const 
	   {
			   ELISE_ASSERT(_dim>=2,"RLE_Pack_Of_Pts::y()");
			   return _pt0[1];
	   };


       // clip the pack of point in the box defined by p0 & p1;
       // the result, is the number of point supressed at the beginig
       // of segment;

       INT   clip(const RLE_Pack_Of_Pts * pck,const INT * p0,const INT * p1);
       INT   clip(const RLE_Pack_Of_Pts * pck,Pt2di p0,Pt2di p1);

       INT   proj_brd
             (
                      const Pack_Of_Pts * pck,
                      const INT * p0,
                      const INT * p1,
                      INT         rab
             );

       bool same_line(const RLE_Pack_Of_Pts *) const;


        // inside : boolean indicating if the set is inside the box
       bool   inside(const INT * p0,const INT * p1) const;
       void   Show_Outside(const INT * p0,const INT * p1) const;
       INT   ind_outside(const INT * p0,const INT * p1) const;
      

       // following constant, a priori. used for "internally manipulated packs"
       // where size_buf is useless because pak is not exported,

       static const INT _sz_buf_infinite ;
       virtual  void  trans(const Pack_Of_Pts *, const INT * tr);
       virtual void select_tab
               (Pack_Of_Pts * pck,const INT * tab_sel) const ;

       virtual Pack_Of_Pts * dup(INT sz_buf = -1) const ;
       virtual void  copy(const Pack_Of_Pts *) ;
       virtual void  kth_pts(INT *,INT k) const;
    private :
       RLE_Pack_Of_Pts (INT dim,INT sz_buf);
       INT4 * _pt0;
       Tprov_INT * _tpr;

};


/***********************************************************/
/*                                                         */
/*   Flux_Pts_Computed                                     */
/*                                                         */
/***********************************************************/


class Flux_Pts_Computed : public Mcheck
{
   friend class Split_to_max_buf_rle;

   public :
       virtual  bool   is_rect_2d(Box2di &);
       virtual  bool   is_line_map_rect();
       virtual  REAL   average_dist();  // to use only if is_line_map_rect -> True


       virtual const Pack_Of_Pts * next(void) = 0;
       virtual  ~Flux_Pts_Computed();
       inline INT                          dim(void) const {return _dim;};
       inline  Pack_Of_Pts::type_pack     type(void) const {return _type;};
       inline INT                       sz_buf(void) const {return _sz_buf;};
       inline bool integral_flux() const {return _type != Pack_Of_Pts::real;}

       Flux_Pts_Computed * convert(Pack_Of_Pts::type_pack) ;

       static void type_common(Flux_Pts_Computed **,Flux_Pts_Computed **);

   protected :

       Flux_Pts_Computed(INT dim,Pack_Of_Pts::type_pack type,INT sz_buf);

   private :
       const INT                              _dim;
       const Pack_Of_Pts::type_pack           _type;
       const INT                              _sz_buf;
};

class RLE_Flux_Pts_Computed : public Flux_Pts_Computed
{
     public :
          static Flux_Pts_Computed * rect_2d_interface( Pt2di p0,
                                                            Pt2di p1,
                                                            INT sz_buf
                                                          );

     protected :
          virtual ~RLE_Flux_Pts_Computed();

          RLE_Flux_Pts_Computed(INT dim,INT sz_buf);
          RLE_Pack_Of_Pts * _rle_pack;
};

class RLE_Flux_Interface : public RLE_Flux_Pts_Computed
{
      friend Flux_Pts_Computed * flx_interface
                (INT dim, Pack_Of_Pts::type_pack type,INT sz_buf);
      private :
          RLE_Flux_Interface(INT dim,INT sz_buf);
          virtual const Pack_Of_Pts * next();
          
};

extern Flux_Pts_Computed *  split_to_max_buf
                                (     
                                      Flux_Pts_Computed *,
                                      const  Arg_Flux_Pts_Comp &   arg
                                );


class Arg_Flux_Pts_Comp
{
      public :
        Arg_Flux_Pts_Comp() ;
        Arg_Flux_Pts_Comp(INT SzBuf) ;
        inline INT sz_buf()  const { return _sz_buf;}
      private :
        INT _sz_buf;
     
};

class Flux_Pts_Not_Comp  : public RC_Object
{    
   public :
     virtual Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp &) = 0;
};



/**************************************************************/
/**************************************************************/
/**************************************************************/

template <class Type> void bitm_marq_line 
                       (Type ** im,INT tx,INT ty,Pt2di p1,Pt2di p2,INT val);
template <class Type> void bitm_marq_line 
                       (Type ** im,INT tx,INT ty,Pt2di p1,Pt2di p2,INT val,REAL width);



#endif /* !  _ELISE_FLUX_PTS */











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
