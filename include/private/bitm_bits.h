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



#ifndef _ELISE_BITM_BITS_H
#define _ELISE_BITM_BITS_H


class Tabul_Bits_Gen
{
      public :
         virtual INT   kieme_val      (INT byte,INT k) const  = 0;
         virtual INT   set_kieme_val  (INT old_byte,INT val,INT k) const = 0;

         static void  init_tabul_bits();
         virtual void  input(INT * out,const U_INT1 * in,INT x0,INT x1) const = 0;
         virtual void  output(U_INT1 * out,const  INT * in,INT x0,INT x1) const = 0;
         virtual void  output(U_INT1 * out,const REAL * in,INT x0,INT x1) const = 0;

	 virtual ~Tabul_Bits_Gen() {}

         static const Tabul_Bits_Gen & tbb(INT nbb,bool msbf);

         static void  unpack
                      (
                         U_INT1 *          out,
                         const U_INT1 *    in,
                         INT               nb,
                         INT               nbb,
                         bool              msbf
                      );

         static void  pack
                      (
                         U_INT1 *          out,
                         const U_INT1 *    in,
                         INT               nb,
                         INT               nbb,
                         bool              msbf
                      );


};

template <const INT nbb>  class DataGenImBits;

template<const INT nbb,const bool msbf> class Tabul_Bits : public Tabul_Bits_Gen
{
    friend PRE_CLASS Tabul_Bits_Gen;
    friend PRE_CLASS DataGenImBits<nbb>;

    public  :

        enum 
        {
              nb_per_byte = 8/nbb,
              nb_val      = 1<<nbb
        };

        static INT sz_line(INT nb_el);

        typedef U_INT1  tLineInputTab[nb_per_byte];
        typedef U_INT1  tLineOutputTab[nb_val][nb_per_byte];

        static  tLineInputTab *  input_tab;
        static  tLineOutputTab * out_tab;

        // static U_INT1  input_tab[256][nb_per_byte];
        // static U_INT1  out_tab[256][nb_val][nb_per_byte];


         void  input(INT * out,const U_INT1 * in,INT x0,INT x1) const;
         void  output(U_INT1 * out,const  INT * in,INT x0,INT x1) const ;
         void  output(U_INT1 * out,const REAL * in,INT x0,INT x1) const ;

         virtual INT   kieme_val      (INT byte,INT k) const;
         virtual INT   set_kieme_val  (INT old_byte,INT val,INT k) const;


        static  Tabul_Bits<nbb,msbf> The_Only_One;
    private:
        Tabul_Bits(int ArgBid);
        static void init_tabul();
};


template <const INT nbb>  class DataGenImBits : public DataGenIm
{
   public :

      DataGenImBits(INT sz_0,INT sz_tot,void * DataLin);  // sz_tot do not incopoarte first dim
      virtual void  input_rle(void *,INT,const void*,INT offs_0) const;


      virtual void  out_rle(void *,INT,const INT*,INT offs_0) const;
      virtual void  out_rle(void *,INT,const REAL*,INT offs_0) const;
      virtual INT sz_tot() const;



       void SetAll(INT aVal);
      bool          mDataLinOwner;
      INT           _sz_tot;
      INT           _sz_line; 
      U_INT1 *      _data_lin; // for afficionados of manipulations
                               // like _data_lin[x+y*_tx]
                               //  use is disrecommended with images

      virtual INT vmax() const;
      virtual INT vmin() const;
      virtual bool integral_type() const;
      virtual GenIm::type_el type() const;


      // max and min integral values of the type: the convention
      // v_max == v_min is used when these values are useless:

      protected :
          virtual ~DataGenImBits();


     // All these methods, generate fatal errors; there are defined for
     // compatibility with ``DataGenIm'', but, as far as I know, can never
     // be reached

      virtual void  striped_input_rle(void *,INT nb,INT dim,const void*,INT offs_0) const;
      virtual void  striped_output_rle(void *,INT nb,INT dim,const void*,INT offs_0) const;
      virtual INT sz_el() const;
      virtual INT sz_base_el() const;
      virtual void  *  data_lin_gen();

      static GenIm::type_el     type_el_bitm;
};



template <const INT nbb> class DataIm2D_Bits :
             public  DataGenImBits<nbb> ,
             public  DataIm2DGen
{
    public :

        enum 
        {
              nb_per_byte = Tabul_Bits<nbb,true>::nb_per_byte,
              nb_val      = Tabul_Bits<nbb,true>::nb_val     

        };

       void SetAll(INT aVal);

       inline INT get(INT x,INT y) const;
       inline INT get_def(INT x,INT y,INT v) const;
       inline void set(INT x,INT y,INT val) const;

      virtual void out_pts_integer(Const_INT_PP coord,INT nb,const void *) ;
      virtual void input_pts_integer(void *,Const_INT_PP coord,INT nb) const;
      virtual void input_pts_reel(REAL *,Const_REAL_PP coord,INT nb) const;

      virtual void  *   calc_adr_seg(INT *);
      virtual ~DataIm2D_Bits();


      virtual INT    dim() const;
      virtual const INT  * p0()  const;
      virtual const INT  * p1()  const;

      U_INT1 **     _data;

      Im2D_U_INT1  gray_im_red(INT & zoom);


      DataIm2D_Bits(INT tx, INT ty,bool to_init,INT v_init,void *);

      virtual void out_assoc
              (
                  void * out, // eventually 0
                  const OperAssocMixte &,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
              ) const;

      virtual void q_dilate
                   (  Std_Pack_Of_Pts<INT> * set_dilated,
                      char **,
                      const Std_Pack_Of_Pts<INT> * set_to_dilate,
                      INT **,
                      INT   nb_v,
                      Image_Lut_1D_Compile   func_selection,
                      Image_Lut_1D_Compile   func_update
                   );

};


// Methodes deplacees dans le header suite a des erreurs de compilation sous MacOS entre gcc4.0 et gcc4.2 (LLVM)
// du type : error: explicit specialization of 'TheType' after instantiation


template <const INT nbb>  void DataGenImBits<nbb>::input_rle
(void * v_out,INT nb,const void* v_in,INT offs_0) const
{
    
    Tabul_Bits<nbb,true>::The_Only_One.input
    (
     C_CAST(INT *,v_out),
     C_CAST(const U_INT1 *,v_in),
     offs_0,
     offs_0+nb
     );
}

template <const INT nbb>  void DataGenImBits<nbb>::out_rle
(void * v_out,INT nb,const INT * v_in,INT offs_0) const
{
    
    Tabul_Bits<nbb,true>::The_Only_One.output
    (
     C_CAST(U_INT1 *,v_out),
     v_in,
     offs_0,
     offs_0+nb
     );
}

template <const INT nbb>  void DataGenImBits<nbb>::out_rle
(void * v_out,INT nb,const REAL * v_in,INT offs_0) const
{
    
    Tabul_Bits<nbb,true>::The_Only_One.output
    (
     C_CAST(U_INT1 *,v_out),
     v_in,
     offs_0,
     offs_0+nb
     );
}

template <const INT nbb>  GenIm::type_el  DataGenImBits<nbb>::type() const
{
    return type_el_bitm;
}




#endif // _ELISE_BITM_BITS_H

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
