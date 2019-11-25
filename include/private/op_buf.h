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



#ifndef _ELISE_PRIVATE_OP_BUF_H
#define _ELISE_PRIVATE_OP_BUF_H


INT adjust_nb_pack_y(INT nb_pack_y,INT nby_tot);


class Arg_FNOPB
{
    friend class Buf_Fonc_Op_buf;

    public :
        Arg_FNOPB(Fonc_Num,Box2di);
        Arg_FNOPB(Fonc_Num,Box2di,GenIm::type_el);
        static const Arg_FNOPB  def;

        bool  really_an_arg() const { return _f.really_a_fonc();}
        Box2di side() const { return _side;}
        GenIm::type_el type() const { return _type;}
		INT  DimF() const;

    private :
        Fonc_Num          _f;
        Box2di         _side;
        GenIm::type_el _type;
};

class Fonc_Num_OP_Buf  
{

      public  :
          friend class cOpBuf_Chc;
          static const Box2di  _bBid;
          virtual ~Fonc_Num_OP_Buf();
      protected  :

           Fonc_Num_OP_Buf
           (
                const Arg_Fonc_Num_Comp &,
                Arg_FNOPB               ar0,
                Arg_FNOPB               ar1,
                Arg_FNOPB               ar2,
				bool                    aLineInit
           );
           void maj_buf_values(INT y);


         REAL *** kth_buf(REAL *,INT k) ;
         INT  *** kth_buf(INT *,INT k)  ;
         U_INT1 *** kth_buf(U_INT1 *,INT k) ;
         INT1 *** kth_buf(INT1 *,INT k) ;
         U_INT2 *** kth_buf(U_INT2 *,INT k) ;
         REAL4 *** kth_buf(REAL4 *,INT k) ;

		 void  convert_data(INT Kth,INT * dest,INT nb,INT dim,INT y,INT x0Src);
		 void  convert_data(INT Kth,REAL * dest,INT nb,INT dim,INT y,INT x0Src);

		 INT  DimKthFonc(INT k);


         Box2di  kth_box_buf(INT k)  ;

         Fonc_Num_Computed * kth_fonc(INT k);

         INT _x0;
         INT _x1;
         INT _y0;
         INT _y1;
         INT _y_cur;

         INT            _x0_side;
         INT            _x1_side;
         INT            _y0_side;
         INT            _y1_side;

         INT            _x0_buf;
         INT            _x1_buf;
         INT            _y0_buf;
         INT            _y1_buf;



         INT                    _nbf;
      private :
         class Buf_Fonc_Op_buf  * _buf_foncs[3];


         bool                  _first;
         INT                   _last_y;


          //******************************************
          // Function to redefine in inherited classes
          //******************************************

           virtual void pre_new_line(bool first) ;  
                      // if first, line 0 is not valid

           virtual void post_new_line(bool first);
};



template <class Type> class Fonc_Num_OPB_TPL  : public Fonc_Num_Comp_TPL<Type>,
                                                public Fonc_Num_OP_Buf
{


        public  :
            // static void instantiate();

        protected :

           ~Fonc_Num_OPB_TPL();
           Fonc_Num_OPB_TPL
           (
                const Arg_Fonc_Num_Comp & arg,
                INT                     dim_out,
                Arg_FNOPB               = Arg_FNOPB::def,
                Arg_FNOPB               = Arg_FNOPB::def,
                Arg_FNOPB               = Arg_FNOPB::def,
				bool  aCatFoncInit      = false
           );

		   bool    mCatFoncInit;
		   INT     mDimOutSpec;
           Type ** _buf_res;



           virtual const Pack_Of_Pts *  values(const Pack_Of_Pts *);

          //******************************************
          // Function to redefine in inherited classes
          //******************************************
		  


};



//      User should use the version (in "include/general/users_op_buf.h")
//   without 5th parameter that fixes the value of compl
//  for  optimization. This version is declared here for bench purpose.

extern Fonc_Num label_maj
         (
            Fonc_Num     label,
            INT          vmax,
            Box2di       side,
            Fonc_Num     pds ,
            bool         /* compl  */
          );




#endif  //  _ELISE_PRIVATE_OP_BUF_H

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
