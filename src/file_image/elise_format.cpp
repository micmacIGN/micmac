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



/********************************************************************/
/*                                                                  */
/*                   Data_Elis_File_Im                              */
/*                                                                  */
/********************************************************************/

class Data_Elise_File_Im : public ElDataGenFileIm
{


   public :
      friend class DataGenIm;
      friend class Elise_File_Im;
      virtual ~Data_Elise_File_Im();

      Data_Elise_File_Im
      (
           const char *     name,
           INT        dim,     // 2 for usual images
           INT *      sz,      // tx,ty for usual images
           GenIm::type_el,     // U_INT1,INT ....
           INT        dim_out, // 1 for gray level, 3 for RVB ...
           tFileOffset        offset_0, // size of header to pass
           INT        szd0   ,// = -1,
           bool       create  // false
      );


      Tprov_char *     _tprov_name;
      char *           _name;
      INT              _dim;
      Tprov_INT  *     _tprov_sz;
      INT *            _sz;
      INT              _szd0;
      tFileOffset              _sz_tot;
      INT              _dim_out;
      GenIm::type_el   _type_el;
      GenIm::type_el   _type_el_if_bits;
      INT              _nbb; 
      tFileOffset      _offset_0;
      
      virtual   Fonc_Num in()     ;
      virtual   Fonc_Num in(REAL) ;
      virtual   Output out()      ;


};


Data_Elise_File_Im::Data_Elise_File_Im
(
               const char *     name,
               INT        dim,     
               INT *      sz,      
               GenIm::type_el type_el, 
               INT        dim_out, 
               tFileOffset        offset_0 ,
               INT        szd0,
               bool       create
)
{
       _tprov_name   = dup_name_std(name);
       _name         = _tprov_name->coord();
       _dim          = dim;
       _tprov_sz     = dup_sz_std(sz,dim);
       _sz           = _tprov_sz->coord();
       _dim_out      = dim_out;
       _type_el      = type_el;
       _nbb          = nbb_type_num(_type_el);
       _type_el_if_bits = (_nbb<8) ? GenIm::u_int1 : _type_el;
       _offset_0     = offset_0;

       INT nb_pb = ElMax(1,8/_nbb);
       if (szd0 == -1)
       {
           _szd0 =  _sz[0]  * dim_out;
           if (_nbb < 8)    // by default : padding
           {
              _szd0 = ((_szd0+nb_pb-1)/nb_pb) * nb_pb;
           }
       }
       else
          _szd0 = szd0;

       if (_nbb < 8)
           _sz_tot = _szd0 / nb_pb;
       else
            _sz_tot = _szd0 * (nbb_type_num(_type_el)/8);

       for (INT d = 1 ; d < dim ; d++)
           _sz_tot *= _sz[d];

       _sz_tot += _offset_0;


       if (create)
       {
          struct stat status;
          ELISE_fp::if_not_exist_create_0(name,&status);


          tFileOffset byte_to_add = _sz_tot - status.st_size;
          if (byte_to_add.BasicLLO()>0)
          {
             ELISE_fp fp(name,ELISE_fp::READ_WRITE);
             fp.seek_end(0);
             fp.write_dummy(byte_to_add);
             fp.close();
          }
       }

       ElDataGenFileIm::init
       (
           _dim,
           _sz,
           _dim_out,
           signed_type_num(_type_el),
           type_im_integral(_type_el),
           _nbb,
           _sz,
           false
       );

}


Data_Elise_File_Im::~Data_Elise_File_Im()
{
     delete _tprov_name;
     delete _tprov_sz;
}

Fonc_Num Data_Elise_File_Im::in()     
{
    return Elise_File_Im(this).in();
}

Fonc_Num Data_Elise_File_Im::in(REAL val) 
{
    return Elise_File_Im(this).in(val);
}

Output   Data_Elise_File_Im::out()
{
    return Elise_File_Im(this).out();
}


/********************************************************************/
/*                                                                  */
/*                   Gen_Elise_File_Im_Comp                         */
/*                                                                  */
/********************************************************************/

class Gen_Elise_File_Im_Comp 
{
   protected :
     Gen_Elise_File_Im_Comp 
         (Data_Elise_File_Im *,bool read,Flux_Pts_Computed *);

     INT calc_offset(const RLE_Pack_Of_Pts *);
     void read_write(const RLE_Pack_Of_Pts *);
     virtual ~Gen_Elise_File_Im_Comp();

     Packed_Flux_Of_Byte    *_pfob;
     Data_Elise_File_Im *  _fi;
     INT                   _last_offset;
     bool                  _read;
     GenIm                 _buf_im;
     DataGenIm             *_bim;
     INT                   _sz_el;
};

Gen_Elise_File_Im_Comp::Gen_Elise_File_Im_Comp
         (Data_Elise_File_Im *fi, bool read,Flux_Pts_Computed * flux ) :

           _fi (fi),
           _last_offset (0),
           _read (read),
           _buf_im   (alloc_im1d(fi->_type_el_if_bits,
                                 fi->_dim_out * flux->sz_buf())
                     )
{
     _pfob = new Std_Packed_Flux_Of_Byte
                 (
                       fi->_name,
                       1,
                       _fi->_offset_0,
                       read ? ELISE_fp::READ : ELISE_fp::READ_WRITE
                  );

     if (fi->_nbb < 8)
     {
        _pfob = new BitsPacked_PFOB
                (
                     _pfob,
                     fi->_nbb,
                     msbf_type_num(fi->_type_el),
                     read,
                     1
                 );
     }
    _bim = _buf_im.data_im();
    _sz_el = _bim->sz_el();
}


INT Gen_Elise_File_Im_Comp::calc_offset(const RLE_Pack_Of_Pts * pack)
{
    INT * pts  = pack->pt0();
    INT offset = pts[pack->dim()-1];
    
    for (INT d = pack->dim()-2; d >0 ; d--)
        offset = offset * _fi->_sz[d] + pts[d];

    return  (offset * _fi->_szd0  + pts[0] * _fi->_dim_out)
            *  _sz_el;
}

Gen_Elise_File_Im_Comp::~Gen_Elise_File_Im_Comp()
{
     delete _pfob;
}

void Gen_Elise_File_Im_Comp::read_write(const RLE_Pack_Of_Pts * pck_in)
{
     INT offset = calc_offset(pck_in);
     if (offset != _last_offset)
         _pfob->Rseek(offset-_last_offset);

     if (_read)
        _pfob->Read
        (
             (U_INT1 *) _bim->data_lin_gen(),
             _sz_el*pck_in->nb()*_fi->_dim_out
        );
      else
        _pfob->Write
         (     
             (U_INT1 *) _bim->data_lin_gen(),     
             _sz_el*pck_in->nb()*_fi->_dim_out     
         );
 
     _last_offset  = offset+_sz_el*pck_in->nb()*_fi->_dim_out;
}
          
/********************************************************************/
/********************************************************************/
/********************************************************************/
/*****                                                           ****/
/*****         INPUT                                             ****/
/*****                                                           ****/
/********************************************************************/
/********************************************************************/
/********************************************************************/


/********************************************************************/
/*                                                                  */
/*                   Elise_File_Im_In_Comp                          */
/*                                                                  */
/********************************************************************/

template <class TypeBase>
     class  Elise_File_Im_In_Comp : public Gen_Elise_File_Im_Comp ,
                                    public Fonc_Num_Comp_TPL<TypeBase>
{
     public :
       Elise_File_Im_In_Comp(const Arg_Fonc_Num_Comp &,Data_Elise_File_Im *,Flux_Pts_Computed *);
     private :
       const Pack_Of_Pts * values(const Pack_Of_Pts *);
};


template <class TypeBase> Elise_File_Im_In_Comp<TypeBase>::Elise_File_Im_In_Comp
          (   const Arg_Fonc_Num_Comp & arg,
              Data_Elise_File_Im *fi,
              Flux_Pts_Computed * flux
          )  :
         
             Gen_Elise_File_Im_Comp(fi,true,flux),
             Fonc_Num_Comp_TPL<TypeBase>(arg,_fi->_dim_out,flux)
{
}

template <class TypeBase>
     const Pack_Of_Pts * Elise_File_Im_In_Comp<TypeBase>::values(const Pack_Of_Pts * pack)
{
     RLE_Pack_Of_Pts * rle_pack = SAFE_DYNC( RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pack));

     if (!rle_pack->inside(PTS_00000000000000,_fi->_sz))
     {
        rle_pack->Show_Outside(PTS_00000000000000,_fi->_sz);
        ASSERT_USER
        (
           false,
          "outside reading file in RLE mode"
        );
     }


    read_write(rle_pack);

    _bim->striped_input_rle
     (
          this->_pack_out->_pts,
          pack->nb(),
          _fi->_dim_out,
          _bim->data_lin_gen(),
          0
     );
    this->_pack_out->set_nb(pack->nb());
    return this->_pack_out;
}

/********************************************************************/
/*                                                                  */
/*                   Elise_File_Im_In_Not_Comp                      */
/*                                                                  */
/********************************************************************/

class Elise_File_Im_In_Not_Comp : public Fonc_Num_Not_Comp
{
       public :
          Elise_File_Im_In_Not_Comp
          (  
             Elise_File_Im,
             Data_Elise_File_Im *,
             bool              with_def_value,
             REAL              def_value
          );

       private :

          Elise_File_Im         _pfi;
          Data_Elise_File_Im *  _fi;
          bool                  _with_def_value;
          REAL                  _def_value;

          virtual bool  integral_fonc (bool) const 
          {return type_im_integral(_fi->_type_el);}

          virtual INT  dimf_out () const {return _fi->_dim_out;}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
         {
               Fonc_Num_Computed * res;

               ASSERT_TJS_USER
               (
                   arg.flux()->type() == Pack_Of_Pts::rle,
                   "Can only handle RLE mode for File-Images"
               );
               if  (type_im_integral(_fi->_type_el)) 
                   res = new Elise_File_Im_In_Comp<INT>(arg,_fi,arg.flux());
               else
                   res = new Elise_File_Im_In_Comp<REAL>(arg,_fi,arg.flux()); 

              return 
                _with_def_value                                               ?
                clip_fonc_num_def_val
                     (arg,res,arg.flux(),PTS_00000000000000,_fi->_sz,_def_value)  :
                res                                                           ;
         }
};


Elise_File_Im_In_Not_Comp::Elise_File_Im_In_Not_Comp
(  
     Elise_File_Im         pfi,
     Data_Elise_File_Im *  fi,
     bool                  with_def_value,
     REAL                  def_value
) :
   _pfi              (pfi),
   _fi               (fi),
   _with_def_value   (with_def_value),
   _def_value        (def_value)
{
}

/********************************************************************/
/********************************************************************/
/********************************************************************/
/*****                                                           ****/
/*****         OUTPUT                                            ****/
/*****                                                           ****/
/********************************************************************/
/********************************************************************/
/********************************************************************/

/********************************************************************/
/*                                                                  */
/*                   Elise_File_Im_Out_Comp                         */
/*                                                                  */
/********************************************************************/

class  Elise_File_Im_Out_Comp : public Gen_Elise_File_Im_Comp ,
                                public Output_Computed
{
     public :
       Elise_File_Im_Out_Comp(Data_Elise_File_Im *,const Arg_Output_Comp &);


     private :
       void update (const Pack_Of_Pts *,const Pack_Of_Pts *);


//  Because of uncomprehensible warning
     Elise_File_Im_Out_Comp & operator =(const class Elise_File_Im_Out_Comp &);

};


Elise_File_Im_Out_Comp::Elise_File_Im_Out_Comp
          (   Data_Elise_File_Im *     fi,
              const Arg_Output_Comp & arg
          )  :
         
             Gen_Elise_File_Im_Comp(fi,false,arg.flux()),
             Output_Computed(fi->_dim_out)
{
}

void Elise_File_Im_Out_Comp::update
     (const Pack_Of_Pts * pts,const Pack_Of_Pts * val)
{
     const RLE_Pack_Of_Pts * rle_pts = SAFE_DYNC(const RLE_Pack_Of_Pts *,pts);

     ASSERT_USER
     (
        rle_pts->inside(PTS_00000000000000,_fi->_sz),
       "outside writing file in RLE mode"
     );

    _bim->striped_output_rle
    (
          _bim->data_lin_gen(),
          pts->nb(),
          _fi->_dim_out,
          val->adr_coord(),
          0
    );
    read_write(rle_pts);
}


/********************************************************************/
/*                                                                  */
/*                   Elise_File_Im_Out_Not_Comp                     */
/*                                                                  */
/********************************************************************/


class Elise_File_Im_Out_Not_Comp : public Output_Not_Comp
{
       public :
          Elise_File_Im_Out_Not_Comp
          (  
             Elise_File_Im,
             Data_Elise_File_Im *,
             bool              cliped
          );

       private :

          Elise_File_Im         _pfi;
          Data_Elise_File_Im *  _fi;
          bool                  _cliped;

         virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
         {
               Output_Computed * res;

               ASSERT_TJS_USER
               (
                   arg.flux()->type() == Pack_Of_Pts::rle,
                   "Can only handle RLE mode for File-Images"
               );

               Tjs_El_User.ElAssert
               (
                    arg.fonc()->idim_out() >= _fi->_dim_out,
                    EEM0  
                     << "dimension of function insufficient for file writing\n"
                     << "|    File " << _fi->_name  << "\n"
                     << "|   Fonc dim : " <<  arg.fonc()->idim_out()
                     << "  File dim " <<  _fi->_dim_out 
                     
               );

               res = new Elise_File_Im_Out_Comp(_fi,arg);
               res = out_adapt_type_fonc
                     (
                        arg,
                        res,
                        type_im_integral(_fi->_type_el) ?
                              Pack_Of_Pts::integer  :
                              Pack_Of_Pts::real
                     );
              if (_cliped)
                 res = clip_out_put(res,arg,PTS_00000000000000,_fi->_sz);
              return  res;
         }
};


Elise_File_Im_Out_Not_Comp::Elise_File_Im_Out_Not_Comp
(  
     Elise_File_Im         pfi,
     Data_Elise_File_Im *  fi,
     bool                  cliped
) :
   _pfi           (pfi),
   _fi            (fi),
   _cliped        (cliped)
{
}


/********************************************************************/
/*                                                                  */
/*                   Elise_File_Im                                  */
/*                                                                  */
/********************************************************************/

Elise_File_Im::~Elise_File_Im(){}

Elise_File_Im::Elise_File_Im 
(
      Data_Elise_File_Im * defi
)   :
    ElGenFileIm(defi)
{
}
    


Elise_File_Im::Elise_File_Im 
(
               const char *     name,
               INT        dim,     
               INT *      sz,      
               GenIm::type_el type_el,
               INT        dim_out, 
               tFileOffset        offset_0,
               INT        szd0 ,
               bool       create
) :
   ElGenFileIm
   (
        new Data_Elise_File_Im
           (name,dim,sz,type_el,dim_out,offset_0,szd0,create)
   )
{
}

Elise_File_Im::Elise_File_Im
(
         const char *     name,
         INT              sz,     
         GenIm::type_el type,     
         tFileOffset    offset_0 ,
         bool       create
)  :  ElGenFileIm(0)
{
   *this = Elise_File_Im(name,1,&sz,type,1,offset_0,-1,create);
}


Elise_File_Im::Elise_File_Im
(
         const char *     name,
         Pt2di      sz,     
         GenIm::type_el type,     
         tFileOffset    offset_0 ,
         bool       create
)  :  ElGenFileIm(0)
{
   INT txy[2];
   sz.to_tab(txy);

   *this = Elise_File_Im(name,2,txy,type,1,offset_0,-1,create);
}


Elise_File_Im::Elise_File_Im
(
         const char *     name,
         Pt3di      sz,     
         GenIm::type_el type,     
         tFileOffset    offset_0 ,
         bool       create
)  :  ElGenFileIm(0)
{
   INT txy[3];
   sz.to_tab(txy);

   *this = Elise_File_Im(name,3,txy,type,1,offset_0,-1,create);
}


Fonc_Num Elise_File_Im::in()
{
    return new Elise_File_Im_In_Not_Comp(*this,defi(),false,0.0);
}

Fonc_Num Elise_File_Im::in(REAL val)
{
    return new Elise_File_Im_In_Not_Comp(*this,defi(),true,val);
}

Output Elise_File_Im::out()
{
   return new Elise_File_Im_Out_Not_Comp(*this,defi(),true);
}

Output Elise_File_Im::onotcl()
{
     return new Elise_File_Im_Out_Not_Comp(*this,defi(),true);

}


Elise_Rect Elise_File_Im::box() const
{

    return Elise_Rect(PTS_00000000000000,defi()->_sz,defi()->_dim);
}

void DataGenIm::load_file(Elise_File_Im f,GenIm i)
{
    Data_Elise_File_Im * def = f.defi();

    Tjs_El_User.ElAssert
    (
          def->_dim_out==1,
          EEM0 << "Requires dim out =1 for load_file "
    );

    Tjs_El_User.ElAssert
    (
          def->_dim==dim(),
          EEM0 << "Incompatible dim in load_file "
               << "(" <<  def->_dim << " and " << dim() << ")"
    );

    for (INT k = 0;  k<dim(); k++)
        Tjs_El_User.ElAssert
        (
              def->_sz[k]==p1()[k],
              EEM0 << "Incompatible sz in load_file for dim " << k
                   << " (" <<  def->_sz[k] << " and " << p1()[k] << ")"
        );

   if (   (type() != def->_type_el)
        ||(nbb_type_num(type()) < 8)
      )
   {
      ELISE_COPY (i.all_pts(),f.in(),i.out());
   }
   else
   {
            
        ELISE_fp   fp(def->_name,ELISE_fp::READ);
        fp.seek_begin(def->_offset_0);
        fp.read ( data_lin_gen(),def->_sz_tot,1);
        fp.close();
   }
}


Elise_Tiled_File_Im_2D   Elise_File_Im::to_elise_tiled(bool  byte_ordered)
{
	return Elise_Tiled_File_Im_2D
		   (
               defi()->_name,
				p1(),
			    type_el(),
                NbChannel(),
                p1(),
                false,  // clip last
                true,  // chunk
                defi()->_offset_0,
		false,
                byte_ordered
           );
}

Tiff_Im Elise_File_Im::to_tiff(bool byte_ordered)
{
	return to_elise_tiled(byte_ordered).to_tiff();
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
