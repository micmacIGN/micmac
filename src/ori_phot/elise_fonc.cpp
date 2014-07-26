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






/********************************************************/
/********************************************************/
/***                                                  ***/
/***              To_Phot_3Std_Comp                   ***/
/***                                                  ***/
/********************************************************/
/********************************************************/

class To_Phot_3Std_Comp : public Fonc_Num_Comp_TPL<REAL>
{

    public  :
       To_Phot_3Std_Comp
       (
               const Arg_Fonc_Num_Comp & arg,
               Fonc_Num_Computed * f,
               Flux_Pts_Computed * flux,
               Data_Ori3D_Gen *    do3
       )  :
          Fonc_Num_Comp_TPL<REAL>(arg,2,flux),
          _f (f),
          _do3 (do3)
       {
       }
       virtual ~To_Phot_3Std_Comp() {delete  _f;}

    private :

       Fonc_Num_Computed *    _f;
       Data_Ori3D_Gen *     _do3;


       const Pack_Of_Pts * values(const Pack_Of_Pts * pack) 
       {
             const Std_Pack_Of_Pts<REAL> * ter = _f->values(pack)->real_cast();
             _pack_out->set_nb(pack->nb());
             _do3->to_photo
             (
                _pack_out->_pts[0],
                _pack_out->_pts[1],
                ter->_pts[0],
                ter->_pts[1],
                ter->_pts[2],
                pack->nb()
             );
             return _pack_out;
       }
};

class To_Phot_3Std_Not_Comp  : public Fonc_Num_Not_Comp
{
     public :

        To_Phot_3Std_Not_Comp(Fonc_Num f,Ori3D_Gen o3d) :
              _o3d (o3d),
              _f   (f)
        {
        }

     private :


        bool  integral_fonc (bool ) const { return false;}
        INT dimf_out() const { return 2;}
        void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

        Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
        {
              Fonc_Num_Computed 
                     * f =_f.compute(Arg_Fonc_Num_Comp(arg.flux()));

              Tjs_El_User.ElAssert
              (
                  f->idim_out() == 3,
                  EEM0 << "need 3-D function for Ori3D_Std::to_photo\n"
              );

              return new To_Phot_3Std_Comp(arg,f,arg.flux(),_o3d.dog());
        }



        Ori3D_Gen         _o3d;
        Fonc_Num            _f;
        
};

Fonc_Num  Ori3D_Std::to_photo(Fonc_Num f)
{
       return new To_Phot_3Std_Not_Comp(Rconv(f),*this);
}


/****************************************************************/
/*                                                              */
/*              Fnum_O3d_phot_et_z_to_terrain                   */
/*                                                              */
/****************************************************************/

class Fnum_O3d_petp_to_terrain  : public Simple_OP_UN<REAL>
{
     public :
           Fnum_O3d_petp_to_terrain (Data_Ori3D_Std * anOri1,Data_Ori3D_Std * anOri2,bool isCarto) :
              mOri1  (anOri1),
              mOri2  (anOri2),
              mCarto (isCarto)
           {
           }

     private :

         Data_Ori3D_Std *  mOri1;
         Data_Ori3D_Std *  mOri2;
         bool              mCarto;

         virtual void  calc_buf
         (
                           REAL ** output,
                           REAL ** input,
                           INT        nb,
                           const Arg_Comp_Simple_OP_UN  &
         ) ;
};

void Fnum_O3d_petp_to_terrain ::calc_buf
     (
              REAL ** output,
              REAL ** input,
              INT        nb,
              const Arg_Comp_Simple_OP_UN  &
     )
{
   REAL * xIm1 = input[0];
   REAL * yIm1 = input[1];
   REAL * xIm2 = input[2];
   REAL * yIm2 = input[3];

   REAL * xRes = output[0];
   REAL * yRes = output[1];
   REAL * zRes = output[2];
   REAL * Dist = output[3];

   for (INT aK=0 ; aK<nb ; aK++)
   {
       Pt3dr aRes  =  mOri1->to_terrain
                      (
                         Pt2dr(xIm1[aK],yIm1[aK]),
                         *mOri2,
                         Pt2dr(xIm2[aK],yIm2[aK]),
                         Dist[aK]
                      );

       if (mCarto)
          aRes = mOri1->terr_to_carte(aRes);

       xRes[aK] = aRes.x;
       yRes[aK] = aRes.y;
       zRes[aK] = aRes.z;
   }
}

Fonc_Num Data_Ori3D_Std ::petp_to_3D
                    (
                       Pt2d<Fonc_Num>     aPtIm1,
                       Data_Ori3D_Std *   ph2,
                       Pt2d<Fonc_Num>     aPtIm2,
                       bool isCarto

                   )
{
    return create_users_oper
           (
                0,
                new Fnum_O3d_petp_to_terrain(this,ph2,isCarto),
                Virgule(aPtIm1.x,aPtIm1.y,aPtIm2.x,aPtIm2.y),
                4
           );
}


Fonc_Num Data_Ori3D_Std ::petp_to_terrain
                  (
                       Pt2d<Fonc_Num> aPtIm1,
                       Data_Ori3D_Std *  ph2,
                       Pt2d<Fonc_Num> aPtIm2
                  )
{
   return petp_to_3D(aPtIm1,ph2,aPtIm2,false);
}
Fonc_Num Data_Ori3D_Std ::petp_to_carto
                  (
                       Pt2d<Fonc_Num> aPtIm1,
                       Data_Ori3D_Std *   ph2,
                       Pt2d<Fonc_Num> aPtIm2
                  )
{
   return petp_to_3D(aPtIm1,ph2,aPtIm2,true);
}




/****************************************************************/
/*                                                              */
/*              Fnum_O3d_phot_et_z_to_terrain                   */
/*                                                              */
/****************************************************************/


class Fnum_O3d_phot_et_z_to_terrain  : public Simple_OP_UN<REAL>
{

    public  :
        Fnum_O3d_phot_et_z_to_terrain(Ori3D_Std o3,int sz_buf) :
             _o3 (o3),
             _xtmp (sz_buf>=0 ? NEW_TAB(sz_buf,REAL) : 0),
             _ytmp (sz_buf>=0 ? NEW_TAB(sz_buf,REAL) : 0)
        {
        }
         virtual  ~Fnum_O3d_phot_et_z_to_terrain()
         {
             if (_xtmp)
             {
                 DELETE_TAB(_xtmp);
                 DELETE_TAB(_ytmp);
             }
         }
    private :
         virtual void  calc_buf
         (
                           REAL ** output,
                           REAL ** input,
                           INT        nb,
                           const Arg_Comp_Simple_OP_UN  &
         ) ;


         virtual  Simple_OP_UN<REAL> *  dup_comp(const Arg_Comp_Simple_OP_UN & arg)
         {
                 return new Fnum_O3d_phot_et_z_to_terrain(_o3,arg.sz_buf());
         }

         Ori3D_Std  _o3;
         REAL * _xtmp;
         REAL * _ytmp;

};


void Fnum_O3d_phot_et_z_to_terrain::calc_buf
     (
          REAL ** output,
          REAL ** input,
          INT        nb,
          const Arg_Comp_Simple_OP_UN  & arg
     )    
{    
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 3,
        EEM0 << "need 3-d Func for phot_et_z_to_terrain"
    );  
    _o3.dos()->photo_et_z_to_terrain
    (
        output[0],
        output[1],
        _xtmp,
        _ytmp,
        input[0],
        input[1],
        input[2],
        nb
    );  
    convert(output[2],input[2],nb);
}   


Fonc_Num Ori3D_Std::photo_et_z_to_terrain(Fonc_Num f)
{
    return create_users_oper
           (
                0,
                new Fnum_O3d_phot_et_z_to_terrain(*this,-1),
                f,
                3
           );
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
