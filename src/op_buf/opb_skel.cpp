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

#define TY_SKEL U_INT1


class Skel_OPB_Comp : public Simple_OPBuf1<INT,U_INT1>
{
   public :

     Skel_OPB_Comp(L_ArgSkeleton larg,bool AvecDist) ;
     virtual ~Skel_OPB_Comp();


   private :

     void  calc_buf
           (
               INT **     output,
               U_INT1 *** input
           );

     virtual Simple_OPBuf1<INT,U_INT1> * dup_comp();

     L_ArgSkeleton      _larg;
     TY_SKEL  **        _im_init;
     TY_SKEL  ***       _skel;
     Pt2di              _sz;
     Im2D_U_INT2        _tmp;
     bool               _AvecDist;
};


Skel_OPB_Comp::~Skel_OPB_Comp() 
{
    if (_skel)
    {
        DELETE_TAB_MATRICE(_skel,dim_in(),Pt2di(0,0),_sz);
        DELETE_MATRICE(_im_init,Pt2di(0,0),_sz);
    }
}

Skel_OPB_Comp::Skel_OPB_Comp(L_ArgSkeleton larg,bool AvecDist) :
    _larg       (larg),
    _im_init    (0),
    _skel       (0),
    _sz         (0,0),
    _tmp        (1,1),
    _AvecDist   (AvecDist)
{
}


Simple_OPBuf1<INT,U_INT1> * Skel_OPB_Comp::dup_comp()
{
     Skel_OPB_Comp * soc = new Skel_OPB_Comp(_larg,_AvecDist);
     soc->_sz = Pt2di(x1Buf()-x0Buf(),y1Buf()-y0Buf());
     soc->_im_init =  NEW_MATRICE
                      (
                           Pt2di(0,0),
                           soc->_sz,
                           TY_SKEL
                      );

     soc->_skel = NEW_TAB_MATRICE
                  (
                      dim_in(),
                      Pt2di(0,0),
                      soc->_sz,
                      TY_SKEL
                  );
     soc->_tmp = Im2D_U_INT2(soc->_sz.x,soc->_sz.y);
     soc->_larg = soc->_larg+ArgSkeleton(TmpSkel(soc->_tmp));

     return soc;
}

void Skel_OPB_Comp::calc_buf(INT ** output,U_INT1 *** input)
{
     if (first_line_in_pack())
     {
         for (INT d =0; d < dim_in() ; d++)
         {
            U_INT1  **  l = input[d];

            for (int y = y0Buf() ; y < y1Buf() ; y++)
                 convert
                 (
                      _im_init[y-y0Buf()],
                      l[y]+x0Buf(),
                      x1Buf()-x0Buf()
                 );

            Skeleton
            (
                  _skel[d],
                  _im_init,
                  _sz.x,
                  _sz.y,
                  _larg
            );
         }
     }

     for (INT d =0; d < dim_in() ; d++)
         convert
         (
              output[d]+x0(),
              _skel[d][y_in_pack()-dy0()]-dx0(),
              tx()
         );

     if (_AvecDist)
         convert
         (
              output[1]+x0(),
              _im_init[y_in_pack()-dy0()]-dx0(),
              tx()
         );

}


Fonc_Num skeleton_gen(Fonc_Num f,INT max_d,L_ArgSkeleton larg,bool AvecDist)
{
      Data_ArgSkeleton   askel(10,10,larg);

      INT  d =  (max_d + 1)/2 + 2;  // euclid dist
      if  (askel._ang >0)
      {
          REAL da = (d * 1.2) / cos(atan(askel._ang+1)) +2;
          d = ElMax (d,round_up(da));
      }
      d =  ElMax(d,askel._surf+2);

      INT per_reaf = (INT) (3 * d) + 5;

      per_reaf = ElMax(per_reaf,500);
                

      return create_op_buf_simple_tpl
             (
                 new Skel_OPB_Comp(larg,AvecDist),
                 f,
                 AvecDist ? 2 : f.dimf_out(),
                 Box2di(d),
                 per_reaf
             );
}

Fonc_Num skeleton(Fonc_Num f,INT max_d,L_ArgSkeleton larg)
{
    return skeleton_gen(f,max_d,larg,false);
}

Fonc_Num skeleton_and_dist(Fonc_Num f,INT max_d,L_ArgSkeleton larg)
{
    Tjs_El_User.ElAssert
    (
        f.dimf_out()==1,
        EEM0 << "dim should equal 1 in skeleton_and_dist"
    );

    return skeleton_gen(f,max_d,larg,true);
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
