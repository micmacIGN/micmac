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
#ifndef _ELISE_GENERAL_H_HOUGH_H
#define _ELISE_GENERAL_H_HOUGH_H


//////////////////////////////////////////////////////////////////////

class H_Hough
{
    public:

      H_Hough(  REAL     pas_rho=1,
                REAL     pas_teta=1,
                INT      nb_min = 20,
                REAL     debut_teta=0,
                REAL     fin_teta = 180
             );

      H_Hough(  Im2D_U_INT1 contour,
                bool        cont,
                Output      w,
                INT         color,
                REAL        pas_rho=1,
                REAL        pas_teta=1,
                INT         nb_min = 20,
                REAL        debut_teta=0,
                REAL        fin_teta = 180
             );

      H_Hough(  Im2D_U_INT1 image,
                REAL        deriche,
                REAL        hyst_bas,
                REAL        hyst_haut,
                Output      w,
                INT         color,
                REAL        pas_rho=1,
                REAL        pas_teta=1,
                INT         nb_min = 20,
                REAL        debut_teta=0,
                REAL        fin_teta = 180
             );

      void accumuler(Im2D_U_INT1 contour, bool cont = true);

      Im2D_REAL8 accumulator();
      ElList<Pt2di> max_loc(INT nb_max = 5);
      ElList<Pt2di> max_loc_p();
      ElList<Pt2di > droites();
      void tracer(Output W,  INT color = 7);
      void tracer(Output W, Pt2di p, INT color = 7);
      void tracer(Output W, ElList<Pt2di> l, INT color = 7);
      void tracer(Im2D_U_INT1 im);
      INT  nb_droite();

   private:

     Im2D_REAL8    _accumulator;
     Im2D_INT4     _accumulator_im;
     Im2D_U_INT1   _contour;
     Im1D_REAL8    _teta;
     Im1D_REAL8    _s_teta;
     Im1D_REAL8    _c_teta;
     Im1D_REAL8    _moderator;
     REAL8**       _accumulator_data;
     INT4**        _accumulator_im_data;
     U_INT1**      _contour_data;
     REAL8*        _teta_data;
     REAL8*        _s_teta_data;
     REAL8*        _c_teta_data;
     REAL8*        _moderator_data;
 
     ElList<Pt2di> _l;
     REAL          _pas_r;
     REAL          _pas_t;
     REAL          _debut_teta;
     REAL          _fin_teta;
     REAL          _intervale_teta;
     REAL          _merge_teta;
     INT           _tr;
     INT           _tt;
     INT           _semi_tr;
     INT           _tx;
     INT           _ty;
     INT           _nb_min;
     Pt2di         _centre;
     INT           _nb_droite;
     ElList<Pt2di> _l_droite;

     void accumuler(Pt2di p);
     void accumuler_im(Pt2di p);
     void desaccumuler(Pt2di p);
     void conversion(Pt2di p);

};


//////////////////////////////////////////////////////////////////////

class H_Hough_3D
{
    public:

      H_Hough_3D(
                   REAL     pas_rho    = 1, 
                   REAL     pas_teta   = 1, 
                   REAL     pas_phi    = 1, 
                   INT      nb_min     = 20, 
                   REAL     debut_teta = 0, 
                   REAL     fin_teta   = 180,
                   REAL     debut_phi  = 0, 
                   REAL     fin_phi    = 180
                );

      H_Hough_3D(
                   Im3D_U_INT1 entrer,
                   Boite    sortir,
                   REAL     pas_rho    = 1, 
                   REAL     pas_teta   = 1, 
                   REAL     pas_phi    = 1, 
                   INT      nb_min     = 20, 
                   REAL     debut_teta = 0, 
                   REAL     fin_teta   = 180,
                   REAL     debut_phi  = 0, 
                   REAL     fin_phi    = 180
             );

      void accumuler(Im3D_U_INT1 entrer, INT debut = 0, INT fin = 0);
      void accumuler(Im3D_U_INT1 entrer, Im2D_U_INT1 masq, INT debut = 0, INT fin = 0);
//      void accumuler(Liste_Pts_REAL nuage);

      Im3D_REAL8 accumulator();
      ElList<Pt3di> max_loc(INT nb_max = 6, Output W = NULL, INT d_teta = 3, INT d_phi = 2, INT d_rho = 1); 
      void tracer(Boite& sortir);
      void tracer(Boite& sortir, Pt3di p, INT color = 1);
      void tracer(Boite& sortir, ElList<Pt3di> l);
      Hplan conversion(Pt3di p);
      ElList<Hplan> conversion(ElList<Pt3di> l);
      ElList<Hplan> plans();

   protected:

     Im3D_REAL8    _accumulator;
     Im3D_REAL8    _accumulator_im;
     Im1D_REAL8    _teta;
     Im1D_REAL8    _phi;
     Im2D_REAL8    _c_phi_s_teta;
     Im2D_REAL8    _c_phi_c_teta;
     Im2D_REAL8    _s_phi;
     U_INT1***     _entrer;
     REAL8***      _accumulator_data;
     REAL8***      _accumulator_im_data;
     REAL8*        _teta_data;
     REAL8*        _phi_data;
     REAL8**       _c_phi_s_teta_data;
     REAL8**       _c_phi_c_teta_data;
     REAL8**       _s_phi_data;

     ElList<Pt3di> _l;
     ElList<Hplan> _lplan;
     REAL          _pas_r;
     REAL          _pas_t;
     REAL          _pas_p;

     REAL          _debut_teta;
     REAL          _fin_teta;
     REAL          _intervale_teta;
     REAL          _merge_teta;

     REAL          _debut_phi;
     REAL          _fin_phi;
     REAL          _intervale_phi;
     REAL          _merge_phi;

     INT           _tr;
     INT           _tt;
     INT           _tp;
     INT           _semi_tr;
     INT           _semi_tp;
     INT           _tx;
     INT           _ty;
     INT           _tz;
     INT           _nb_min;
     Pt3di         _centre;

     void accumuler(Pt3di p);
     void accumuler_im(Pt3di p);
     void accumuler_nuage(Pt3dr p);
     void desaccumuler(Pt3di p);

};


class H_Hough_mne:public H_Hough_3D
{
    public:

      H_Hough_mne(
                   INT      diff       = 1,
                   REAL     pas_rho    = 1, 
                   REAL     pas_teta   = 1, 
                   REAL     pas_phi    = 1, 
                   INT      nb_min     = 20, 
                   REAL     debut_teta = 0, 
                   REAL     fin_teta   = 180,
                   REAL     debut_phi  = 0, 
                   REAL     fin_phi    = 180
                );

      H_Hough_mne(
                   Mne&     mne,
                   INT      diff       = 1,
                   REAL     pas_rho    = 1, 
                   REAL     pas_teta   = 1, 
                   REAL     pas_phi    = 1, 
                   INT      nb_min     = 20, 
                   REAL     debut_teta = 0, 
                   REAL     fin_teta   = 180,
                   REAL     debut_phi  = 0, 
                   REAL     fin_phi    = 180
             );

      void accumuler(Mne& mne, INT diff = 1);
      ElList<Pt3di> max_loc(INT nb_max = 6, Output W = NULL, INT d_teta = 3, INT d_phi = 2, INT d_rho = 1); 

   protected:

      INT _diff;
};



#endif //_ELISE_GENERAL_H_HOUGH_H

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
