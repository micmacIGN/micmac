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



#ifndef _ELISE_FLUX_PTS_TPL_H
#define _ELISE_FLUX_PTS_TPL_H


class  Std_Pack_Of_Pts_Gen : public Pack_Of_Pts
{
       public :
            virtual void interv(const Std_Pack_Of_Pts_Gen * pck,INT n1,INT n2) = 0;

            // suppose the type of to_cat is the same than this
            virtual void cat_gen (const Std_Pack_Of_Pts_Gen * to_cat) = 0;
           
            virtual void rgb_bgr() = 0;
            virtual void rgb_bgr(const Std_Pack_Of_Pts_Gen *) = 0;

            
       protected :
            Std_Pack_Of_Pts_Gen(INT dim,INT _sz_buf,type_pack);


};

template <class Type> class Std_Pack_Of_Pts : public Std_Pack_Of_Pts_Gen
{
    public :
       virtual ~Std_Pack_Of_Pts(void);
       virtual  void show (void) const; // debug purpose
       virtual  void show_kth (INT k) const; 
       static Std_Pack_Of_Pts<Type>  * new_pck(INT dim,INT sz_buf);

	   // give two dif name, elsewhere visual is confused
       virtual void convert_from_int(const Std_Pack_Of_Pts<INT> *);
       virtual void convert_from_real(const Std_Pack_Of_Pts<REAL> *);
       virtual void convert_from_rle(const RLE_Pack_Of_Pts *) ;


       //  _pts shoulde be private with a definition like
       // inline  Type ** pts() const {return _pts;}
       // but I have REALLY some problem with template instatiation
       // and g++

       Type ** _pts;
       virtual  void * adr_coord() const;

       virtual void select_tab(Pack_Of_Pts * pck,const INT * tab_sel) const;

       void interv(const Std_Pack_Of_Pts_Gen * pck,INT n1,INT n2);

       void push(Type *);

       Pack_Of_Pts * dup(INT sz_buf= -1) const; // -1 => sz_buf will be set to _nb

       virtual void  copy(const Pack_Of_Pts *);
       virtual void trans (const Pack_Of_Pts * pack, const INT * tr);

       //  cat this at the end of res, replace res by a new set of size
       //  sz_buf (at least) if res is unsuifficient (and delete res).
       //  Return-value is the res or the eventually new set creates.

       Std_Pack_Of_Pts<Type> * cat_and_grow
             (Std_Pack_Of_Pts<Type> *  res,INT new_sz_buf,bool & chang) const;

       void cat (const Std_Pack_Of_Pts<Type> * to_cat);
       virtual void cat_gen (const Std_Pack_Of_Pts_Gen * to_cat);

       void auto_reverse();
       virtual void  kth_pts(INT *,INT k) const;

       INT   proj_brd
             (
                      const Pack_Of_Pts * pck,
                      const INT * p0,
                      const INT * p1,
                      INT         rab    // 0 for INT, 1 for real
             );

       void verif_inside 
       (
            const INT * pt_min, const INT * pt_max,
            Type        rap_p0,
            Type        rap_p1
       ) const;

      virtual void rgb_bgr(const Std_Pack_Of_Pts_Gen *);

    private :
       Std_Pack_Of_Pts<Type> (INT dim,INT sz_buf);

       // _pts[d][i] == coordinate of the ith Pts in the dth dimension
       Tab_Prov<Tab_Prov<Type> *>   * _tprov_tprov;
       Tab_Prov<Type *>             * _tprov_ptr;

       static CONST_STAT_TPL Pack_Of_Pts::type_pack type_glob;

      virtual void rgb_bgr() ;

};


Std_Pack_Of_Pts<INT> * lpt_to_pack(ElList<Pt2di> l);

template <class Type> class Std_Flux_Of_Points : public Flux_Pts_Computed
{
    protected :

       Std_Flux_Of_Points    (INT dim,INT sz_buf,bool sz_buf_0 = 0);
       virtual ~Std_Flux_Of_Points();
       Std_Pack_Of_Pts<Type>   * _pack;

   private :
       static const Pack_Of_Pts::type_pack _type_pack;
};

#define Int_Pack_Of_Pts  Std_Pack_Of_Pts<INT>
#define Real_Pack_Of_Pts Std_Pack_Of_Pts<REAL>

template <class Type> class Std_Flux_Interface : public Std_Flux_Of_Points<Type>
{
    public :
 
       virtual const Pack_Of_Pts * next(); 
       Std_Flux_Interface (INT dim,INT sz_buf);
};
Flux_Pts_Computed * interface_flx_chc (Flux_Pts_Computed * flx,Fonc_Num_Computed * f);
Flux_Pts_Computed * flx_interface
                (INT dim, Pack_Of_Pts::type_pack type,INT sz_buf);

Flux_Pts_Computed * flx_interface(Flux_Pts_Computed *);

        // Conserve, eventually the i_rect_2d properties with good parameter
Flux_Pts_Computed * tr_flx_interface(Flux_Pts_Computed *,INT *);






class Trace_Digital_line
{
   public :

     Trace_Digital_line(Pt2di p1,Pt2di p2,bool conx_8,bool include_p2);
     Trace_Digital_line(){};

     // fill x and y with the nb next point, where nb is the maximun
     // value <= sz_buf, return nb;
     INT next_buf(INT *x,INT *y,INT sz_buf);

     void next_pt()
     {
        if (ElAbs(_delta+_delta_1)<ElAbs(_delta+_delta_2))
        {
           _p_cur += _u1;
           _delta += _delta_1;
        }
        else
        {
           _p_cur += _u2;
           _delta += _delta_2;
        }
        _nb_pts--;
     }
     Pt2di pcur () { return _p_cur;}
     INT nb_residu() {return _nb_pts;}

   private :
     Pt2di  _u;  // vector p1 p2
     Pt2di  _u1; // firts direction approximation
     Pt2di  _u2; // second ....
     Pt2di  _p_cur; // current point

     //  _delta, stores  the quantity (_p_cur-_p0) ^ _u, that is, to a scaling factor
     //  the distance between _p_cur and the line p1,p2. _delta1 (2), represent
     //  the variation of delta if you add _u2 to _p_cur

     // because, _u1,_u2 is alway anti-clockwise, 
     // we always have _delta1 =< 0 and _delta2 >= 0;

     INT _delta_1;
     INT _delta_2;
     INT _delta;

     INT _nb_pts;
};



#endif  /* ! _ELISE_FLUX_PTS_TPL_H */

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
