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



#ifndef _ELISE_PRIVATE_MORPHO_H
#define _ELISE_PRIVATE_MORPHO_H

Liste_Pts_U_INT2 Skeleton
(
     U_INT1 **    result,
     U_INT1 **    image,
     int                 tx,
     int                 ty,
     int                 surf_threshlod,
     double              angular_threshlod,
     bool                skel_of_disk,
     bool                prolgt_extre,
     bool                Withresult,
     bool                cx8,
     U_INT2 **           tmp
);

Fonc_Num binarize(Fonc_Num f,INT val,bool neg = false);


Liste_Pts_U_INT2 Skeleton
(
     U_INT1 **  skel,
     U_INT1 **  image,
     INT        tx,
     INT        ty,
     L_ArgSkeleton  larg
);

class Data_ArgSkeleton
{
      public :
          Data_ArgSkeleton(INT tx,INT ty,L_ArgSkeleton);

          INT        _tx;
          INT        _ty;
          REAL       _ang;
          INT        _surf;
          bool       _skel_of_disk;
          bool       _prolgt_extre;
          bool       _result;
          bool       _cx8;
          U_INT2 **  _tmp;
};




typedef struct ResultVeinSkel
{
      unsigned short *  x;
      unsigned short *  y;
      int               nb;
} ResultVeinSkel;
void freeResultVeinSkel(ResultVeinSkel *);

ResultVeinSkel VeinerizationSkeleton
(
     unsigned char **    out,
     unsigned char **    in,
     int                 tx,
     int                 ty,
     int                 surf_threshlod,
     double              angular_threshlod,
     bool                skel_of_disk,
     bool                prolgt_extre,
     bool                with_result,
     bool                cx8,
     unsigned short **   tmp
);

const unsigned char * NbBitsOfFlag();

template <class Type> class SkVeinIm;




class SkVein
{

    public :




    // protected :

        SkVein();
        static inline int abs(int v){ return (v<0) ? -v : v;}
        static inline int ElMin(int v1,int v2) {return (v1<v2) ? v1 : v2;}
        static inline int ElMax(int v1,int v2) {return (v1>v2) ? v1 : v2;}
        static inline void set_min(int &v1,int v2) {if (v1 > v2) v1 = v2;}
        static inline int square(int x) {return x*x;}

        enum
        {
           sz_brd = 3
        };


       enum
       {
           pds_x = 3,
           pds_y = pds_x * 3,
           pds_tr_x = pds_y * 3,
           pds_tr_y = pds_tr_x * 7,
           pds_moy  = pds_tr_y * 7,
           pds_cent  = pds_moy * 29,
           pds_Mtrx = pds_moy + pds_tr_x,
           pds_Mtry = pds_moy + pds_tr_y,
           fact_corr_diag = pds_cent / 3,
           pds_out = -1000
       };


        // Data structure to have list of direction
        class ldir
        {
           public :
             ldir (ldir * next,int dir);

             ldir * _next;
             int     _dir;
             int     _x;
             int     _y;
             int     _flag_dsym;
             int     _flag;
        };

        // for a given flag, gives the list of active bits
        static ldir * BitsOfFlag[256];

        // for a given flag,  the number of bits active
        static U_INT1    NbBitFlag[256];
        // for a given flag,  =1 if NbBitFlag == 1 and 0 elsewhere
        static U_INT1    FlagIsSing[256];

        static U_INT1    UniqueBits[256];

        class cc_vflag
        {
           public :
              ldir       * _dirs;
              cc_vflag * _next;
              cc_vflag(cc_vflag * next,int flag);

              static cc_vflag * lcc_of_flag(int & flag,bool cx8,cc_vflag *);
        };


        static cc_vflag * CC_cx8[256];
        static cc_vflag * CC_cx4[256];

        static void flag_cc(int & cc,int b,int flag);


        static int v8x[8];
        static int v8y[8];

        static int v4x[4];
        static int v4y[4];

        static U_INT1 flag_dsym[8];
        static U_INT1 dir_sym[8];
        static U_INT1 dir_suc[8];
        static U_INT1 dir_pred[8];


// Bench purpose
        static void show(ldir *);
        static void show(cc_vflag *);
        static void showcc(int i,bool cx8);

    private :
        static bool _init;
};


class  SKVRLE
{
     public :
        inline SKVRLE(INT x0,INT x1);
        inline SKVRLE();
        inline INT x0();
        inline INT x1();
        inline INT nbx();

     private :
        INT _x0,_x1;
};

template <class Type> class SkVeinIm : public SkVein
{

         static const int max_val;
         static const int min_val;
     public :

         SkVeinIm(Type ** d,int tx,int ty);
         ~SkVeinIm();

          void close_sym(ElPartition<SKVRLE> &);
          void open_sym(ElPartition<SKVRLE> &);
          void set_brd(Type val,int width);
          void binarize(ElPartition<SKVRLE> &,Type val);
          void dist32(ElPartition<SKVRLE> &,int vmax = max_val-1);
          void pdsd32vein1l
              (INT4 * pds,ElPartition<SKVRLE> &,INT4 y,INT4 *def);
          void d32_veinerize 
               (ElPartition<SKVRLE> &,SkVeinIm<U_INT1>  veines,bool cx8);
          void reset();
          void perm_circ_lines();

          void push_extr(ElPartition<SKVRLE> &,ElFifo<Pt2di> &);
          void push_extr(ElPartition<SKVRLE> &,ElFifo<Pt2di> &,U_INT2 ** som);

          void get_arc_sym(ElFifo<Pt2di> &);
          bool get_arc_sym(INT4 x,INT4 y);
          void  prolong_extre(ElPartition<SKVRLE> &);
          Pt2di  prolong_extre(int x,int y,int k);
          void  prolong_extre_rec(int x,int y,int k,int & nb,ElFifo<Pt3di> &f);

          void  pruning
          (
                  ElPartition<SKVRLE> &,
                  SkVeinIm<U_INT1>  veines,
                  U_INT2 **         som,
                  double            ang,
                  INT4              surf,
                  bool              skp,
                  ElFifo<Pt3di> &    Fgermes
          );

          void  isolated_points
          (
                  SkVeinIm<U_INT1>  veines,
                  ElFifo<Pt3di> &    Fgermes
          );

        int  calc_surf(Pt2di );


     // private :

         Type **           _d;
         INT4     _tx;
         INT4     _ty;
         bool           _free;

        // to pass as paremeter in calc_surf
        double _ang;
        INT4   _surf;
        U_INT1  ** _dvein;
};



#endif // _ELISE_PRIVATE_MORPHO_H

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
