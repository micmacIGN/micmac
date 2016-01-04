/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/


#if (1)

#ifdef USE_NOYAU_MINI
#include "commun.h"
#endif

#include "StdAfx.h"


void bench_doc()
{
    Im1D_INT4  sel   (10,"0 0 0 0 1 0 1 1  1 0");
    Im1D_INT4  update(10,"0 0 0 0 2 0 3 3  3 0");

   Im2D_U_INT1  i2(20,20);

   Neigh_Rel r1 = i2.neigh_test_and_set(Neighbourhood::v4(),sel,update);

   Neigh_Rel r2 = i2.neigh_test_and_set
                  (
                      Neighbourhood::v4(),
                          NewLPt2di(Pt2di(4,2))+Pt2di(6,3)+Pt2di(7,3)+Pt2di(8,3),
                     10
                  );
}
/*************************************************************
    BUG :
       border_rect(Pt2di(3,3),Pt2di(4,3)) => renvoie 1 pt au lieu de 0;
   !!!!!!!!! A RAJOUTER :

  Pour les fichier generaux :

       *  rajouter un protocole pour savoir si :

             o un fichier est accesible en lecture

             o une dalle est writable


      * mieux gerer les fichier avec specif de creation quand
        ils existent deja; les possibilites :

          [1] ecraser et remplacer par un nouveau fichier
               aux nouvelles specif (politique actuelle)

          [2] generer une erreur quoiqu'il arrive

          [3] verifier si le format sepcifie est  exactement le 
              format actuel, si c'est le cas on ne fait rien
              et on renvoit le fichier existant; sinon on
              rechoisit une des solutions 1 ou 2;


    * RAJOUTER : UN FICHIER CONNAIT SON NOM pour les MESSAGE d'ERREURS !!!!!

  JE ME SUIS FAIS BAISER A CAUSE DE CA SUR 255-> Im Tiff a 16 NIVEAU !!!!

            * controle de valeur generique sur les output
            * utiliser ce controle sur les fichiers

            * rajouter une opition de "clipping" de valeur sur tous
              les outputs primitifs.

   BUG sur les palette circulaire en ps (pas effectue % 256)

          
           

   !!!!!!!!  A BENCHER  :

            * chc sur les output
            * inside()
            * interior() (?)

            * dilate,erod,close,open + extinc (non 32)

            * Gif en output

            *  GenIm::load_file

            * init d'image par string

            
   !!!!!!  idee "sympas" :
            * "fonctionaliser" la couleur dans les plots
            * faire des ouput primitifs pour gerer les Z-buffer


Version "optimisee de Identite" qui renvoie directemnet le paquet de
point si mode != RLE (+ verif dim OK)  ....


Sur OpBuf U_INT1, verifier domaine de valeur


Mettre PRC0 (RC_Object *) en private

Bencher les fonction rajouter ad-hoc pour la DGI

ElFifo :   bencher alloc desalloc copie en effectuant des operations bidons
sur une classes qui alloue



IDEE ALGO :

   *  faire une version rectangle buf de filtrage directionnel

   *  faire un filtre lineaire KTH

***************************************************************/



#define IGN_BENCH 0

#define PRIV_BENCH 1



#include "general/all.h"
#include "bench.h"


/*********** UTIL ********************************/



INT sigma_inf_x(INT x)
{
    return (x*(x-1)) / 2;
}

INT som_x(INT x1,INT x2)
{
    if (x1 > x2)
       ElSwap(x1,x2);
    return
         sigma_inf_x(x2) - sigma_inf_x(x1)
      ;
}

void exec_bench(void (*fbench)(),const char * name)
{
	 ncout() << "Enter " << name << "\n";
     All_Memo_counter MC_INIT;
     stow_memory_counter(MC_INIT);
     fbench();
     verif_memory_state(MC_INIT);
	 ncout() << "End Of  " << name << "\n";

}

#define  Macro_Exec_bench(Fonc)  exec_bench(Fonc,#Fonc)

/*************************************************************/


void  all()
{

     ncout () << "DEBUT DU BENCH \n";
     ELISE_DEBUG_USER = true;

/*
     Macro_Exec_bench(bench_reduc_image);
     Macro_Exec_bench(bench_tiff_im);
          Macro_Exec_bench(bench_ext_stl_0);
          Macro_Exec_bench( bench_qt);

      Macro_Exec_bench(bench_bits_flow);
      Macro_Exec_bench(BenchEqAppGrid);
      Macro_Exec_bench(bench_mep_relative);
      Macro_Exec_bench(bench_Fraser);

      Macro_Exec_bench(bench_xml);
      Macro_Exec_bench(bench_im_tpl_0);
      Macro_Exec_bench(bench_least_square);
      Macro_Exec_bench(bench_new_mep_rel);
      Macro_Exec_bench(bench_matrix);



          Macro_Exec_bench(Bench_SommeSpec);

          Macro_Exec_bench(bench_orilib);
          Macro_Exec_bench(bench_ext_stl_0);

          Macro_Exec_bench(bench_liste_pts);
      // Macro_Exec_bench(bench_GenerationCodeFormelle);
      // Macro_Exec_bench(bench_auto_calibration);

      Macro_Exec_bench(bench_cmpFN);
      // Macro_Exec_bench(testElementsFinis);
       Macro_Exec_bench(bench_deriv_formal);

     // Macro_Exec_bench(bench_rel_flag);

     // Macro_Exec_bench(bench_cox);
     // Macro_Exec_bench(BenchCorrSub);
     // Macro_Exec_bench(bench_rel_flag);
     // Macro_Exec_bench(bench_least_square);
*/


    //-Macro_Exec_bench(bench_deriche);
    //-getchar();
    //-bench_Proj32();

     Macro_Exec_bench(bench_command);

     for(int i =0; i<1 ; i++)
     {
	 /*
          //Macro_Exec_bench(bench_bits_flow);
          Macro_Exec_bench(bench_xml);
          Macro_Exec_bench(somme_cste_int_rect_2d_by_hand);
          Macro_Exec_bench(bench_Tab_CPT_REF);
          Macro_Exec_bench(verif_som_coord_rect);
          Macro_Exec_bench(verif_max_min);
          Macro_Exec_bench(bench_cmpFN);
          Macro_Exec_bench(verif_bitm);
          Macro_Exec_bench(verif_flux);
          Macro_Exec_bench(test_sigm_cat_coord);
          Macro_Exec_bench(test_op_complex);
          Macro_Exec_bench(test_chc);
          Macro_Exec_bench(test_rect_kd);
          Macro_Exec_bench(test_bitm_1d);
          Macro_Exec_bench(bench_liste_pts);
          Macro_Exec_bench(bench_border_rect);
          Macro_Exec_bench(bench_flux_pts_user);
          Macro_Exec_bench(bench_fnum_symb);
          Macro_Exec_bench(bench_flux_line_map_rect);
          Macro_Exec_bench(bench_filtr_line_map_rect);
          Macro_Exec_bench(bench_shading);
          Macro_Exec_bench(bench_bitm_win);
          Macro_Exec_bench(test_im_mode_reel);
          Macro_Exec_bench(test_box_seg);
          Macro_Exec_bench(bench_op_buf_0);
          Macro_Exec_bench(bench_op_buf_cat);
          Macro_Exec_bench(bench_op_buf_1);
          Macro_Exec_bench(bench_op_buf_2);
          Macro_Exec_bench(bench_op_buf_3);
          Macro_Exec_bench(bench_histo);
          Macro_Exec_bench(bench_flux_geom);
          Macro_Exec_bench(bench_oper_flux);
          Macro_Exec_bench(bench_dist_chamfrain);
          Macro_Exec_bench(bench_r2d_adapt_lin);
          Macro_Exec_bench(bench_im2d_bits);
          Macro_Exec_bench(bench_pnm);
	  
          Macro_Exec_bench(bench_tiff_im);
          Macro_Exec_bench(bench_tiles_elise_file); 
          Macro_Exec_bench(bench_algo_spec);
          Macro_Exec_bench(bench_red_op_ass);
          Macro_Exec_bench(bench_orilib);
          Macro_Exec_bench(bench_im3d);
          Macro_Exec_bench(bench_user_oper);
          Macro_Exec_bench(bench_sort);
          Macro_Exec_bench(bench_zonec_dilate);
          Macro_Exec_bench(bench_rel_flag);
		
          Macro_Exec_bench(bench_flag_front);
          Macro_Exec_bench(bench_deriche);
          Macro_Exec_bench(bench_algo_geo_0);
          Macro_Exec_bench(bench_env_conv);
          Macro_Exec_bench(bench_ext_stl_0);
          Macro_Exec_bench(bench_im_tpl_0);
          Macro_Exec_bench( bench_Telcopy_0);
          Macro_Exec_bench(bench_graphe_elem);
          Macro_Exec_bench(bench_algo_graphe_0);
          Macro_Exec_bench(bench_delaunay);
          Macro_Exec_bench( bench_qt);
          Macro_Exec_bench( bench_qt_support);
          Macro_Exec_bench(bench_vecteur_raster_0);
          Macro_Exec_bench(bench_mep_relative);
          Macro_Exec_bench(bench_GenerationCodeFormelle);
          
	  // Macro_Exec_bench(bench_least_square);
          Macro_Exec_bench(bench_matrix);
          Macro_Exec_bench(bench_deriv_formal);
          Macro_Exec_bench(bench_optim_0);
          Macro_Exec_bench(bench_optim_1);
          Macro_Exec_bench(bench_compr_im);
          Macro_Exec_bench(bench_hough);
          Macro_Exec_bench(BenchEqAppGrid);

          Macro_Exec_bench(bench_nappes);
          Macro_Exec_bench(bench_cLineMapRect);

          Macro_Exec_bench(Bench_SommeSpec);
		  
          Macro_Exec_bench(bench_im_rle);
          Macro_Exec_bench(bench_geo3d);
          Macro_Exec_bench(bench_fft_correl);
          Macro_Exec_bench(bench_epipole);
          Macro_Exec_bench(BenchCorrSub);
          Macro_Exec_bench(bench_pjeq234);
          Macro_Exec_bench(bench_new_mep_rel);
          Macro_Exec_bench(bench_auto_calibration);
	  */
     }


     ncout () << "OK BENCH 0 \n";
	 EliseBRKP();
}


void bug_out()
{
printf("DEB bugout \n");
All_Memo_counter MC_INIT;
stow_memory_counter(MC_INIT);

{
    ELISE_COPY
    (
        rectangle(0,100),
        1,
        Output::onul()
    );
/*
    int x,y,z;
    Im1D_INT4 I4(100);
    Im1D_INT4 J4(100);
    Im1D_INT4 K4(100);
    ELISE_COPY
    (
        rectangle(0,100),
        (FX,1),
           Output::onul()
        |  (sigma(x) , VMax(y))
        |  I4.out()
        |  (J4.out() << (99-FX))
        |  (K4.out().chc(99-FX))
        |  (sigma(z).chc(99-FX))
    );
    ELISE_COPY
    (
        select(rectangle(0,100),1),
        (FX,1),
        K4.out()
    );
*/
}
verif_memory_state(MC_INIT);
printf( "FIN bugout \n");
}


//int main(int,char **)
void bench ()
{
    for (INT i = 0; i <1 ; i++)
    {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);
         bug_out();
         all();
         verif_memory_state(MC_INIT);
		 // pour forcer les consoles DOS
		 printf("ALL %d \n",i);
    }
	//return 0;
}

#if (USE_NOYAU)
MAIN_STANDARD (bench)
#else
int main(int,char **)
{
    bench();
}
#endif


/*
      b_00_0 : some suport class or function

           o   MSBitFirst_Flux_Of_VarLI
*/



/* 
   b_0_1 :

     *  somme de x et y sur un rectangle 2 d;
     *  verif de max et min sur un rectangle 2-d


   b_0_2 :
     * verif ecriture RLE sur une bitmap 2-d
     * verif addition avec types mixtes
     * verif de beaucoup d'operateurs arithmetiques
     * verif lecture RLE sur une bitmap 2-d
     * verif lecture RLE avec valeur par defaut sur des bitmap 2-d
       de differents types
     * verif lecture integer  avec valeur par defaut sur des bitmap 2-d
       de differents types

   b_0_3 :
     * verif de select + ecriture integer
     * verif de select + ecriture integer avec clip autom
     * verif de select + lecture integer
     * verif que, en mode pts integer, la convertion se fait bien

  b_0_4 :
     *  verif du cat de coordonnees (",") en intput (avec des sigma
        pour les output)
     *  verif que les operateurs marchent avec des image de resultat K-dim
     *  verif du cat de coordonnees (",") en output
     *  verif de la redirection "<<" sur les outputs
     *  verif du pipe dur les outputs

     * inside + clip_def

  b_0_5 :
     * verif multiplication sur les complexes;
     * verif carre sur les complexes;
     * verif polaire sur les complexes

  b_0_6 :
     * test de changement de coordonnes sur une image

  b_0_7 :
     * test sur les rectangle de dimension K

  b_0_8 :
     * test sur les images de dimension 1.

  b_0_9 :
     * liste de points.

  b_0_10 :
     *  rectangle creux de dimension 2.
     *  rectangle creux de dimension K.

  b_0_11 :
     *  flux de points specifie par des coordonnees

  b_0_12 :
     *  bench sur les symbole de fonction

  b_0_13 :
     *  bench sur les line_map_rect 

  b_0_14 :
     *  bench sur le shading

  b_0_15 :
     * bench sur les Bitm_Win
     * bench sur El_Window | El_Window
     * bench sur Elise_Palette::to_rgb

  b_0_16 :  Benchs sur les flux de points reels;
     *   test b2d interpole
     *   test fonc cooord reelle
     *   test fonc b2 interpole + clip def
     *   test b1d interpole
     *   test fonc b1 interpole + clip def

  b_0_17 :  Benchs sur des operations vecteurs :

          * cliping d'une box 

  b_0_18 :  Benchs sur des operateur rect buf :

          * gradient de robert
          * reduc assoc
      
  b_0_19 :  Benchs sur des operateur rect buf :

          * filtre exponentiel de canny
          * filtre kth sur un rectangle
          * rect_var_som

  b_0_19_1 :  Benchs sur des operateur rect buf :

           - flag vois
           - erosion homotopique
      
  b_0_19_2 :  Benchs sur des operateur rect buf :

           -  som_masq


  b_0_20 :  Benchs sur les output de type histogrammes

          * 

  b_0_21 :  Benchs sur cerles, disque, ellipses ,polygone

          * 
  b_0_22 :  Benchs sur les operateurs sur les flux :

          *  || (e.g. concatenation)

  b_0_23 :  Benchs sur les distances du chamfrein

  b_0_24 :  Benchs sur l'adaptation au rect 2D des filtres lineaires.

  b_0_25 :  Benchs sur les bitmaps sur 1,2 et 4 bits de dimension 2.

  b_0_26 :  Benchs sur les images tiff

  b_0_27 :  Benchs sur certains algo "ad hoc"

  b_0_28 :  Benchs sur reduction assoiciative de relation

  b_0_29 :  Benchs orilib (seulement bench de coherence)

  b_0_29 :  Benchs orilib (seulement bench de coherence)

  b_0_30 : bench_im3d

  b_0_31 : bench sur les "user's operator"
  
  b_0_32 : bench sur les tri

  b_0_33 : bench sur zonec/dilate

  b_0_34   : skel + vecto
  b_0_34_1 : frontiere (+ vecto ?)

  b_0_35 : deriche (vs Tuan Dang version)

  b_0_36 : approx + dist seg-droite 
  b_0_36_1 : enveloppe convexe

  
  b_0_37 : structure de donnees de bases de ext_stl

  b_0_38 : fonction d32 de im_tpl

  b_0_39 : Flux,Fonc,Out de im_tpl

  b_0_40 : manipulation elementaire sur les graphe
           (ajout , supression, iterateur, iterateur
            de sous-graphes);

  b_0_41 : faces, pcc

  b_0_42 : vecteur-raster

  b_0_43 : delaunay

  b_0_44 : quod-tree
  b_0_45 : quod-tree

  b_0_46 : photogrametrie

  b_0_47 : optimisation
*/



#else

//#include "general/all.h"
//#include <iostream>
//#include <>

/*
#include "commun.h"

#include <strstream>

#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <string>
#include <float.h>


#include <sys/types.h>
#include <sys/stat.h>

#include <new>

#include <strstream>


#include <iostream>
#include <ostream>
#include <fstream>

// using namespace std;


#include "general/sys_dep.h"

#include <vector>
#include <deque>
#include <list>
#include <string>
  */



int som_x(int x1,int x2)
{
   return x1+x2;
      ;
}


void test_elise()// (int,char **)
{
	printf("Coucou \n"); 
	std::cout << "coucou"<<endl;
	{
		char c; 
		std::cin >> c;
	}
	//return 1;
}

MAIN_STANDARD ( test_elise )


#endif
