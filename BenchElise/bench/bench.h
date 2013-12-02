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

#ifndef ELISE_BENCH_H
#define ELISE_BENCH_H



#if (ELISE_windows)
#define ELISE_NAME_DIR   "./"
#else   // So, Unix
#define ELISE_NAME_DIR "./"
#endif                 

#if (ELISE_windows)
#define  DIRECT_NAME  "\\"
#define ELISE_SYS_RM "del "
#else   // So, Unix
#define DIRECT_NAME "/"
#define ELISE_SYS_RM "rm "
#endif                    


#define  ELISE_NAME_DATA_DIR ELISE_NAME_DIR  "data" DIRECT_NAME
#define  ELISE_BFI_DATA_DIR  ELISE_NAME_DIR "BENCH_FILE_IM" DIRECT_NAME   


const REAL epsilon = 1e-7;
const REAL BIG_epsilon = 1e-4;
const REAL GIGANTESQUE_epsilon = 1e-1;


INT sigma_inf_x(INT x);
INT som_x(INT x1,INT x2);

#define USE_X11  1


void bench_bits_flow();
void somme_cste_int_rect_2d_by_hand();
void bench_Tab_CPT_REF();
void verif_som_coord_rect();
void verif_max_min();
void bench_cmpFN();
void verif_bitm();
void verif_flux();
void test_sigm_cat_coord();
void test_op_complex();
void test_chc();
void test_rect_kd();
void test_bitm_1d();
void bench_liste_pts();
void bench_border_rect();
void bench_flux_pts_user();
void bench_fnum_symb();
void bench_flux_line_map_rect();
void bench_filtr_line_map_rect();
void bench_shading();
void bench_bitm_win();
void test_im_mode_reel();
void test_box_seg();
void bench_op_buf_0();
void bench_op_buf_cat();
void bench_op_buf_1();
void bench_op_buf_2();
void bench_op_buf_3();
void bench_histo();
void bench_flux_geom();
void bench_oper_flux();
void bench_dist_chamfrain();
void bench_r2d_adapt_lin();
void bench_im2d_bits();
void bench_pnm();
void bench_tiff_im();
void bench_tiles_elise_file();

void bench_algo_spec();
void bench_red_op_ass();
void bench_orilib();
void bench_im3d();
void bench_user_oper();
void bench_sort();
void bench_zonec_dilate();
void bench_bugvecto();
void bench_rel_flag();
void bench_flag_front();
void bench_deriche();
void bench_algo_geo_0();
void bench_env_conv();
void bench_ext_stl_0();
void bench_im_tpl_0();
void bench_Telcopy_0();
void bench_graphe_elem();
void bench_algo_graphe_0();
void bench_vecteur_raster_0();
void bench_vecteur_raster_0();
void bench_delaunay() ;
void bench_qt_support() ;
void bench_qt();
void bench_deriv_formal();
void bench_matrix();
void bench_mep_relative();
void bench_least_square();

void BenchEqAppGrid();

void bench_optim_0();
void bench_optim_1();
void testElementsFinis();
void bench_GenerationCodeFormelle();

void bench_compr_im();

void bench_hough();
void bench_im_rle();
void bench_geo3d();

void  Bench_SommeSpec();

void bench_fft_correl();
void bench_epipole();
void  bench_auto_calibration();
void bench_cox();
void BenchRoy();
void BenchCorrSub();
void bench_nappes();
void bench_cLineMapRect();
void bench_pjeq234();

void bench_new_mep_rel();
void bench_xml();
void bench_Fraser();

void bench_reduc_image();

void bench_Proj32();

void bench_command();

#if (ELISE_unix)
#define RM "rm "
#else
#define RM "del "
#endif




#endif

