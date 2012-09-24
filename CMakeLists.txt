cmake_minimum_required( VERSION 2.8.1 )
project( MICMAC )

set(VERBOSE 0)

set (MICMAC_VERSION_MAJOR 1)
set (MICMAC_VERSION_MINOR 0)

# Use Kakadu
OPTION(WITH_KAKADU  "Include KAKADU Support" OFF)
# Use Matis orientation
OPTION(WITH_MATIS_ORI "Include Matis orrientation" OFF )
# Use Subversion
OPTION(WITH_SVN "Use SVN" OFF )

IF(${WITH_SVN})
	FIND_PACKAGE(Subversion)
	  IF(SUBVERSION_FOUND)
	    Subversion_WC_INFO(${PROJECT_SOURCE_DIR} MICMAC)
	    MESSAGE("Current revision is ${MICMAC_WC_REVISION}")
	    Subversion_WC_LOG(${PROJECT_SOURCE_DIR} MICMAC)
	    MESSAGE("Last changed log is ${MICMAC_LAST_CHANGED_LOG}")
	  ELSE()
		MESSAGE("Impossible de trouver la version SVN")
		set(MICMAC_WC_REVISION 0)
	ENDIF()

	IF(NOT ${PROJECT_NAME}_CONFIG_HAS_BEEN_RUN_BEFORE)
		UNSET(CMAKE_INSTALL_PREFIX CACHE)
		SET(CMAKE_INSTALL_PREFIX ${${PROJECT_NAME}_SOURCE_DIR} CACHE PATH "")
	ENDIF()
ENDIF()



MESSAGE("---- PROJET MICMAC ----")

Message("Options :")

IF(${WITH_KAKADU})
	MESSAGE("	- Kakadu support")
ELSE()
	MESSAGE("	- No Kakadu support")
ENDIF()

IF(${WITH_MATIS_ORI})
	MESSAGE("	- Matis orientation support")
ELSE()
	MESSAGE("	- No Matis orientation support")
ENDIF()


CONFIGURE_FILE (
  "include/general/MM_InstalDir.h.in"
  "${PROJECT_BINARY_DIR}/include/general/MM_InstalDir.h"
  )

INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)

IF(${WITH_KAKADU})

	FIND_PACKAGE( KAKADU )
	IF(NOT KAKADU_FOUND)
		MESSAGE("Warning! Kakadu not found. Micmac will be built without JPEG2000 support.")
	ELSE()
		ADD_DEFINITIONS(-D__USE_JP2__)
		LINK_DIRECTORIES( ${KAKADU_LIBRARY_DIRS})
		INCLUDE_DIRECTORIES( ${KAKADU_INCLUDE_DIRS})
		MESSAGE("KAKADU : ${KAKADU_LIBRARIES}")
	ENDIF()

ENDIF()

IF(${WITH_MATIS_ORI})
	FIND_PACKAGE( MatisOrientation )
	IF(NOT MatisOrientation_FOUND)
		MESSAGE("Warning! MatisOrientation not found. Micmac will be built without MatisOrientation support.")
	ELSE()
		OPTION(MICMAC_USE_PROJ4 "Use PROJ4" OFF)
		#-- recherche de boost
		SET(Boost_USE_MULTITHREADED ON)
		FIND_PACKAGE( Boost 1.46.0 COMPONENTS unit_test_framework system filesystem serialization thread  REQUIRED )
		FIND_PACKAGE(TICPP REQUIRED)
		IF ( TICPP_FOUND)
			LINK_DIRECTORIES(${TICPP_LIBRARY_DIRS})
			INCLUDE_DIRECTORIES(${TICPP_INCLUDE_DIRS})
		ENDIF()
	
		IF (MICMAC_USE_PROJ4)
			FIND_PACKAGE( PROJ4 REQUIRED )
			LINK_DIRECTORIES(${PROJ4_LIBRARY_DIRS})
			INCLUDE_DIRECTORIES(${PROJ4_INCLUDE_DIRS})
		ENDIF()
	
		ADD_DEFINITIONS(-D__USE_ORIENTATIONMATIS__ )
		LINK_DIRECTORIES(${MatisOrientation_LIBRARY_DIRS})
		INCLUDE_DIRECTORIES(${MatisOrientation_INCLUDE_DIRS})

	ENDIF()
ENDIF()


FIND_PACKAGE(X11)
IF (X11_FOUND)
	MESSAGE("X11 Found : ${X11_INCLUDE_DIRS} | ${X11_LIBRARIES} | ${X11_LIBRARY_DIRS}")
ELSE()
	MESSAGE("X11 Not Found : pas de SaisieMasq")
ENDIF()

IF (MSVC)
	add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /nologo" )
ENDIF()

INCLUDE_DIRECTORIES(
	${PROJECT_SOURCE_DIR}/include
)

IF (CMAKE_COMPILER_IS_GNUCC)
	ADD_DEFINITIONS(-O2)
ENDIF()

add_library( elise 
src/AMD/amd_1.cpp
src/AMD/amd_2.cpp
src/AMD/amd_aat.cpp
src/AMD/amd_control.cpp
src/AMD/amd_defaults.cpp
src/AMD/amd_demo_1.cpp
src/AMD/amd_dump.cpp
src/AMD/amd_global.cpp
src/AMD/amd_info.cpp
src/AMD/amd_order.cpp
src/AMD/amd_post_tree.cpp
src/AMD/amd_postorder.cpp
src/AMD/amd_preprocess.cpp
src/AMD/amd_valid.cpp
CodeGenere/File2String/Str_DefautChantierDescripteur.cpp
CodeGenere/File2String/Str_ParamChantierPhotogram.cpp
CodeGenere/File2String/Str_SuperposImage.cpp
src/HassanArrangt/cElHJSol3D.cpp
src/HassanArrangt/cElHJaArrangt.cpp
src/HassanArrangt/cElHJaArrangt_Visu.cpp
src/HassanArrangt/cElHJaAttrArcPlani.cpp
src/HassanArrangt/cElHJaAttrSomPlani.cpp
src/HassanArrangt/cElHJaDroite.cpp
src/HassanArrangt/cElHJaFacette.cpp
src/HassanArrangt/cElHJaPlan3D.cpp
src/HassanArrangt/cElHJaPoint.cpp
src/HassanArrangt/cElHJaSomEmpr.cpp
src/HassanArrangt/cElHJa_InstanceGraphe.cpp
src/HassanArrangt/cFullSubGrWithP.cpp
src/SAN/cylindre.cpp
src/SAN/ortho_cyl.cpp
src/SimulPhgr/cSimulCamera.cpp
src/XML_GEN/ParamChantierPhotogram.cpp
src/XML_GEN/SuperposImage.cpp
src/algo_speciaux/deriche.cpp
src/algo_speciaux/hongr.cpp
src/algo_speciaux/hongrois.cpp
src/algo_speciaux/opb_deb_phase.cpp
src/api/vecto_hough.cpp
src/api/vecto_skel.cpp
src/bitm/bitm_op_ass.cpp
src/bitm/cCpleImMasq.cpp
src/bitm/comput.cpp
src/bitm/comput_tpl.cpp
src/bitm/flx_out_im2.cpp
src/bitm/font_bitm.cpp
src/bitm/im2d_bits.cpp
src/bitm/im2d_reech_grid.cpp
src/bitm/im2d_tpl.cpp
src/bitm/im3d_tpl.cpp
src/bitm/im_compr.cpp
src/bitm/imxd.cpp
src/bitm/lpts.cpp
src/bitm/matrix.cpp
src/bitm/pj_eq234.cpp
src/bitm/polyn2DReel.cpp
src/bitm/polynome.cpp
src/bitm/real_poly.cpp
src/bitm/scale_im_compr.cpp
src/compression/huffman.cpp
src/compression/lzw.cpp
src/compression/pack_bits.cpp
src/compression/range_code.cpp
src/correl/correl_2D.cpp
src/correl/correl_init.cpp
src/correl/correl_special.cpp
src/file_image/bmp.cpp
src/file_image/elise_format.cpp
src/file_image/fich_2d_gen.cpp
src/file_image/gef.cpp
src/file_image/gif.cpp
src/file_image/pnm.cpp
src/file_image/tga.cpp
src/flux_pts/curser_pack_pts.cpp
src/flux_pts/flx_gen_tpl.cpp
src/flux_pts/flx_general.cpp
src/flux_pts/flx_oper.cpp
src/flux_pts/front_to_surf.cpp
src/flux_pts/lineique_2d.cpp
src/flux_pts/rle_pack.cpp
src/flux_pts/select.cpp
src/flux_pts/surf_geom_2d.cpp
src/fonc_num/compos.cpp
src/fonc_num/coord.cpp
src/fonc_num/fcteurs_uti.cpp
src/fonc_num/fnum_compile.cpp
src/fonc_num/fnum_compile2.cpp
src/fonc_num/fnum_gen_tpl.cpp
src/fonc_num/fnum_general.cpp
src/fonc_num/linear_filter.cpp
src/fonc_num/linear_filtre_mne.cpp
src/fonc_num/linear_order.cpp
src/fonc_num/linear_proj.cpp
src/fonc_num/proj_brd.cpp
src/fonc_num/symb_fnum.cpp
src/geom2d/Shewchuk_api.cpp
src/geom2d/Shewchuk_implem.cpp
src/geom2d/basic.cpp
src/geom2d/dist.cpp
src/geom2d/env_conv.cpp
src/geom2d/gpc.cpp
src/geom2d/gpc_interf.cpp
src/geom2d/inter_cerle_losange.cpp
src/geom2d/intersection.cpp
src/geom2d/proj.cpp
src/geom2d/region_plan.cpp
src/geom2d/seg_comp.cpp
src/geom2d/triangle_comp.cpp
src/geom3d/NuageToGrid.cpp
src/geom3d/cElNuage3DMaille.cpp
src/geom3d/cElNuageLaser.cpp
src/geom3d/cGridNuageP3D.cpp
src/geom3d/cImplemElNuage3DMaille.cpp
src/geom3d/cMailageSphere.cpp
src/geom3d/cZBuffer.cpp
src/geom3d/geo3basic.cpp
src/hough/cPtOfCorrel.cpp
src/hough/hough_algo.cpp
src/hough/hough_basic.cpp
src/hough/hough_clip.cpp
src/hough/hough_extrac_basic.cpp
src/hough/hough_file.cpp
src/hough/hough_filtr_seg_basic.cpp
src/hough/hough_gen.cpp
src/hough/hough_inst_fseg.cpp
src/hough/hough_merger.cpp
src/hough/hough_sub_pixel.cpp
src/memory/cpt_ref.cpp
src/memory/liste.cpp
src/memory/new_cpt_ref.cpp
src/memory/smart_pointeur.cpp
src/memory/tab_prov.cpp
src/morpho_cabl/chamfer_dist.cpp
src/morpho_cabl/dequantif.cpp
src/morpho_cabl/morph_rle.cpp
src/morpho_cabl/skel_interface.cpp
src/morpho_cabl/skel_vein.cpp
src/mullgesuhlig/mubasic.cpp
src/mullgesuhlig/muflaguer.cpp
src/mullgesuhlig/mufmueller.cpp
src/mullgesuhlig/muvmblock.cpp
src/neighboor/b2d_spec_neigh.cpp
src/neighboor/dilat_conc.cpp
src/neighboor/neigh_filter.cpp
src/neighboor/neigh_general.cpp
src/neighboor/red_op_neigh.cpp
src/op_buf/opb_assoc.cpp
src/op_buf/opb_cany.cpp
src/op_buf/opb_chamfer.cpp
src/op_buf/opb_chscale.cpp
src/op_buf/opb_conc.cpp
src/op_buf/opb_etiq.cpp
src/op_buf/opb_flag.cpp
src/op_buf/opb_fonc_a_trou.cpp
src/op_buf/opb_general.cpp
src/op_buf/opb_max_loc_dir.cpp
src/op_buf/opb_ord.cpp
src/op_buf/opb_pts_interets.cpp
src/op_buf/opb_simple.cpp
src/op_buf/opb_skel.cpp
src/op_buf/opb_som_rvar.cpp
src/op_buf/users_opb.cpp
src/operator/assoc_mixte.cpp
src/operator/binarie_int.cpp
src/operator/binarie_mixte.cpp
src/operator/colour.cpp
src/operator/compar_op.cpp
src/operator/complex.cpp
src/operator/func_opbinmix_tpl.cpp
src/operator/func_opun_tpl.cpp
src/operator/inst_users_op.cpp
src/operator/math_op.cpp
src/operator/op_bin_ndim.cpp
src/operator/op_def.cpp
src/operator/oper_naire.cpp
src/operator/users_op.cpp
src/optim/cox_roy.cpp
src/optim/func_mean_square.cpp
src/optim/nappes.cpp
src/optim/opt_cube_flux.cpp
src/optim/opt_dmr_f1v.cpp
src/optim/opt_elgrowingsetind.cpp
src/optim/opt_forml1.cpp
src/optim/opt_mat_creuse.cpp
src/optim/opt_nvar.cpp
src/optim/opt_somme_formelle.cpp
src/optim/opt_sysl2.cpp
src/optim/opt_syssuresolu.cpp
src/ori_phot/astro.cpp
src/ori_phot/cOriMntCarto.cpp
src/ori_phot/elise_fonc.cpp
src/ori_phot/elise_interface.cpp
src/ori_phot/filmdist.cpp
src/ori_phot/lambgeo.cpp
src/ori_phot/matrices.cpp
src/ori_phot/orilib.cpp
src/ori_phot/orisol.cpp
src/output/oper.cpp
src/output/out_general.cpp
src/output/reduction.cpp
src/output/transform_intern.cpp
src/photogram/ChantierNameAssoc.cpp
src/photogram/DataBase.cpp
src/photogram/FiltrageSpatial_CNA.cpp
src/photogram/cParamEtal.cpp
src/photogram/phgr_Basculement.cpp
src/photogram/phgr_autocalibr.cpp
src/photogram/phgr_basic.cpp
src/photogram/phgr_cEqObsRotVect.cpp
src/photogram/phgr_cGereEtats.cpp
src/photogram/phgr_cGridIncImageMnt.cpp
src/photogram/phgr_correl_grid.cpp
src/photogram/phgr_correl_subpix.cpp
src/photogram/phgr_dist.cpp
src/photogram/phgr_ebner_brown_dist.cpp
src/photogram/phgr_ellipse.cpp
src/photogram/phgr_epip_cam.cpp
src/photogram/phgr_epipole.cpp
src/photogram/phgr_eqhomog.cpp
src/photogram/phgr_extrinseque.cpp
src/photogram/phgr_formel.cpp
src/photogram/phgr_homog.cpp
src/photogram/phgr_inconnue_tmp.cpp
src/photogram/phgr_intrinseque.cpp
src/photogram/phgr_laser_image.cpp
src/photogram/phgr_low_level.cpp
src/photogram/phgr_mep_rel.cpp
src/photogram/phgr_or_code_gen00.cpp
src/photogram/phgr_or_code_gen1.cpp
src/photogram/phgr_or_code_gen2.cpp
src/photogram/phgr_or_code_gen3.cpp
src/photogram/phgr_or_code_gen4.cpp
src/photogram/phgr_or_code_gen5.cpp
src/photogram/phgr_or_code_gen6.cpp
src/photogram/phgr_or_opt.cpp
src/photogram/phgr_or_param_extr.cpp
src/photogram/phgr_or_param_intr.cpp
src/photogram/phgr_orient_relative.cpp
src/photogram/phgr_polyg.cpp
src/photogram/phgr_pts_interet.cpp
src/photogram/phgr_sys_coord.cpp
src/photogram/phgr_triangu_appar_image.cpp
src/photogram/phgr_triangul.cpp
src/photogram/transfo_xml_phgr.cpp
src/photogram/transfo_xml_phgr2.cpp
src/plotter/draw_plot1d.cpp
src/plotter/gen_plot1.cpp
src/post_script/disp.cpp
src/post_script/dxf.cpp
src/post_script/filters.cpp
src/post_script/iocomp.cpp
src/post_script/prim_graph.cpp
src/post_script/win_bitm.cpp
src/post_script/wind.cpp
src/radiom/egal_radiom.cpp
src/ras_vect/prolong_seg_dr.cpp
src/ras_vect/seg_dr.cpp
src/recipes/refft.cpp
src/recipes/regaussj.cpp
src/recipes/rejacobi.cpp
src/recipes/resparse.cpp
src/tiff/cLazyTiffFile.cpp
src/tiff/tiff_args_opt.cpp
src/tiff/tiff_compr_tiles.cpp
src/tiff/tiff_header.cpp
src/tiff/tiff_nc_tiles.cpp
src/tiff/tiff_pckbit_un_load.cpp
src/tiff/tiff_tags.cpp
src/tiff/el_dcraw.c
src/top_level/copy.cpp
src/util/affin2d.cpp
src/util/all.cpp
src/util/arg_main.cpp
src/util/bicub.cpp
src/util/bits_flow.cpp
src/util/box.cpp
src/util/cElStatErreur.cpp
src/util/cEquiv1D.cpp
src/util/cGPAO.cpp
src/util/cSysCoor.cpp
src/util/current_fonc.cpp
src/util/dates.cpp
src/util/error.cpp
src/util/fifo.cpp
src/util/files.cpp
src/util/num.cpp
src/util/num_tpl.cpp
src/util/pt2di.cpp
src/util/randomm.cpp
src/util/regex.cpp
src/util/sort.cpp
src/util/string_dyn.cpp
src/util/stringifie.cpp
src/util/tabul.cpp
src/util/wildmatch.cpp
src/util/xml.cpp
src/util/xml2cpp.cpp
src/util/win_regex.c
src/vectorize/aprrox_poly.cpp
src/vectorize/cont_vect.cpp
src/vectorize/inermats.cpp
src/vectorize/sk_vect.cpp
src/xinterf/Ok_eLise.cpp
src/xinterf/bi_scroller.cpp
src/xinterf/bitm_win.cpp
src/xinterf/cElImageFlipper.cpp
src/xinterf/display.cpp
src/xinterf/fen_graph_window.cpp
src/xinterf/fen_windows.cpp
src/xinterf/fen_x11.cpp
src/xinterf/gen_win.cpp
src/xinterf/graphics.cpp
src/xinterf/image_interactor.cpp
src/xinterf/imfile_scrol.cpp
src/xinterf/incruster.cpp
src/xinterf/integ_out.cpp
src/xinterf/pop_up_menu_transp.cpp
src/xinterf/rle_out.cpp
src/xinterf/scroller.cpp
src/xinterf/xcolour.cpp
)

add_definitions(-DNO_JPEG -DNO_LCMS)

# MICMAC
add_executable(MICMAC
	CodeGenere/File2String/Str_ParamMICMAC.cpp
	applis/MICMAC/cAppliMICMAC_CorrelPonctuelle.cpp
	applis/MICMAC/Jp2ImageLoader.cpp
	applis/MICMAC/MICMAC.cpp
	applis/MICMAC/PartiesCachees.cpp
	applis/MICMAC/cAppliMICMAC.cpp
	applis/MICMAC/cAppliMICMAC_ChCorrel.cpp
	applis/MICMAC/cAppliMICMAC_GPU.cpp
	applis/MICMAC/cAppliMICMAC_MNE.cpp
	applis/MICMAC/cAppliMICMAC_MEC.cpp
	applis/MICMAC/cAppliMICMAC_Result1.cpp
	applis/MICMAC/cAppliMICMAC_Result2.cpp
	applis/MICMAC/cBasculeMnt.cpp
	applis/MICMAC/cCaracOfDeZoom.cpp
	applis/MICMAC/cEtapeMecComp.cpp
	applis/MICMAC/cFilePx.cpp
	applis/MICMAC/cGBV2_ProgDynOptimiseur.cpp
	applis/MICMAC/cGeomImage.cpp
	applis/MICMAC/cGeomXXX.cpp
	applis/MICMAC/cLoadedImage.cpp
	applis/MICMAC/cMicMacVisu.cpp
	applis/MICMAC/cModeleAnalytiqueComp.cpp
	applis/MICMAC/cNewProgDyn.cpp
	applis/MICMAC/cOptimisationDiff.cpp
	applis/MICMAC/cOrientationCon.cpp
	applis/MICMAC/cOrientationGrille.cpp
	applis/MICMAC/cOrientationRTO.cpp
	applis/MICMAC/cParamMICMAC.cpp
	applis/MICMAC/cPriseDeVue.cpp
	applis/MICMAC/cProgDynOptimiseur.cpp
	applis/MICMAC/cSimulation.cpp
	applis/MICMAC/cStatNDistrib.cpp
	applis/MICMAC/cStdTiffModuleImageLoader.cpp
	applis/MICMAC/cSurfaceOptimiseur.cpp
	applis/MICMAC/uti_MICMAC.cpp
	applis/MICMAC/OrthoLocAnam.cpp
)

target_link_libraries(MICMAC elise ${X11_LIBRARIES}   ${KAKADU_LIBRARIES}  ${TICPP_LIBRARIES} ${Boost_LIBRARIES})

IF ( MatisOrientation_FOUND)
	target_link_libraries(MICMAC elise ${MatisOrientation_LIBRARIES})
	IF (MICMAC_USE_PROJ4)
		target_link_libraries(MICMAC elise ${PROJ4_LIBRARIES})
	ENDIF()
ENDIF()


add_executable(MpDcraw 
        applis/MpDcraw/cArgMpDCRaw.cpp  
        applis/MpDcraw/cNChanel.cpp  
        applis/MpDcraw/cOneChanel.cpp  
        applis/MpDcraw/MpDcraw.cpp
)
target_link_libraries(MpDcraw elise ${X11_LIBRARIES})

#Outils uti_phgrm
add_executable(Malt applis/uti_phgrm/Malt.cpp)
target_link_libraries(Malt elise ${X11_LIBRARIES})

add_executable(Tapioca applis/uti_phgrm/Tapioca.cpp)
target_link_libraries(Tapioca elise ${X11_LIBRARIES})

add_executable(Pastis applis/uti_phgrm/Pastis.cpp)
target_link_libraries(Pastis elise ${X11_LIBRARIES})

add_executable(Nuage2Ply applis/uti_phgrm/Nuage2Ply.cpp)
target_link_libraries(Nuage2Ply elise ${X11_LIBRARIES})

add_executable(Pasta applis/uti_phgrm/Pasta.cpp)
target_link_libraries(Pasta elise ${X11_LIBRARIES})

add_executable(Bascule applis/uti_phgrm/Bascule.cpp)
target_link_libraries(Bascule elise ${X11_LIBRARIES})

add_executable(MakeGrid applis/uti_phgrm/MakeGrid.cpp)
target_link_libraries(MakeGrid elise ${X11_LIBRARIES})

add_executable(TestCam applis/uti_phgrm/TestCam.cpp)
target_link_libraries(TestCam elise ${X11_LIBRARIES})

add_executable(Tapas applis/uti_phgrm/Tapas.cpp)
target_link_libraries(Tapas elise ${X11_LIBRARIES})

add_executable(AperiCloud applis/uti_phgrm/AperiCloud.cpp)
target_link_libraries(AperiCloud elise ${X11_LIBRARIES})

#Outils uti_image
add_executable(ScaleIm applis/uti_image/ScaleIm.cpp)
target_link_libraries(ScaleIm elise ${X11_LIBRARIES})

add_executable(ElDcraw applis/uti_image/ElDcraw.cpp)
target_link_libraries(ElDcraw elise ${X11_LIBRARIES})

add_executable(to8Bits applis/uti_image/to8Bits.cpp)
target_link_libraries(to8Bits elise ${X11_LIBRARIES})

add_executable(GrShade applis/uti_image/GrShade.cpp)
target_link_libraries(GrShade elise ${X11_LIBRARIES})

add_executable(PastDevlop applis/uti_image/PastDevlop.cpp)
target_link_libraries(PastDevlop elise ${X11_LIBRARIES})

add_executable(Devlop applis/uti_image/Devlop.cpp)
target_link_libraries(Devlop elise ${X11_LIBRARIES})

add_executable(tiff_info applis/uti_image/tiff_info.cpp)
target_link_libraries(tiff_info elise ${X11_LIBRARIES})

add_executable(Dequant applis/uti_image/Dequant.cpp)
target_link_libraries(Dequant elise ${X11_LIBRARIES})

IF (X11_FOUND)
	add_executable(SaisieMasq applis/uti_image/SaisieMasq.cpp)
	target_link_libraries(SaisieMasq elise ${X11_LIBRARIES})
	INSTALL(TARGETS SaisieMasq
		RUNTIME DESTINATION bin
	)

	add_executable(SEL applis/uti_phgrm/SEL.cpp)
	target_link_libraries(SEL elise ${X11_LIBRARIES})
        INSTALL(TARGETS SEL
                RUNTIME DESTINATION bin
        )

	add_executable(MICMACSaisieLiaisons applis/MICMAC/SaisieLiaisons.cpp
		CodeGenere/File2String/Str_ParamMICMAC.cpp
        applis/MICMAC/cAppliMICMAC_CorrelPonctuelle.cpp
        applis/MICMAC/Jp2ImageLoader.cpp
        applis/MICMAC/PartiesCachees.cpp
        applis/MICMAC/cAppliMICMAC.cpp
        applis/MICMAC/cAppliMICMAC_ChCorrel.cpp
        applis/MICMAC/cAppliMICMAC_GPU.cpp
        applis/MICMAC/cAppliMICMAC_MNE.cpp
        applis/MICMAC/cAppliMICMAC_MEC.cpp
        applis/MICMAC/cAppliMICMAC_Result1.cpp
        applis/MICMAC/cAppliMICMAC_Result2.cpp
        applis/MICMAC/cBasculeMnt.cpp
        applis/MICMAC/cCaracOfDeZoom.cpp
        applis/MICMAC/cEtapeMecComp.cpp
        applis/MICMAC/cFilePx.cpp
        applis/MICMAC/cGBV2_ProgDynOptimiseur.cpp
        applis/MICMAC/cGeomImage.cpp
        applis/MICMAC/cGeomXXX.cpp
        applis/MICMAC/cLoadedImage.cpp
        applis/MICMAC/cMicMacVisu.cpp
        applis/MICMAC/cModeleAnalytiqueComp.cpp
        applis/MICMAC/cNewProgDyn.cpp
        applis/MICMAC/cOptimisationDiff.cpp
        applis/MICMAC/cOrientationCon.cpp
        applis/MICMAC/cOrientationGrille.cpp
        applis/MICMAC/cOrientationRTO.cpp
        applis/MICMAC/cParamMICMAC.cpp
        applis/MICMAC/cPriseDeVue.cpp
        applis/MICMAC/cProgDynOptimiseur.cpp
        applis/MICMAC/cSimulation.cpp
        applis/MICMAC/cStatNDistrib.cpp
        applis/MICMAC/cStdTiffModuleImageLoader.cpp
        applis/MICMAC/cSurfaceOptimiseur.cpp
        applis/MICMAC/uti_MICMAC.cpp
	applis/MICMAC/OrthoLocAnam.cpp
	)
	target_link_libraries(MICMACSaisieLiaisons elise ${X11_LIBRARIES})
        INSTALL(TARGETS MICMACSaisieLiaisons
                RUNTIME DESTINATION bin
        )


ENDIF()

#Outils uti_files
add_executable(BatchFDC applis/uti_files/BatchFDC.cpp)
target_link_libraries(BatchFDC elise ${X11_LIBRARIES})

add_executable(MapCmd applis/uti_files/MapCmd.cpp)
target_link_libraries(MapCmd elise ${X11_LIBRARIES})

# Apero
include_directories(
	${PROJECT_SOURCE_DIR}/applis/Apero
)
add_executable(Apero
applis/Apero/Apero.cpp
CodeGenere/File2String/Str_ParamApero.cpp
applis/Apero/PtsFlottant.cpp
applis/Apero/cAppliApero.cpp
applis/Apero/cPackMultipleLiaison.cpp
applis/Apero/Basculement.cpp
applis/Apero/Redressement.cpp
applis/Apero/cCalib.cpp
applis/Apero/cPackObsLiaison.cpp
applis/Apero/CarteProfExport.cpp
applis/Apero/Simule.cpp
applis/Apero/cChoixImagesMicMac.cpp
applis/Apero/cParamApero.cpp
applis/Apero/ExportNuage.cpp
applis/Apero/Tracage.cpp
applis/Apero/cCorrel.cpp
applis/Apero/cPonderateur.cpp
#applis/Apero/GenParamApero.cpp
applis/Apero/VisuMult.cpp
applis/Apero/cExport.cpp
applis/Apero/cPose.cpp
applis/Apero/GraphePose.cpp
applis/Apero/cAA_Compensation.cpp
#applis/Apero/cImageAppuis.cpp
applis/Apero/cSurfParam.cpp
#applis/Apero/ImProfAndL2App.cpp
applis/Apero/cAA_InitInconnues.cpp
applis/Apero/cLayersIm.cpp
applis/Apero/PowelOptimize.cpp
applis/Apero/cAA_LevenbergMarkard.cpp
applis/Apero/cPackAppuis.cpp
)

target_link_libraries(Apero elise ${X11_LIBRARIES})

INSTALL(TARGETS MICMAC MpDcraw Malt Dequant Tapioca Pastis Nuage2Ply Pasta Bascule MakeGrid TestCam Tapas AperiCloud ScaleIm ElDcraw to8Bits GrShade PastDevlop Devlop tiff_info BatchFDC MapCmd Apero
	RUNTIME DESTINATION bin
)

