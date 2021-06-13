add_executable(mm3d mm3d.cpp)

target_link_libraries(mm3d elise ${X11_LIBRARIES} ${WINSOCK2_LIBRARY} ${DYN_LIB_LIBRARY} ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ANN)

if(MINGW)
	#~ target_link_libraries(mm3d msvcr80 gettextlib.dll)
	target_link_libraries(mm3d msvcr100)
endif()

if(QT_ENABLED AND UNIX)
        target_link_libraries(mm3d dl)
endif()

if(${CUDA_ENABLED})
         target_link_libraries(mm3d ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt})
endif()

if(IgnSocle_FOUND)
    target_link_libraries(mm3d ${IgnSocle_LIBRARIES})

        if(PROJ4_FOUND)
                target_link_libraries(mm3d ${PROJ4_LIBRARIES})
        endif()

        if(GDAL_FOUND)
                target_link_libraries(mm3d ${GDAL_LIBRARIES})
        endif()
        if(ECWJ2_FOUND)
                target_link_libraries(mm3d ${ECWJ2_LIBRARIES})
        endif()
        if(Geos_FOUND)
                target_link_libraries(mm3d ${Geos_LIBRARIES})
        endif()
        if(TinyXML_FOUND)
                add_definitions( "-DTIXML_USE_STL" )
                target_link_libraries(mm3d ${TinyXML_LIBRARIES})
        endif()
endif()


if(KAKADU_FOUND)
        if(UNIX)
                target_link_libraries(mm3d pthread ${KAKADU_LIBRARIES})
        else()
                target_link_libraries(mm3d ${KAKADU_LIBRARIES})
        endif()
endif()

if(Boost_FOUND)
        target_link_libraries(mm3d ${Boost_LIBRARIES} ${Boost_THREADAPI})
endif()

if(WITH_GRAPHVIZ)
    target_link_libraries(mm3d "${gvc_LIBRARIES}" -lcgraph)
endif()

install(TARGETS mm3d RUNTIME DESTINATION ${Install_Dir})

include(InstallRequiredSystemLibraries)

if(NOT ${BUILD_ONLY_ELISE_MM3D})
        add_executable(AperiCloud  AperiCloud.c)
        install(TARGETS AperiCloud RUNTIME DESTINATION ${Install_Dir})

        add_executable(Digeo  Digeo.c)
        install(TARGETS Digeo RUNTIME DESTINATION ${Install_Dir})

        add_executable(Apero  Apero.c)
        install(TARGETS Apero RUNTIME DESTINATION ${Install_Dir})

        add_executable(Apero2PMVS Apero2PMVS.c)
        install(TARGETS Apero2PMVS RUNTIME DESTINATION ${Install_Dir})

        add_executable(mmxv mmxv.c)
        install(TARGETS mmxv RUNTIME DESTINATION ${Install_Dir})

        add_executable(Bascule  Bascule.c)
        install(TARGETS Bascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(BatchFDC  BatchFDC.c)
        install(TARGETS BatchFDC RUNTIME DESTINATION ${Install_Dir})

        add_executable(CmpCalib  CmpCalib.c)
        install(TARGETS CmpCalib RUNTIME DESTINATION ${Install_Dir})

        add_executable(Dequant  Dequant.c)
        install(TARGETS Dequant RUNTIME DESTINATION ${Install_Dir})

        add_executable(vic  vic.c)
        install(TARGETS vic RUNTIME DESTINATION ${Install_Dir})

        add_executable(Devlop  Devlop.c)
        install(TARGETS Devlop RUNTIME DESTINATION ${Install_Dir})

        add_executable(Drunk Drunk.c)
        install(TARGETS Drunk RUNTIME DESTINATION ${Install_Dir})

        add_executable(ElDcraw  ElDcraw.c)
        install(TARGETS ElDcraw RUNTIME DESTINATION ${Install_Dir})

        add_executable(GCPBascule  GCPBascule.c)
        install(TARGETS GCPBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(CenterBascule  CenterBascule.c)
        install(TARGETS CenterBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(GIMMI GIMMI.c)
        install(TARGETS GIMMI RUNTIME DESTINATION ${Install_Dir})

        add_executable(GenXML2Cpp  GenXML2Cpp.c)
        install(TARGETS GenXML2Cpp RUNTIME DESTINATION ${Install_Dir})

        add_executable(Gri2Bin  Gri2Bin.c)
        install(TARGETS Gri2Bin RUNTIME DESTINATION ${Install_Dir})

        add_executable(GrShade  GrShade.c)
        install(TARGETS GrShade RUNTIME DESTINATION ${Install_Dir})

        add_executable(MakeGrid  MakeGrid.c)
        install(TARGETS MakeGrid RUNTIME DESTINATION ${Install_Dir})

        add_executable(Malt  Malt.c)
        install(TARGETS Malt RUNTIME DESTINATION ${Install_Dir})

        add_executable(MapCmd  MapCmd.c)
        install(TARGETS MapCmd RUNTIME DESTINATION ${Install_Dir})

        add_executable(MICMAC  MICMAC.c)
        install(TARGETS MICMAC RUNTIME DESTINATION ${Install_Dir})

        add_executable(MpDcraw  MpDcraw.c)
        install(TARGETS MpDcraw RUNTIME DESTINATION ${Install_Dir})

        add_executable(Nuage2Ply  Nuage2Ply.c)
        install(TARGETS Nuage2Ply RUNTIME DESTINATION ${Install_Dir})

        add_executable(Pasta  Pasta.c)
        install(TARGETS Pasta RUNTIME DESTINATION ${Install_Dir})

        add_executable(PastDevlop  PastDevlop.c)
        install(TARGETS PastDevlop RUNTIME DESTINATION ${Install_Dir})

        add_executable(Pastis  Pastis.c)
        install(TARGETS Pastis RUNTIME DESTINATION ${Install_Dir})

        add_executable(Porto  Porto.c)
        install(TARGETS Porto RUNTIME DESTINATION ${Install_Dir})

        add_executable(Reduc2MM  Reduc2MM.c)
        install(TARGETS Reduc2MM RUNTIME DESTINATION ${Install_Dir})

        add_executable(ReducHom  ReducHom.c)
        install(TARGETS ReducHom RUNTIME DESTINATION ${Install_Dir})

        add_executable(RepLocBascule  RepLocBascule.c)
        install(TARGETS RepLocBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(SBGlobBascule  SBGlobBascule.c)
        install(TARGETS SBGlobBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(ScaleIm  ScaleIm.c)
        install(TARGETS ScaleIm RUNTIME DESTINATION ${Install_Dir})

        add_executable(ScaleNuage  ScaleNuage.c)
        install(TARGETS ScaleNuage RUNTIME DESTINATION ${Install_Dir})

        add_executable(SysCoordPolyn  SysCoordPolyn.c)
        install(TARGETS SysCoordPolyn RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tapas  Tapas.c)
        install(TARGETS Tapas RUNTIME DESTINATION ${Install_Dir})

        add_executable(Campari  Campari.c)
        install(TARGETS Campari RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tapioca  Tapioca.c)
        install(TARGETS Tapioca RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tarama  Tarama.c)
        install(TARGETS Tarama RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tawny  Tawny.c)
        install(TARGETS Tawny RUNTIME DESTINATION ${Install_Dir})

        add_executable(TestCam  TestCam.c)
        install(TARGETS TestCam RUNTIME DESTINATION ${Install_Dir})

        add_executable(TestCmds  TestCmds.c)
        install(TARGETS TestCmds RUNTIME DESTINATION ${Install_Dir})

        add_executable(tiff_info  tiff_info.c)
        install(TARGETS tiff_info RUNTIME DESTINATION ${Install_Dir})

        add_executable(to8Bits  to8Bits.c)
        install(TARGETS to8Bits RUNTIME DESTINATION ${Install_Dir})

        add_executable(Vino  Vino.c)
        install(TARGETS Vino RUNTIME DESTINATION ${Install_Dir})

        # for x11 tools
        if (X11_FOUND)
                add_executable(SaisieAppuisInit  SaisieAppuisInit.c)
                install(TARGETS SaisieAppuisInit RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisieAppuisPredic  SaisieAppuisPredic.c)
                install(TARGETS SaisieAppuisPredic RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisieBasc  SaisieBasc.c)
                install(TARGETS SaisieBasc RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisieMasq  SaisieMasq.c)
                install(TARGETS SaisieMasq RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisiePts  SaisiePts.c)
                install(TARGETS SaisiePts RUNTIME DESTINATION ${Install_Dir})

                add_executable(SEL  SEL.c)
                install(TARGETS SEL RUNTIME DESTINATION ${Install_Dir})

                if(${WITH_ETALONPOLY})
                        # Etalonnage polygone
                        add_executable(EPExeCompens  EPExeCompens.c)
                        install(TARGETS EPExeCompens RUNTIME DESTINATION ${Install_Dir})

                        add_executable(ScriptCalib  ScriptCalib.c)
                        install(TARGETS ScriptCalib RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeCatImSaisie  EPExeCatImSaisie.c)
                        install(TARGETS EPExeCatImSaisie RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeCalibFinale  EPExeCalibFinale.c)
                        install(TARGETS EPExeCalibFinale RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeCalibInit  EPExeCalibInit.c)
                        install(TARGETS EPExeCalibInit RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeConvertPolygone  EPExeConvertPolygone.c)
                        install(TARGETS EPExeConvertPolygone RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExePointeInitPolyg  EPExePointeInitPolyg.c)
                        install(TARGETS EPExePointeInitPolyg RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeRechCibleDRad  EPExeRechCibleDRad.c)
                        install(TARGETS EPExeRechCibleDRad RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeRechCibleInit EPExeRechCibleInit.c)
                        install(TARGETS EPExeRechCibleInit RUNTIME DESTINATION ${Install_Dir})
                endif()
        endif()
endif()
