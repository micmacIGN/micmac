add_executable(mm3d mm3d.cpp)
target_link_libraries(mm3d elise ${X11_LIBRARIES} ${WINSOCK2_LIBRARY} ${DYN_LIB_LIBRARY} ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ANN)

if(${CUDA_ENABLED})
         target_link_libraries(mm3d ${libStatGpGpuTools} ${libStatGpGpuInterfMicMac} ${libStatGpGpuOpt})
endif()

if(IgnSocle_FOUND)
    target_link_libraries(mm3d ${IgnSocle_LIBRARIES})

        IF ( PROJ4_FOUND )
                target_link_libraries(mm3d ${PROJ4_LIBRARIES})
        ENDIF()

        IF ( GDAL_FOUND )
                target_link_libraries(mm3d ${GDAL_LIBRARIES})
        ENDIF()
        IF ( ECWJ2_FOUND )
                target_link_libraries(mm3d ${ECWJ2_LIBRARIES})
        ENDIF()
        IF ( Geos_FOUND )
                target_link_libraries(mm3d ${Geos_LIBRARIES})
        ENDIF()
        IF(TinyXML_FOUND)
                add_definitions( "-DTIXML_USE_STL" )
                target_link_libraries(mm3d ${TinyXML_LIBRARIES})
        ENDIF()
endif()


if(KAKADU_FOUND)
        if(UNIX)
          target_link_libraries(mm3d pthread  ${KAKADU_LIBRARIES})
        else()
                target_link_libraries(mm3d  ${KAKADU_LIBRARIES})
        endif()
endif()


if(Boost_FOUND)
    target_link_libraries(mm3d ${Boost_LIBRARIES} ${Boost_THREADAPI})
endif()



INSTALL(TARGETS mm3d RUNTIME DESTINATION ${Install_Dir})


if (NOT ${BUILD_ONLY_ELISE_MM3D})
        add_executable(AperiCloud  AperiCloud.c)
        INSTALL(TARGETS AperiCloud RUNTIME DESTINATION ${Install_Dir})

        add_executable(Digeo  Digeo.c)
        INSTALL(TARGETS Digeo RUNTIME DESTINATION ${Install_Dir})

        add_executable(Apero  Apero.c)
        INSTALL(TARGETS Apero RUNTIME DESTINATION ${Install_Dir})

        add_executable(Apero2PMVS Apero2PMVS.c)
        INSTALL(TARGETS Apero2PMVS RUNTIME DESTINATION ${Install_Dir})

        add_executable(mmxv mmxv.c)
        INSTALL(TARGETS mmxv RUNTIME DESTINATION ${Install_Dir})

        add_executable(Bascule  Bascule.c)
        INSTALL(TARGETS Bascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(BatchFDC  BatchFDC.c)
        INSTALL(TARGETS BatchFDC RUNTIME DESTINATION ${Install_Dir})

        add_executable(CmpCalib  CmpCalib.c)
        INSTALL(TARGETS CmpCalib RUNTIME DESTINATION ${Install_Dir})

        add_executable(Dequant  Dequant.c)
        INSTALL(TARGETS Dequant RUNTIME DESTINATION ${Install_Dir})

        add_executable(vic  vic.c)
        INSTALL(TARGETS vic RUNTIME DESTINATION ${Install_Dir})

        add_executable(Devlop  Devlop.c)
        INSTALL(TARGETS Devlop RUNTIME DESTINATION ${Install_Dir})

        add_executable(Drunk Drunk.c)
        INSTALL(TARGETS Drunk RUNTIME DESTINATION ${Install_Dir})

        add_executable(ElDcraw  ElDcraw.c)
        INSTALL(TARGETS ElDcraw RUNTIME DESTINATION ${Install_Dir})

        add_executable(GCPBascule  GCPBascule.c)
        INSTALL(TARGETS GCPBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(CenterBascule  CenterBascule.c)
        INSTALL(TARGETS CenterBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(GenXML2Cpp  GenXML2Cpp.c)
        INSTALL(TARGETS GenXML2Cpp RUNTIME DESTINATION ${Install_Dir})

        add_executable(Gri2Bin  Gri2Bin.c)
        INSTALL(TARGETS Gri2Bin RUNTIME DESTINATION ${Install_Dir})

        add_executable(GrShade  GrShade.c)
        INSTALL(TARGETS GrShade RUNTIME DESTINATION ${Install_Dir})

        add_executable(MakeGrid  MakeGrid.c)
        INSTALL(TARGETS MakeGrid RUNTIME DESTINATION ${Install_Dir})

        add_executable(Malt  Malt.c)
        INSTALL(TARGETS Malt RUNTIME DESTINATION ${Install_Dir})

        add_executable(MapCmd  MapCmd.c)
        INSTALL(TARGETS MapCmd RUNTIME DESTINATION ${Install_Dir})

        add_executable(MICMAC  MICMAC.c)
        INSTALL(TARGETS MICMAC RUNTIME DESTINATION ${Install_Dir})

        add_executable(MpDcraw  MpDcraw.c)
        INSTALL(TARGETS MpDcraw RUNTIME DESTINATION ${Install_Dir})

        add_executable(Nuage2Ply  Nuage2Ply.c)
        INSTALL(TARGETS Nuage2Ply RUNTIME DESTINATION ${Install_Dir})

        add_executable(Pasta  Pasta.c)
        INSTALL(TARGETS Pasta RUNTIME DESTINATION ${Install_Dir})

        add_executable(PastDevlop  PastDevlop.c)
        INSTALL(TARGETS PastDevlop RUNTIME DESTINATION ${Install_Dir})

        add_executable(Pastis  Pastis.c)
        INSTALL(TARGETS Pastis RUNTIME DESTINATION ${Install_Dir})

        add_executable(Porto  Porto.c)
        INSTALL(TARGETS Porto RUNTIME DESTINATION ${Install_Dir})

        add_executable(Reduc2MM  Reduc2MM.c)
        INSTALL(TARGETS Reduc2MM RUNTIME DESTINATION ${Install_Dir})

        add_executable(ReducHom  ReducHom.c)
        INSTALL(TARGETS ReducHom RUNTIME DESTINATION ${Install_Dir})

        add_executable(RepLocBascule  RepLocBascule.c)
        INSTALL(TARGETS RepLocBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(SBGlobBascule  SBGlobBascule.c)
        INSTALL(TARGETS SBGlobBascule RUNTIME DESTINATION ${Install_Dir})

        add_executable(ScaleIm  ScaleIm.c)
        INSTALL(TARGETS ScaleIm RUNTIME DESTINATION ${Install_Dir})

        add_executable(ScaleNuage  ScaleNuage.c)
        INSTALL(TARGETS ScaleNuage RUNTIME DESTINATION ${Install_Dir})

        add_executable(SysCoordPolyn  SysCoordPolyn.c)
        INSTALL(TARGETS SysCoordPolyn RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tapas  Tapas.c)
        INSTALL(TARGETS Tapas RUNTIME DESTINATION ${Install_Dir})

        add_executable(Campari  Campari.c)
        INSTALL(TARGETS Campari RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tapioca  Tapioca.c)
        INSTALL(TARGETS Tapioca RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tarama  Tarama.c)
        INSTALL(TARGETS Tarama RUNTIME DESTINATION ${Install_Dir})

        add_executable(Tawny  Tawny.c)
        INSTALL(TARGETS Tawny RUNTIME DESTINATION ${Install_Dir})

        add_executable(TestCam  TestCam.c)
        INSTALL(TARGETS TestCam RUNTIME DESTINATION ${Install_Dir})

        add_executable(TestCmds  TestCmds.c)
        INSTALL(TARGETS TestCmds RUNTIME DESTINATION ${Install_Dir})

        add_executable(tiff_info  tiff_info.c)
        INSTALL(TARGETS tiff_info RUNTIME DESTINATION ${Install_Dir})

        add_executable(to8Bits  to8Bits.c)
        INSTALL(TARGETS to8Bits RUNTIME DESTINATION ${Install_Dir})

        # for x11 tools
        if (X11_FOUND)
                add_executable(SaisieAppuisInit  SaisieAppuisInit.c)
                INSTALL(TARGETS SaisieAppuisInit RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisieAppuisPredic  SaisieAppuisPredic.c)
                INSTALL(TARGETS SaisieAppuisPredic RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisieBasc  SaisieBasc.c)
                INSTALL(TARGETS SaisieBasc RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisieMasq  SaisieMasq.c)
                INSTALL(TARGETS SaisieMasq RUNTIME DESTINATION ${Install_Dir})

                add_executable(SaisiePts  SaisiePts.c)
                INSTALL(TARGETS SaisiePts RUNTIME DESTINATION ${Install_Dir})

                add_executable(SEL  SEL.c)
                INSTALL(TARGETS SEL RUNTIME DESTINATION ${Install_Dir})

                if(${WITH_ETALONPOLY})
                        # Etalonnage polygone
                        add_executable(EPExeCompens  EPExeCompens.c)
                        INSTALL(TARGETS EPExeCompens RUNTIME DESTINATION ${Install_Dir})

                        add_executable(ScriptCalib  ScriptCalib.c)
                        INSTALL(TARGETS ScriptCalib RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeCatImSaisie  EPExeCatImSaisie.c)
                        INSTALL(TARGETS EPExeCatImSaisie RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeCalibFinale  EPExeCalibFinale.c)
                        INSTALL(TARGETS EPExeCalibFinale RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeCalibInit  EPExeCalibInit.c)
                        INSTALL(TARGETS EPExeCalibInit RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeConvertPolygone  EPExeConvertPolygone.c)
                        INSTALL(TARGETS EPExeConvertPolygone RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExePointeInitPolyg  EPExePointeInitPolyg.c)
                        INSTALL(TARGETS EPExePointeInitPolyg RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeRechCibleDRad  EPExeRechCibleDRad.c)
                        INSTALL(TARGETS EPExeRechCibleDRad RUNTIME DESTINATION ${Install_Dir})

                        add_executable(EPExeRechCibleInit EPExeRechCibleInit.c)
                        INSTALL(TARGETS EPExeRechCibleInit RUNTIME DESTINATION ${Install_Dir})
                endif()
        endif()
endif()
