add_executable(mm3d mm3d.cpp)
target_link_libraries(mm3d elise ${X11_LIBRARIES} ${X11_LIBRARIES} ${KAKADU_LIBRARIES})
INSTALL(TARGETS mm3d RUNTIME DESTINATION ${Install_Dir})

add_executable(AperiCloud  AperiCloud.c)
INSTALL(TARGETS AperiCloud RUNTIME DESTINATION ${Install_Dir})

add_executable(Apero  Apero.c)
INSTALL(TARGETS Apero RUNTIME DESTINATION ${Install_Dir})

add_executable(Bascule  Bascule.c)
INSTALL(TARGETS Bascule RUNTIME DESTINATION ${Install_Dir})

add_executable(BatchFDC  BatchFDC.c)
INSTALL(TARGETS BatchFDC RUNTIME DESTINATION ${Install_Dir})

add_executable(CmpCalib  CmpCalib.c)
INSTALL(TARGETS CmpCalib RUNTIME DESTINATION ${Install_Dir})

add_executable(Dequant  Dequant.c)
INSTALL(TARGETS Dequant RUNTIME DESTINATION ${Install_Dir})

add_executable(Devlop  Devlop.c)
INSTALL(TARGETS Devlop RUNTIME DESTINATION ${Install_Dir})

add_executable(ElDcraw  ElDcraw.c)
INSTALL(TARGETS ElDcraw RUNTIME DESTINATION ${Install_Dir})

add_executable(GCPBascule  GCPBascule.c)
INSTALL(TARGETS GCPBascule RUNTIME DESTINATION ${Install_Dir})

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

add_executable(Tapas  Tapas.c)
INSTALL(TARGETS Tapas RUNTIME DESTINATION ${Install_Dir})

add_executable(Tapioca  Tapioca.c)
INSTALL(TARGETS Tapioca RUNTIME DESTINATION ${Install_Dir})

add_executable(Tarama  Tarama.c)
INSTALL(TARGETS Tarama RUNTIME DESTINATION ${Install_Dir})

add_executable(Tawny  Tawny.c)
INSTALL(TARGETS Tawny RUNTIME DESTINATION ${Install_Dir})

add_executable(TestCam  TestCam.c)
INSTALL(TARGETS TestCam RUNTIME DESTINATION ${Install_Dir})

add_executable(tiff_info  tiff_info.c)
INSTALL(TARGETS tiff_info RUNTIME DESTINATION ${Install_Dir})

add_executable(to8Bits  to8Bits.c)
INSTALL(TARGETS to8Bits RUNTIME DESTINATION ${Install_Dir})
