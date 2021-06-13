cmake_minimum_required(VERSION 2.8)


find_package(OpenGL REQUIRED)


set(GIMM_SRC ${GIMMI_DIR}/Aperi.cpp
             ${GIMMI_DIR}/Bascule.cpp
             ${GIMMI_DIR}/C3dc.cpp
             ${GIMMI_DIR}/Campari.cpp
             ${GIMMI_DIR}/Cmd.cpp
             ${GIMMI_DIR}/CropRPC.cpp
             ${GIMMI_DIR}/ChSys.cpp
             ${GIMMI_DIR}/Console.cpp
             ${GIMMI_DIR}/Convert2GenBundle.cpp
             ${GIMMI_DIR}/SaisiePts.cpp
             ${GIMMI_DIR}/GrShade.cpp
             ${GIMMI_DIR}/main.cpp
             ${GIMMI_DIR}/MainWindow.cpp
             ${GIMMI_DIR}/Malt.cpp
             ${GIMMI_DIR}/Meshlab.cpp
             ${GIMMI_DIR}/MM2DPosSism.cpp
             ${GIMMI_DIR}/NewProject.cpp
             ${GIMMI_DIR}/SaisieMasq.cpp
             ${GIMMI_DIR}/Sel.cpp
             ${GIMMI_DIR}/Tapas.cpp
             ${GIMMI_DIR}/Tapioca.cpp
             ${GIMMI_DIR}/Tarama.cpp
)

set(GIMM_HEADER 
                 ${GIMMI_DIR}/Aperi.h
                 ${GIMMI_DIR}/Bascule.h
                 ${GIMMI_DIR}/C3dc.h
                 ${GIMMI_DIR}/Campari.h
                 ${GIMMI_DIR}/Cmd.h
                 ${GIMMI_DIR}/CropRPC.h
                 ${GIMMI_DIR}/ChSys.h
                 ${GIMMI_DIR}/GrShade.h
                 ${GIMMI_DIR}/Console.h
                 ${GIMMI_DIR}/Convert2GenBundle.h
                 ${GIMMI_DIR}/SaisiePts.h
                 ${GIMMI_DIR}/MainWindow.h
                 ${GIMMI_DIR}/Malt.h
                 ${GIMMI_DIR}/Meshlab.h
                 ${GIMMI_DIR}/MM2DPosSism.h
                 ${GIMMI_DIR}/SaisieMasq.h
                 ${GIMMI_DIR}/Sel.h
                 ${GIMMI_DIR}/Tapas.h
                 ${GIMMI_DIR}/Tapioca.h
                 ${GIMMI_DIR}/Tarama.h
)


QT5_WRAP_CPP(GIMM_HEADER_MOC ${GIMM_HEADER} )

ADD_EXECUTABLE(GIMMI_ ${GIMM_SRC} ${GIMM_HEADER_MOC})

TARGET_LINK_LIBRARIES(GIMMI_ elise ${OPENGL_gl_LIBRARY} ${OPENGL_LIBRARIES} ${QT_LIBRARIES} ${X11_LIBRARIES} ${KAKADU_LIBRARIES} ANN)

INSTALL(TARGETS GIMMI_ RUNTIME DESTINATION ${Install_Dir})
