cmake_minimum_required(VERSION 2.8.12)
project(saisieQT)

if(POLICY CMP0020)
	cmake_policy(SET CMP0020 NEW) #windows specific: Automatically link Qt executables to qtmain
endif()

set(saisieQT_SRCS main/saisieQT_main.cpp
                  main/saisieMasqQT_main.cpp
                  main/saisieAppuisInitQT_main.cpp
                  main/saisieAppuisPredicQT_main.cpp
                  main/saisieBascQT_main.cpp
                  main/saisieBoxQT_main.cpp)

foreach(src ${saisieQT_SRCS})
	list(APPEND saisieqt_files "${CMAKE_SOURCE_DIR}/src/saisieQT/${src}")
endforeach()
set(saisieqt_files ${saisieqt_files} PARENT_SCOPE)

set(FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE} ${saisieQT_SRCS} PARENT_SCOPE)

if(UPDATE_TRANSLATIONS)
    set(translation_dir ${CMAKE_SOURCE_DIR}/include/qt/translations)
    #message ("translation_dir " ${translation_dir})

    set(languages fr es)
    set(language_ts_files)

    foreach(language ${languages})
        set(ts_file ${ts_files} ${translation_dir}/saisie_${language}.ts)
        set(language_ts_files ${language_ts_files} ${ts_file})
        SET_SOURCE_FILES_PROPERTIES(${ts_file} PROPERTIES OUTPUT_LOCATION ${translation_dir})
    endforeach(language)

    qt5_create_translation(QM_FILES ${FILES_TO_TRANSLATE} ${language_ts_files})
    #message("language ts files: " ${language_ts_files})

    add_custom_target(update_translations DEPENDS ${QM_FILES} ${language_ts_files})

    install(FILES ${QM_FILES} DESTINATION ${translation_dir})
endif()

if (WIN32)
	add_executable(SaisieQT WIN32 ${saisieQT_SRCS} ${QM_FILES})
	set_target_properties(SaisieQT PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
else()
    add_executable(SaisieQT ${saisieQT_SRCS} ${QM_FILES})
endif()

target_link_libraries(SaisieQT elise ${X11_LIBRARIES} ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ${KAKADU_LIBRARIES} ANN)
if (UNIX)
	target_link_libraries(SaisieQT dl pthread)
endif()

install(TARGETS SaisieQT DESTINATION ${Install_Dir})
