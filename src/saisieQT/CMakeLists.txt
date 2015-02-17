cmake_minimum_required(VERSION 2.8.1)
project(saisieQT)

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()

set(saisieQT_SRCS main/saisieQT_main.cpp
                  main/saisieMasqQT_main.cpp
                  main/saisieAppuisInitQT_main.cpp
                  main/saisieAppuisPredicQT_main.cpp
                  main/saisieBascQT_main.cpp
                  main/saisieBoxQT_main.cpp)

set (FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE} ${saisieQT_SRCS} PARENT_SCOPE)
#message ("saisie srcs: " ${saisieQT_SRCS} )

if ( NOT ${qt_version} EQUAL 0 )

    if (UPDATE_TRANSLATIONS)
        set(translation_dir ${CMAKE_SOURCE_DIR}/include/qt/translations)
        #message ("translation_dir " ${translation_dir})

        set(languages fr es)
        set(language_ts_files)

        foreach(language ${languages})
            set(ts_file ${ts_files} ${translation_dir}/saisie_${language}.ts)
            set(language_ts_files ${language_ts_files} ${ts_file})
            SET_SOURCE_FILES_PROPERTIES(${ts_file} PROPERTIES OUTPUT_LOCATION ${translation_dir})
        endforeach(language)

        if ( ${qt_version} EQUAL 5 )
                qt5_create_translation(QM_FILES ${FILES_TO_TRANSLATE} ${language_ts_files})
                #message("language ts files: " ${language_ts_files})
        else()
                qt4_create_translation(QM_FILES ${FILES_TO_TRANSLATE} ${language_ts_files})
        endif()

        add_custom_target(update_translations DEPENDS ${QM_FILES} ${language_ts_files})

        install(FILES ${QM_FILES} DESTINATION ${translation_dir})
    endif()

    if (WIN32)	
		if(${qt_version} EQUAL 5 AND POLICY CMP0020)
				cmake_policy(SET CMP0020 NEW)
		endif()		
		add_executable(SaisieQT WIN32 ${saisieQT_SRCS} ${QM_FILES})
    else()
      add_executable(SaisieQT ${saisieQT_SRCS} ${QM_FILES})
    endif()

    if ( ${qt_version} EQUAL 5 )
        target_link_libraries(SaisieQT elise ${Qt5Core_LIBRARIES} ${Qt5Gui_LIBRARIES} ${Qt5Xml_LIBRARIES} ${Qt5OpenGL_LIBRARIES} ${X11_LIBRARIES})
    else()
        target_link_libraries(SaisieQT elise ${X11_LIBRARIES})
    endif()

    install(TARGETS SaisieQT DESTINATION ${Install_Dir})
endif()
