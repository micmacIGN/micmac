cmake_minimum_required(VERSION 2.8.1)
project(saisieQT)

IF ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    add_definitions(-DELISE_Darwin)
endif()

set(saisieQT_SRCS saisieQT_main.cpp
                  saisieMasqQT_main.cpp
                  saisieAppuisInitQT_main.cpp
                  saisieAppuisPredicQT_main.cpp
                  saisieBascQT_main.cpp
                  saisieBoxQT_main.cpp)

set (FILES_TO_TRANSLATE ${FILES_TO_TRANSLATE} ${saisieQT_SRCS} PARENT_SCOPE)
#message ("saisie srcs: " ${saisieQT_SRCS} )

if ( NOT ${qt_version} EQUAL 0 )

    set(translation_dir ${PROJECT_SOURCE_DIR}/include/qt/translations)
    #message ("translation_dir " ${translation_dir})

    set(languages fr es ar ru cn)
    set(language_ts_files)

    foreach(language ${languages})
        set(ts_file ${ts_files} ${translation_dir}/saisie_${language}.ts)
        set(language_ts_files ${language_ts_files} ${ts_file})
        SET_SOURCE_FILES_PROPERTIES(${ts_file} PROPERTIES OUTPUT_LOCATION ${translation_dir})
    endforeach(language)

    if ( ${qt_version} EQUAL 5 )

        if (UPDATE_TRANSLATIONS)
            qt5_create_translation(QM_FILES ${FILES_TO_TRANSLATE} ${language_ts_files})
            #message("language ts files: " ${language_ts_files})
            add_custom_target(update_translations DEPENDS ${qm_files} ${language_ts_files})
        else()
            qt5_add_translation(QM_FILES ${language_ts_files})
        endif()
    else()
        if (UPDATE_TRANSLATIONS)
            qt4_create_translation(QM_FILES ${FILES_TO_TRANSLATE} ${language_ts_files})
            add_custom_target(update_translations DEPENDS ${qm_files} ${language_ts_files})
        else()
            qt4_add_translation(QM_FILES ${language_ts_files})
        endif()
    endif()

    add_custom_target (translations_target DEPENDS ${QM_FILES})

    install(FILES ${QM_FILES} DESTINATION ${translation_dir})

    if ( ${qt_version} EQUAL 5 AND WIN32)
      add_executable(SaisieQT WIN32 ${saisieQT_SRCS} ${QM_FILES})
    else()
      add_executable(SaisieQT ${saisieQT_SRCS} ${QM_FILES})
    endif()

    target_link_libraries(SaisieQT elise ${X11_LIBRARIES})

    install(TARGETS SaisieQT DESTINATION ${Install_Dir})
endif()
