set(Poisson_Src_Files
    ${POISSON_DIR}/CmdLineParser.cpp
    ${POISSON_DIR}/Factor.cpp
    ${POISSON_DIR}/Geometry.cpp
    ${POISSON_DIR}/MarchingCubes.cpp
    ${POISSON_DIR}/MultiGridOctest.cpp
    ${POISSON_DIR}/ply.cpp
    ${POISSON_DIR}/plyfile.cpp
    ${POISSON_DIR}/Time.cpp
)

SOURCE_GROUP(Poisson FILES ${Poisson_Src_Files})

list( APPEND Elise_Src_Files ${Poisson_Src_Files})

