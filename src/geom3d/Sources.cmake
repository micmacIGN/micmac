set(Geom3d_Src_Files
	${GEOM3D_DIR}/NuageToGrid.cpp
	${GEOM3D_DIR}/cElNuage3DMaille.cpp
	${GEOM3D_DIR}/cElNuageLaser.cpp
	${GEOM3D_DIR}/cGridNuageP3D.cpp
	${GEOM3D_DIR}/cImplemElNuage3DMaille.cpp
	${GEOM3D_DIR}/cMailageSphere.cpp
	${GEOM3D_DIR}/cZBuffer.cpp
	${GEOM3D_DIR}/geo3basic.cpp
	${GEOM3D_DIR}/cMesh3D.cpp
	${GEOM3D_DIR}/cMasq3D.cpp
	${GEOM3D_DIR}/PbTopoNuage.cpp
)

source_group(Geom3d FILES ${Geom3d_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Geom3d_Src_Files}
)
