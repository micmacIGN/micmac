#ifndef API_MM3D_H
#define API_MM3D_H

#include "StdAfx.h"
#include "../src/uti_phgrm/NewOri/NewOri.h"
/**
@file
@brief New methods for python API and existing classes
**/

class CamStenope;
//-------------------- Nouvelles methodes ---------------------

//!internal usage
void mm3d_init();

//! Create CamStenope form a XML file
CamStenope *  CamOrientFromFile(std::string filename);

//! Create XML of an ideal Cam
void createIdealCamXML(double focale, Pt2dr aPP, Pt2di aSz, std::string oriName, std::string imgName, std::string idCam, ElRotation3D &orient, double prof, double rayonUtile);
//void createIdealCamXML(CamStenope * aCam, Pt2dr aPP, Pt2di aSz, std::string oriName, std::string imgName);

//! Convert a python 9-element list into a ElRotation3D
ElRotation3D list2rot(std::vector<double> l);

//! Convert a quaternion into a ElRotation3D
ElRotation3D quaternion2rot(double a, double b, double c, double d);

//! Convert a ElRotation3D into a python 9-element list
std::vector<double> rot2list(ElRotation3D &r);

std::vector<std::string> getFileSet(std::string dir, std::string pattern);

//! Get the list of triplets
cXml_TopoTriplet StdGetFromSI_Xml_TopoTriplet(const std::string & aNameFileObj);

//! Get the triplet orientations
cXml_Ori3ImInit  StdGetFromSI_Xml_Ori3ImInit(const std::string & aNameFileObj);

//! Bundle intersection
Pt3dr ElSeg3D_L2InterFaisceaux_wrapper(std::vector<ElSeg3D> aVSeg);

#endif //API_MM3D_H
