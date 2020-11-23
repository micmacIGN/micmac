/*Header-MicMac-eLiSe-25/06/2007peroChImMM_main

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "StdAfx.h"
#include <unordered_map>


// --------------------------------------------------------------
// Function to compute intersection between 2 lines in a 3D space
// Input: lines L1 and L2 (for each reference point and vector)
// Output: distance between L1 and L2
// --------------------------------------------------------------
double distanceLinesInSpace(std::vector<double> P1, std::vector<double> V1, std::vector<double> P2, std::vector<double> V2){

    double dr[3] = {0,0,0};
    dr[0] = P1.at(0)-P2.at(0);
    dr[1] = P1.at(1)-P2.at(1);
    dr[2] = P1.at(2)-P2.at(2);

    double n[3] = {0,0,0};
    n[0] = V1.at(1)*V2.at(2) - V1.at(2)*V2.at(1);
    n[1] = V1.at(2)*V2.at(0) - V1.at(0)*V2.at(2);
    n[2] = V1.at(0)*V2.at(1) - V1.at(1)*V2.at(0);

    double norm = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);

    // Parallel lines
    if (norm == 0){

            double vnorm = sqrt(V1.at(0)*V1.at(0) + V1.at(1)*V1.at(1) + V1.at(2)*V1.at(2));

            double pn[3] = {0,0,0};
            pn[0] = dr[1]*V1.at(2) - dr[2]*V1.at(1);
            pn[1] = dr[2]*V1.at(0) - dr[0]*V1.at(2);
            pn[2] = dr[0]*V1.at(1) - dr[1]*V1.at(0);

            return sqrt(pn[0]*pn[0] + pn[1]*pn[1] + pn[2]*pn[2])/vnorm;

    }

    n[0] = n[0]/norm;
    n[1] = n[1]/norm;
    n[2] = n[2]/norm;

    return std::abs(n[0]*dr[0] + n[1]*dr[1] + n[2]*dr[2]);

}


// --------------------------------------------------------------
// Function to compute intersection between 2 lines in a 3D space
// Interface for distanceLinesInSpace with cXml_One3DLine object
// Input: lines L1 and L2 as cXml_One3DLine
// Output: distance between L1 and L2
// --------------------------------------------------------------
double distanceBtwLinesInSpace(cXml_One3DLine L1, cXml_One3DLine L2){
	
	std::vector<double> P1;
	std::vector<double> P2;
	std::vector<double> V1;
	std::vector<double> V2;
	
	P1.push_back(L1.Pt().x); P1.push_back(L1.Pt().y); P1.push_back(L1.Pt().z);
	P2.push_back(L2.Pt().x); P2.push_back(L2.Pt().y); P2.push_back(L2.Pt().z);																		 
	V1.push_back(L1.Vec().x); V1.push_back(L1.Vec().y); V1.push_back(L1.Vec().z);										   
	V2.push_back(L2.Vec().x); V2.push_back(L2.Vec().y); V2.push_back(L2.Vec().z);	
																		 
	return distanceLinesInSpace(P1, V1, P2, V2);
																		 
}


// --------------------------------------------------------------
// Function to transform SaisieAppuisInit output to Line3D
// --------------------------------------------------------------
int CPP_GCP2MeasureLine2D(int argc,char ** argv){
    // The argument need some global computation 
    MMD_InitArgcArgv(argc,argv);
    std::string  aNameFileGCP;
    std::string  aNameLine3GCP;
	
	double x1, x2, y1, y2;
	
	int line_number = 0;
	int image_number = 0;
	
	// ----------------------------------------------------------
	// Arguments
	// ----------------------------------------------------------
    ElInitArgMain
    (
        argc,argv,
           // Mandatory arg
        LArgMain()
                    << EAMC(aNameFileGCP,"File for GCP", eSAM_IsPatFile),
           // Mandatory arg
        LArgMain()
                    << EAM(aNameLine3GCP,"Out",true,"")
    );

    if (!EAMIsInit(&aNameLine3GCP))
    {
       aNameLine3GCP = "L2D_"+aNameFileGCP;
    }
	
	// ----------------------------------------------------------
	// Output structure
	// ----------------------------------------------------------
	cXml_OneMeasure3DLineInIm mesure;
	cXml_SetMeasure3DLineInOneIm mesureInImage;
	cXml_SetMeasureGlob3DLine mesureInAllImages;

	std::list< cXml_OneMeasure3DLineInIm > mesures;
	std::list< cXml_SetMeasure3DLineInOneIm > allMesures;
	
	mesureInImage.Measures() = mesures;
	mesureInAllImages.AllMeasures() = allMesures;
	

    cSetOfMesureAppuisFlottants aGCPDic = StdGetFromPCP(aNameFileGCP,SetOfMesureAppuisFlottants);

   // Pour each image
   for (const auto & aItIm : aGCPDic.MesureAppuiFlottant1Im()){
	   
      // we cast to vector 4 sort
      std::vector<cOneMesureAF1I> aVMes(aItIm.OneMesureAF1I().begin(),aItIm.OneMesureAF1I().end());
      std::sort(
         aVMes.begin(),
         aVMes.end(),
         [](const cOneMesureAF1I &aM1,const cOneMesureAF1I &aM2) 
         {return aM1.NamePt()<aM2.NamePt();}
      );
      int aNbM = (aVMes.size());
	  
      if ((aNbM%2)!=0){
          std::cout << "NameIm = " << aItIm.NameIm() << " " << aNbM<< "\n";
          ELISE_ASSERT(false,"Odd size of point ");
      }
	   
	  line_number += aNbM/2;
		
      for (int aKM=0 ; aKM<aNbM ; aKM+=2){
		  
          std::string aPref1,aPref2,aPost1,aPost2;

          SplitIn2ArroundCar(aVMes[aKM].NamePt()  ,'_',aPref1,aPost1,false);
          SplitIn2ArroundCar(aVMes[aKM+1].NamePt(),'_',aPref2,aPost2,false);
		 
          ELISE_ASSERT(aPref1==aPref2,"Prefix different");
          ELISE_ASSERT(aPost1=="1","Bad postfix");
          ELISE_ASSERT(aPost2=="2","Bad postfix");
		    
		  x1 = aVMes[aKM].PtIm().x;
		  y1 = aVMes[aKM].PtIm().y;
		  
		  x2 = aVMes[aKM+1].PtIm().x;
		  y2 = aVMes[aKM+1].PtIm().y;
		  		  
		  mesure.P1() = Pt2dr(x1,y1);
		  mesure.P2() = Pt2dr(x2,y2);
		  mesure.NameLine3D() = aPref1;
		  
		  mesureInImage.Measures().push_back(mesure);

      }
	  
	  if (aNbM > 0){
		  image_number += 1;
		  mesureInImage.NameIm() = aItIm.NameIm();
		  mesureInAllImages.AllMeasures().push_back(mesureInImage);
		  mesureInImage.Measures().clear();
	  	  std::cout << aItIm.NameIm() << " " << aNbM/2 << " line(s)" << std::endl;
	  }
    
   }
		
   MakeFileXML(mesureInAllImages, aNameLine3GCP);
	
   std::cout << "-------------------------------------------------------------------------------" << std::endl;	
   std::cout << aNameFileGCP << ": " << line_number << " line(s) found in " << image_number << " image(s) -> " << aNameLine3GCP <<  std::endl; 
   std::cout << "-------------------------------------------------------------------------------" << std::endl;	
	
   return EXIT_SUCCESS;
}


// --------------------------------------------------------------
// Computes line intersection of two 3D planes
// Input:
//  - Plane 1 [a1,b1,c1,d1] : a1x+b1y+c1z+d1=0
//  - Plane 2 [a2,b2,c2,d2] : a2x+b2y+c2z+d2=0
// --------------------------------------------------------------
// Output:
//  - 3D line [x,y,z,u,v,w]
//    with (x,y,z) a point and (u,v,w) a vector
// --------------------------------------------------------------
cXml_One3DLine planeIntersect(std::vector<double> line1, std::vector<double> line2){

    double ra = line1.at(0)/line2.at(0);
    double rb = line1.at(1)/line2.at(1);
    double rc = line1.at(2)/line2.at(2);

    double dab = std::abs(ra-rb);
    double dac = std::abs(ra-rc);

    double eps = 1e-10;

    // ---------------------------------------------------
    // Tests if planes are parallel
    // ---------------------------------------------------
    if ((dab<eps) && (dac<eps)){
		ELISE_ASSERT(false, "Parallel planes -> cannot compute intersection")
    }
	
	// ---------------------------------------------------
    // Computing interection
    // ---------------------------------------------------
	double det = line1.at(0)*line2.at(1) - line2.at(0)*line1.at(1);

	// Solution for z=0
	double x0 = (-line2.at(1)*line1.at(3)+line1.at(1)*line2.at(3))/det;
	double y0 = (+line2.at(0)*line1.at(3)-line1.at(0)*line2.at(3))/det;
	
	// Solution for z=1
	double x1 = (+line2.at(1)*(-line1.at(3)-line1.at(2))-line1.at(1)*(-line2.at(3)-line2.at(2)))/det;
	double y1 = (-line2.at(0)*(-line1.at(3)-line1.at(2))+line1.at(0)*(-line2.at(3)-line2.at(2)))/det;
	
	//double test1 = line1.at(0)*x0 + line1.at(1)*y0 + line1.at(2)*0 + line1.at(3);
	//double test2 = line2.at(0)*x0 + line2.at(1)*y0 + line2.at(2)*0 + line2.at(3);
	//double test3 = line1.at(0)*x1 + line1.at(1)*y1 + line1.at(2)*1 + line1.at(3);
	//double test4 = line2.at(0)*x1 + line2.at(1)*y1 + line2.at(2)*1 + line2.at(3);
	//std::cout << test1 << " " << test2 << " " << test3 << " " << test4 << std::endl;
	
	cXml_One3DLine intersection;
	
	double ux = x1-x0;
	double uy = y1-y0;
	double uz = 1;
	double nu = sqrt(ux*ux + uy*uy + uz*uz);
	ux /= nu; uy /= nu; uz /= nu;
	
	Pt3dr pref(x0, y0, 0);
	Pt3dr vect(ux, uy, uz);

	intersection.Pt() = pref;
	intersection.Vec() = vect;
	
    return intersection;

}


// --------------------------------------------------------------
// Function get plane equation from 2 bundles
// --------------------------------------------------------------
std::vector<double> PlaneFromBundles(ElSeg3D aSeg1, ElSeg3D aSeg2){
	
	// --------------------------------------------------------------
	// Computing "plane bundle"
	// --------------------------------------------------------------
	double ux = aSeg1.P1().x - aSeg1.P0().x;  double vx = aSeg2.P1().x - aSeg2.P0().x;
	double uy = aSeg1.P1().y - aSeg1.P0().y;  double vy = aSeg2.P1().y - aSeg2.P0().y;
	double uz = aSeg1.P1().z - aSeg1.P0().z;  double vz = aSeg2.P1().z - aSeg2.P0().z;
		
	// Plane normal vector
	double nx = uy*vz-uz*vy;
	double ny = uz*vx-ux*vz;
	double nz = ux*vy-uy*vx;
	double nn = sqrt(ux*ux + uy*uy + uz*uz);
	nx /= nn;
	ny /= nn;
	nz /= nn;
	
	// Plane equation
	double d = -(nx*aSeg1.P0().x + ny*aSeg1.P0().y + nz*aSeg1.P0().z);
	
	std::vector<double> parameters;
	
	parameters.push_back(nx);
	parameters.push_back(ny);
	parameters.push_back(nz);
	parameters.push_back(d);
	
	return parameters;
	
}

// --------------------------------------------------------------
// Function get unit vector from a bundle segment
// --------------------------------------------------------------
Pt3dr UnitVec(ElSeg3D seg){

	Pt3dr vec;
	vec.x = seg.P1().x - seg.P0().x;
	vec.y = seg.P1().y - seg.P0().y;
	vec.z = seg.P1().z - seg.P0().z;
	
	double norm = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
	
	vec.x = vec.x/norm;
	vec.y = vec.y/norm;
	vec.z = vec.z/norm;
	
	return vec;
	
}

// --------------------------------------------------------------
// Function to transform Seg3D to Line 3D
// --------------------------------------------------------------
cXml_One3DLine SegToLine(ElSeg3D seg){

	cXml_One3DLine output;
	
	output.Pt() = seg.P0();
	output.Vec() = UnitVec(seg);
	
	return output;

}

// --------------------------------------------------------------
// Function to transform Line3D measurements to Line3D in space
// --------------------------------------------------------------
int CPP_MeasureL2D2L3D(int argc,char ** argv){
	
	MMD_InitArgcArgv(argc,argv);
    std::string aNameLine2D;
    std::string aNameLine3D;
	std::string aOrientation;

	std::string line_name;
	std::string image_name;
	
	ElInitArgMain(argc,argv,
        LArgMain() << EAMC(aNameLine2D,"2D Line measurement xml file", eSAM_IsPatFile)
		           << EAMC(aOrientation,"Orientation directory"),
        LArgMain() << EAM(aNameLine3D,"3D Line output xml file",true,"")
    );
	
	if (!EAMIsInit(&aNameLine3D)){
       aNameLine3D = "L3D_"+aNameLine2D;
    }
	
	
	std::cout << "-----------------------------------------------------------------" << std::endl;
	
	// Orientation name correction
	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
	StdCorrecNameOrient(aOrientation, anICNM->Dir());
	
	cXml_SetMeasureGlob3DLine mesures_lines = StdGetFromSI(aNameLine2D, Xml_SetMeasureGlob3DLine);
	std::vector<cXml_SetMeasure3DLineInOneIm> MESURES{std::begin(mesures_lines.AllMeasures()), std::end(mesures_lines.AllMeasures()) };

	
	std::vector<std::string> LINE_NAMES;
	std::unordered_map<std::string, std::vector<ElSeg3D>> BUNDLES;
	
	std::list<cXml_One3DLine> lines_3d;
	cXml_Set3DLine RESULTS;
	RESULTS.AllLines() = lines_3d;
	

	// Loop on images
	for (unsigned i=0; i<MESURES.size(); i++){
		
		image_name = MESURES.at(i).NameIm();
		std::vector<cXml_OneMeasure3DLineInIm> LINES{std::begin(MESURES.at(i).Measures()), std::end(MESURES.at(i).Measures())};
		std::cout << "IMAGE " << image_name << " " << LINES.size() << " LINE MEASUREMENTS" << std::endl;
		
		// Image orientation
		CamStenope * aCam = CamOrientGenFromFile("Ori-"+aOrientation+"/Orientation-"+image_name+".xml", anICNM);
		
		// Loop on lines
		for (unsigned j=0; j<LINES.size(); j++){
			
			line_name = LINES.at(j).NameLine3D();	
			
			ElSeg3D aSeg1 = aCam->Capteur2RayTer(LINES.at(j).P1());
			ElSeg3D aSeg2 = aCam->Capteur2RayTer(LINES.at(j).P2());
			
			// If line is encountered for the 1st time
			if (BUNDLES.find(line_name) == BUNDLES.end()){
				std::vector<ElSeg3D> new_line_bundle;
				BUNDLES[line_name] = new_line_bundle;
				LINE_NAMES.push_back(line_name);
			} 
			
			// Adding plane equation obs to 3D line
			BUNDLES[line_name].push_back(aSeg1);
			BUNDLES[line_name].push_back(aSeg2);
			
		}
	}
	
	
	double total_rmse = 0;
	
	// Resolving 3D lines
	for (unsigned i=0; i<LINE_NAMES.size(); i++){
		
		std::cout << "-----------------------------------------------------------------" << std::endl;	
		
		std::vector<ElSeg3D> bundles = BUNDLES[LINE_NAMES.at(i)];
		std::cout << bundles.size() << " bundle observations for 3D line " << LINE_NAMES.at(i) << std::endl;
		if (bundles.size() < 4){
			std::cout << "Warning: cannot resolve line " << LINE_NAMES.at(i) << " (insufficient number of observations)" << std::endl;
			continue;
		}
		
		std::vector<double> plane1;
		std::vector<double> plane2;
		
		double jmin = 0;
		double kmin = 0;
		double rmse_min = pow(10,100);
			
		for (unsigned j=0; j<bundles.size()-2; j+=2){
			for (unsigned k=j+2; k<bundles.size(); k+=2){
				
				// Computing bundle planes
				plane1 = PlaneFromBundles(bundles.at(j), bundles.at(j+1));
		 		plane2 = PlaneFromBundles(bundles.at(k), bundles.at(k+1));
		
				// Computing intersection
				cXml_One3DLine output = planeIntersect(plane1, plane2);
				std::cout << MESURES.at((int)(j/2)).NameIm() << " " << MESURES.at((int)(k/2)).NameIm();
				
				// -------------------------------------------------------
				// Cross-validation of intersection
				// -------------------------------------------------------
				double rmse = 0;
				for (unsigned l=0; l<bundles.size(); l++){
					if ((l == j) || (l == j+1) || (l == k) || (l == k+1)) continue;			
					cXml_One3DLine control_line = SegToLine(bundles.at(l));
					double residu = distanceBtwLinesInSpace(output, control_line);
					rmse += residu*residu;
				}
 				if (bundles.size() == 4){
					rmse = 0;
				}else{
					rmse = sqrt(rmse/(bundles.size()-4));
				}
				
				std::cout << "  RMSE = " << rmse << std::endl;
				
				if (rmse < rmse_min){
					rmse_min = rmse;
					jmin = j;
					kmin = k;
				}
				
				// -------------------------------------------------------
		
			}	
		
		}
		
		// Registering 3D line
		plane1 = PlaneFromBundles(bundles.at(jmin), bundles.at(jmin+1));
		plane2 = PlaneFromBundles(bundles.at(kmin), bundles.at(kmin+1));
		
		cXml_One3DLine output = planeIntersect(plane1, plane2);

		output.NameLine3D() = LINE_NAMES.at(i);
		RESULTS.AllLines().push_back(output);
		
		std::cout << "MIN: " << MESURES.at((int)(jmin/2)).NameIm() << " " << MESURES.at((int)(kmin/2)).NameIm();
		std::cout << "  RMSE = " << rmse_min << std::endl;
		total_rmse += rmse_min*rmse_min;
		
	}
	
	MakeFileXML(RESULTS, aNameLine3D);
	
	total_rmse = sqrt(total_rmse/RESULTS.AllLines().size());
			
	std::cout << "==================================================================" << std::endl;	
	std::cout << "Output: " << RESULTS.AllLines().size() << " line(s) written in " << aNameLine3D << std::endl;
	std::cout << "Total RMSE = " << total_rmse << std::endl;
	std::cout << "==================================================================" << std::endl;
	
	return 0;
}

// --------------------------------------------------------------
// Function to transform Line3D in space to ply file
// --------------------------------------------------------------
int CPP_L3D2Ply(int argc,char ** argv){

	MMD_InitArgcArgv(argc,argv);
    std::string aNameLine3D;
    std::string aNameLinePly;
	std::string nb_points;
	std::string length;
	
	ElInitArgMain(argc,argv,
        LArgMain() << EAMC(aNameLine3D,"3D Lines xml file", eSAM_IsPatFile),
        LArgMain() << EAM(aNameLinePly,"Out",true,"Ply output file")
				   << EAM(length,      "Length",true,"Length of line")
			       << EAM(nb_points,   "Nb",true,"Number of points")
    );
	
	if (!EAMIsInit(&aNameLinePly)){
       aNameLinePly = aNameLine3D+".ply";
    }
	
	if (!EAMIsInit(&nb_points)){
       nb_points = "100000";
    }
	
	
	if (!EAMIsInit(&length)){
       length = "500";
    }
	
	int nb = std::stoi(nb_points);
	double lgth = std::stod(length);
	
	// Loading 3D lines from xml
	cXml_Set3DLine LINES = StdGetFromSI(aNameLine3D, Xml_Set3DLine);
	
	std::cout << LINES.AllLines().size() << " lines loaded from " << aNameLine3D << std::endl;


	
	// Writing 3D linesto ply
	cPlyCloud cloud;
	std::vector<cXml_One3DLine> LINE_VEC{std::begin(LINES.AllLines()), std::end(LINES.AllLines()) };
	for (unsigned i=0; i<LINE_VEC.size(); i++){
		
		std::cout << "LINE " << LINE_VEC.at(i).NameLine3D();
		
		Pt3dr P = LINE_VEC.at(i).Pt();
		Pt3dr V = LINE_VEC.at(i).Vec();
		
		double p1x = P.x - lgth*V.x;
		double p1y = P.y - lgth*V.y;
		double p1z = P.z - lgth*V.z;
		
		double p2x = P.x + lgth*V.x;
		double p2y = P.y + lgth*V.y;
		double p2z = P.z + lgth*V.z;
		
		Pt3dr p1 = Pt3dr(p1x, p1y, p1z);
		Pt3dr p2 = Pt3dr(p2x, p2y, p2z);
		
		cloud.AddSeg(cPlyCloud::RandomColor(), p1, p2, nb);
	
		std::cout << "  Ok" << std::endl;
		
	}
	
	cloud.PutFile(aNameLinePly);
	
		
	std::cout << "-------------------------------------------------------------" << std::endl;
	std::cout << "File " << aNameLinePly << " written with success" << std::endl;
	std::cout << "-------------------------------------------------------------" << std::endl;
	
	return 0;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
